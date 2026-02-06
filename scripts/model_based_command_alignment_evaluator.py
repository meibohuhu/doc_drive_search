"""
Model-based Command Alignment Evaluator using Qwen2.5-VL-7B
mh 20260202:
使用Qwen2.5-VL-7B视觉语言模型评估预测的waypoints是否与navigation command对齐
"""

import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    try:
        from transformers import Gemma2ForConditionalGeneration as Gemma3ForConditionalGeneration
    except ImportError:
        Gemma3ForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # process_vision_info 只在 Qwen 模型中使用，Gemma-3 不需要
    process_vision_info = None

import re


class ModelBasedCommandAlignmentEvaluator:
    """
    使用视觉语言模型评估模型预测的waypoints是否与navigation command对齐
    支持Qwen2.5-VL、Qwen3-VL和Gemma-3模型
    """
    
    # Command定义 (与agent_simlingo.py中的map_command一致)
    COMMAND_MAP = {
        1: 'go left at the next intersection',
        2: 'go right at the next intersection', 
        3: 'go straight at the next intersection',
        4: 'follow the road',
        5: 'do a lane change to the left',
        6: 'do a lane change to the right',
    }
    
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_new_tokens: int = 200,
                 model_type: str = "auto",
                 output_file: Optional[str] = None,
                 save_interval: int = 5,
                 verbose_prompt: bool = False):
        """
        Args:
            model_name: Qwen模型名称（支持Qwen2.5-VL和Qwen3-VL）
            device: 运行设备 ('cuda' or 'cpu')
            max_new_tokens: 生成的最大token数
            model_type: 模型类型 ('auto', 'qwen2.5', 'qwen3')，auto会根据model_name自动检测
            output_file: 输出JSON文件路径（可选），如果提供则每save_interval个结果自动保存
            save_interval: 每多少个结果保存一次（默认5）
            verbose_prompt: 是否打印使用的prompt（默认False）
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        
        # 自动检测模型类型
        if model_type == "auto":
            model_name_lower = model_name.lower()
            if "gemma" in model_name_lower or "gemma-3" in model_name_lower or "gemma3" in model_name_lower:
                model_type = "gemma3"
            elif "qwen3" in model_name_lower:
                model_type = "qwen3"
            elif "qwen2.5" in model_name_lower or "qwen2" in model_name_lower:
                model_type = "qwen2.5"
            else:
                # 默认尝试Qwen2.5
                model_type = "qwen2.5"
        
        print(f"[INFO] Detected model type: {model_type} from model name: {model_name}", flush=True)
        
        self.model_type = model_type
        
        model_type_name = "Gemma-3" if model_type == "gemma3" else "Qwen"
        print(f"[INFO] Loading {model_type_name} model: {model_name} (type: {model_type}) on {device}...", flush=True)
        
        # 加载模型和processor
        try:
            # 根据模型类型选择加载方式
            if model_type == "gemma3":
                # Gemma 3 使用 AutoTokenizer 和 Gemma3ForConditionalGeneration
                # 注意：Gemma 3 支持多模态，但 tokenizer 可能不支持图像处理
                # 如果 AutoProcessor 失败，回退到 AutoTokenizer
                from transformers import AutoTokenizer
                if Gemma3ForConditionalGeneration is None:
                    raise ImportError("Gemma3ForConditionalGeneration not available. Please update transformers: pip install --upgrade transformers")
                
                # 尝试使用 AutoProcessor（如果支持图像）
                try:
                    self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                    print(f"[INFO] Using AutoProcessor for Gemma-3", flush=True)
                except Exception as e:
                    # 如果 AutoProcessor 失败，使用 AutoTokenizer
                    print(f"[WARNING] AutoProcessor failed, using AutoTokenizer: {e}", flush=True)
                    self.processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto" if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else device,
                    trust_remote_code=True
                )
            elif model_type == "qwen3":
                # Qwen 3 模型使用 AutoProcessor
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                if Qwen3VLForConditionalGeneration is None:
                    raise ImportError("Qwen3VLForConditionalGeneration not available. Please update transformers: pip install --upgrade transformers")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto" if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else device,
                    trust_remote_code=True
                )
            else:  # qwen2.5 or default
                # Qwen 2.5 模型使用 AutoProcessor
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                if Qwen2_5_VLForConditionalGeneration is None:
                    raise ImportError("Qwen2.5-VL model not available. Please update transformers: pip install --upgrade transformers")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto" if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else device,
                    trust_remote_code=True
                )
            
            self.model.eval()
            print(f"[INFO] Model loaded successfully! (Type: {model_type})", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}", flush=True)
            if model_type == "gemma3":
                print(f"[INFO] For Gemma-3, ensure you have the latest transformers version", flush=True)
            elif model_type in ["qwen2.5", "qwen3"]:
                print(f"[INFO] You may need to install: pip install qwen-vl-utils", flush=True)
                if model_type == "qwen3":
                    print(f"[INFO] For Qwen3-VL, ensure you have the latest transformers version", flush=True)
            raise
        
        # 存储每个step的评估结果
        self.step_results = []
        # 自动保存相关配置
        self.output_file = output_file
        self.save_interval = save_interval
        # 是否打印prompt
        self.verbose_prompt = verbose_prompt
        
    def _create_prompt(self, command_str: str, use_last_command: bool = False) -> str:
        """
        创建评估prompt
        
        Args:
            command_str: navigation command的文本描述
            use_last_command: 是否使用上一个command
            
        Returns:
            prompt: 用于评估的prompt
        """
        command_lower = command_str.lower()
        
        # 1. Go left/right/straight at intersection (use_last_command=False)
        if (("go left at the next intersection" in command_lower or 
             "go right at the next intersection" in command_lower or
             "go straight at the next intersection" in command_lower) and not use_last_command):
            prompt = """You are a driving trajectory analyst. Your task is to classify the intended driving maneuver based on the visual flow of the red waypoints in the image.

### Definitions of Maneuvers:
- **Go left**: The trajectory shows a clear intention to turn or curve toward the left, typically to head into a left-side path or street at the upcoming junction.
- **Go right**: The trajectory shows a clear intention to turn or curve toward the right, typically to head into a right-side path or street at the upcoming junction.
- **Go straight**: The trajectory follows the main continuation of the current road or proceeds directly through the upcoming intersection without turning.

### Analysis Logic:
1. Follow the flow of the red dots from the bottom of the image toward the horizon.
2. Identify if the path's overall direction points toward a turn at an intersection or stays aligned with the road's forward progression.

### Requirement:
Identify the driving intention. 

**Output only one of the following labels:** Go left / Go right / Go straight"""
        
        # 2. Go left at intersection (use_last_command=True)
        elif "go left at the next intersection" in command_lower and use_last_command:
            prompt = """You are a driving trajectory analyst. Your task is to classify the intended driving maneuver based on the visual flow of the red dots in the image.

### Definitions of Maneuvers:
- **Go left**: The trajectory shows a clear visual trend or directional pull toward the left (i.e., the red dots form a distinct curve or arc bending to the left).
- **Go straight**:The trajectory must strictly maintain a direct and linear path, proceeding directly through the intersection with minimal lateral deviation. The red dots should align closely with the forward direction of travel, forming a straight line.

### Analysis Logic:
Observe the geometric shape of the red dots from the bottom of the image toward the horizon.

### Requirement:
Identify the driving intention. 

**Output only one of the following labels:** Go left / Go straight"""
        
        # 3. Go right at intersection (use_last_command=True)
        elif "go right at the next intersection" in command_lower and use_last_command:
            prompt = """You are a driving trajectory analyst. Your task is to classify the intended driving maneuver based on the visual flow of the red dots in the image.

### Definitions of Maneuvers:
- **Go right**: The trajectory shows a clear visual trend or directional pull toward the right (i.e., the red dots form a distinct curve or arc bending to the right).
- **Go straight**: The trajectory must strictly maintain a direct and linear path, proceeding directly through the intersection with minimal lateral deviation. The red dots should align closely with the forward direction of travel, forming a straight line.

### Analysis Logic:
Observe the geometric shape of the red dots from the bottom of the image toward the horizon. 

### Requirement:
Identify the driving intention. 

**Output only one of the following labels:** Go right / Go straight"""
        
        # 4. Go straight at intersection (use_last_command=True)
        elif "go straight at the next intersection" in command_lower and use_last_command:
            prompt = """You are a driving trajectory analyst. Your task is to classify the intended driving maneuver based on the visual flow of the red dots in the image.

### Definitions of Maneuvers:
- **Go left**: The trajectory shows a clear visual trend or directional pull toward the left (i.e., the red dots form a distinct curve or arc bending to the left).
- **Go right**: The trajectory shows a clear visual trend or directional pull toward the right (i.e., the red dots form a distinct curve or arc bending to the right).
- **Go straight**: The trajectory must maintain a direct and nearly linear path, proceeding directly through the intersection with minimal lateral deviation. The red dots should align closely with the forward direction of travel, forming a straight line or very slight curve consistent with lane centering.

### Analysis Logic:
Observe the geometric shape of the red dots from the bottom of the image toward the horizon. Determine whether the path forms a noticeable arc or directional bend toward the left, right, or remains linear.

### Requirement:
Identify the driving intention. 

**Output only one of the following labels:** Go left / Go right / Go straight"""
        
        # 5. Follow the road
        elif "follow the road" in command_lower:
            prompt = """You are a driving trajectory analyst. Your task is to classify the intended driving maneuver based solely on the visual path represented by the red waypoints in the image.

### Definitions of Maneuvers:
- **Go left**: The trajectory shows a clear leftward turn or curve intended to exit the current road at the next intersection or junction.
- **Go right**: The trajectory shows a clear rightward turn or curve intended to exit the current road at the next intersection or junction.
- **Follow the road**: The trajectory follows the general topology of the visible road ahead. 
    *Crucially*, this includes:
    - Staying within the road's natural extension (even if the road curves).
    - Lateral shifts or diagonal movements to change lanes.
    - S-curves or maneuvers to bypass obstacles.
    - Slight arcing to re-center the vehicle within road boundaries.
    As long as the trajectory does not exit the current road onto a crossing or side street at an intersection, it must be classified as "Follow the road".

### Analysis Logic:
1. Observe the red waypoints from the vehicle's perspective toward the upcoming intersection or road extension.
2. Distinguish between "changing course to a new road at an intersection" (Go left/right) and "maneuvering within the current road" (Follow the road).

### Requirement:
Identify the driving intention. 

**Output only one of the following labels:** Go left / Go right / Follow the road"""
        
        # 6. Lane change to left
        elif "do a lane change to the left" in command_lower:
            prompt = """You are a driving trajectory analyst. Your task is to identify if the red dots are performing a lane change to left or maintaining its lane.

### Definitions of Maneuvers:
- **Lane change to left**: The trajectory shows a clear lateral shift to the left, crossing or moving toward the left lane boundary relative to the current vehicle heading.
- **Follow the road**: The trajectory stays within the natural extension of the current lane/road, following its curvature without shifting into an adjacent lane.

### Analysis Logic:
Observe the sequence of the red waypoints starting from the vehicle's front.

### Requirement:
Based on the red trajectory in the image, identify the driving intention.

**Output only one of the following labels:** left / Follow the road"""
        
        # 7. Lane change to right
        elif "do a lane change to the right" in command_lower:
            prompt = """You are a driving trajectory analyst. Your task is to identify if the red dots are performing a lane change to right or maintaining its lane.

### Definitions of Maneuvers:
- **Lane change to right**: The trajectory shows a clear lateral shift to the right, crossing or moving toward the right lane boundary relative to the current vehicle heading.
- **Follow the road**: The trajectory stays within the natural extension of the current lane/road, following its curvature without shifting into an adjacent lane.

### Analysis Logic:
Observe the sequence of the red waypoints starting from the vehicle's front.

### Requirement:
Based on the red trajectory in the image, identify the driving intention.

**Output only one of the following labels:** right / Follow the road"""
        
        # Default prompt (fallback)
        else:
            prompt = f"""You are a driving trajectory evaluator. Your task is to check if the predicted path (red dots) matches the navigation command.

### Navigation Command:
{command_str}

### Evaluation Logic:
1. Identify the 'Target Direction' from the command.
2. Observe the 'Visual Trajectory' of the red waypoints in the image.
3. Compare if Visual Trajectory matches the intent of the Target Direction.

### Constraints:
- Ignore lane markings, traffic lights, and safety.
- Focus ONLY on the directional alignment between the command and the red dots.
- Output ONLY one word: "Yes" or "No".
"""
        
        return prompt
    
    def _parse_response(self, response: str, command_str: str) -> tuple[bool, str]:
        """
        解析模型响应，提取Yes/No和原因，并根据command检查answer中的关键词
        
        Args:
            response: 模型的原始响应
            command_str: navigation command的文本描述，用于提取关键词
            
        Returns:
            tuple: (is_aligned: bool, reason: str)
        """
        response_original = response.strip()
        response_lower = response_original.lower()
        command_lower = command_str.lower()
        reason = ""
        
        # 尝试提取原因（查找"Reason:"后的内容，直到字符串结束）
        # 首先尝试匹配完整的Reason:后的所有内容（不包含Reason:前缀）
        reason_match = re.search(r'reason:\s*(.+?)(?=\n\s*(?:answer|question|command):|$)', response_original, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
            # 移除可能残留的"Reason:"前缀（如果模型在原因中重复了）
            reason = re.sub(r'^\s*reason:\s*', '', reason, flags=re.IGNORECASE)
            # 清理原因文本，移除多余的空白
            reason = re.sub(r'\s+', ' ', reason).strip()
        else:
            # 如果没有找到明确的Reason:标签，尝试其他模式
            reason_patterns = [
                r'reason:\s*(.+)',  # 匹配到字符串结束
                r'reasoning:\s*(.+?)(?:\n|$)',
                r'explanation:\s*(.+?)(?:\n|$)',
                r'because:\s*(.+?)(?:\n|$)',
            ]
            
            for pattern in reason_patterns:
                match = re.search(pattern, response_original, re.IGNORECASE | re.DOTALL)
                if match:
                    reason = match.group(1).strip()
                    # 移除可能残留的标签前缀
                    reason = re.sub(r'^\s*(?:reason|reasoning|explanation|because):\s*', '', reason, flags=re.IGNORECASE)
                    # 清理原因文本，移除多余的空白
                    reason = re.sub(r'\s+', ' ', reason).strip()
                    if reason:
                        break
        
        # 如果没有找到明确的原因，尝试从Answer之后提取
        if not reason:
            # 查找Answer之后的所有内容
            answer_match = re.search(r'answer:\s*(yes|no)\s*(.+?)(?:\n|$)', response_original, re.IGNORECASE | re.DOTALL)
            if answer_match:
                remaining = answer_match.group(2).strip()
                if remaining and len(remaining) > 5:
                    reason = remaining
        
        # 如果还是没有原因，尝试提取整个响应中除了Answer之外的部分
        if not reason:
            # 移除Answer行，保留其余部分
            cleaned = re.sub(r'answer:\s*(yes|no)\s*', '', response_original, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 5:
                reason = cleaned
        
        # 如果还是没有原因，使用整个响应（但标记为不完整）
        if not reason:
            reason = response_original
            if len(reason) < 20:  # 如果响应太短，可能是截断了
                reason += " [Response may be truncated]"
        
        # 根据command提取关键词并检查answer中是否包含
        is_aligned = False
        
        # 提取answer部分（模型可能直接输出label，如"Go left"、"Go straight"等）
        # 尝试提取明确的answer标签
        answer_label_match = re.search(r'(?:answer|label|output):\s*([^\n]+)', response_original, re.IGNORECASE)
        answer_text = ""
        if answer_label_match:
            answer_text = answer_label_match.group(1).strip().lower()
        else:
            # 如果没有明确的answer标签，使用整个响应
            answer_text = response_lower
        
        # 根据command类型检查关键词
        if "go straight" in command_lower or "straight" in command_lower:
            # 检查answer中是否有"straight"关键词
            if re.search(r'\bstraight\b', answer_text):
                is_aligned = True
            else:
                is_aligned = False
        elif ("go left" in command_lower or "do a lane change to the left" in command_lower or 
              "lane change to the left" in command_lower or "lane change to left" in command_lower):
            # 检查answer中是否有"left"关键词
            if re.search(r'\bleft\b', answer_text):
                is_aligned = True
            else:
                is_aligned = False
        elif ("go right" in command_lower or "do a lane change to the right" in command_lower or 
              "lane change to the right" in command_lower or "lane change to right" in command_lower):
            # 检查answer中是否有"right"关键词
            if re.search(r'\bright\b', answer_text):
                is_aligned = True
            else:
                is_aligned = False
        elif "follow the road" in command_lower or "follow" in command_lower:
            # 检查answer中是否有"follow"关键词
            if re.search(r'\bfollow\b', answer_text):
                is_aligned = True
            else:
                is_aligned = False
        else:
            # 如果command不匹配上述模式，打印警告并跳过
            print(f"[WARNING] Command '{command_str}' does not match any known pattern. Skipping keyword-based alignment check.", flush=True)
            is_aligned = False
        
        return is_aligned, reason
    
    def evaluate_from_image_path(self,
                                 image_path: Union[str, Path],
                                 actual_command: Union[int, str],
                                 step: Optional[int] = None,
                                 use_last_command: bool = False) -> Dict:
        """
        从图像路径评估command alignment
        
        Args:
            image_path: 包含waypoints可视化的图像路径
            actual_command: 实际command（可以是数值1-6或文本）
            step: step编号（可选）
            use_last_command: 是否使用上一个command（用于选择不同的prompt）
            
        Returns:
            result: 包含评估结果的字典
        """
        # 转换command为文本
        if isinstance(actual_command, int):
            command_str = self.COMMAND_MAP.get(actual_command, "follow the road")
        else:
            command_str = actual_command
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}", flush=True)
            return {
                'step': step,
                'actual_command': actual_command,
                'actual_command_str': command_str,
                'is_aligned': False,
                'error': str(e)
            }
        
        # 创建prompt
        prompt = self._create_prompt(command_str, use_last_command)
        
        # 打印prompt（如果启用）
        if self.verbose_prompt:
            print(f"\n{'=' * 80}", flush=True)
            print(f"[PROMPT] Step {step if step is not None else 'N/A'}", flush=True)
            print(f"[PROMPT] Command: {command_str}", flush=True)
            print(f"[PROMPT] Use last command: {use_last_command}", flush=True)
            print(f"[PROMPT] Prompt:", flush=True)
            print("-" * 80, flush=True)
            print(prompt, flush=True)
            print("-" * 80, flush=True)
            print(f"{'=' * 80}\n", flush=True)
        
                # 调用模型
        try:
            with torch.no_grad():
                if self.model_type == "gemma3":
                    # Gemma-3 使用 messages 格式，支持图像输入
                    # 注意：messages 格式应该是列表的列表（每个对话是一个列表）
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    # 使用 tokenizer/processor 的 apply_chat_template 处理
                    # 如果 processor 是 AutoProcessor，它应该能处理图像
                    # 如果 processor 是 AutoTokenizer，可能需要特殊处理
                    try:
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                    except Exception as e:
                        # 如果 apply_chat_template 不支持图像，尝试只使用文本
                        print(f"[WARNING] Image processing failed, using text only: {e}", flush=True)
                        text_messages = [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}]
                            }
                        ]
                        inputs = self.processor.apply_chat_template(
                            text_messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                    
                    # 移动到设备并转换数据类型
                    # 注意：input_ids 和 attention_mask 必须是整数类型（Long/Int），不能转换为 bfloat16
                    if hasattr(self.model, 'device'):
                        target_device = self.model.device
                    elif hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                        target_device = list(self.model.hf_device_map.values())[0]
                    else:
                        target_device = self.device if isinstance(self.device, str) else "cuda:0"
                    
                    # 处理每个输入，确保类型正确
                    processed_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            # input_ids 和 attention_mask 保持原类型（Long/Int），只移动设备
                            if k in ['input_ids', 'attention_mask']:
                                processed_inputs[k] = v.to(target_device)
                            # 其他张量（如 pixel_values）可以转换数据类型
                            else:
                                if self.device == "cuda":
                                    processed_inputs[k] = v.to(target_device).to(torch.bfloat16)
                                else:
                                    processed_inputs[k] = v.to(target_device).to(torch.float32)
                        else:
                            processed_inputs[k] = v
                    inputs = processed_inputs

                    # 移除 token_type_ids（Gemma不使用）
                    if 'token_type_ids' in inputs:
                        inputs.pop('token_type_ids')

                    # 生成响应
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            temperature=0.7,
                            top_p=1.0,
                            min_p=0.0,
                            top_k=50,
                            repetition_penalty=1.0
                        )
                    
                    # 提取生成的部分（去掉输入部分）
                    # 获取输入的长度
                    input_length = inputs['input_ids'].shape[1]
                    # 只解码生成的部分
                    generated_ids_trimmed = generated_ids[:, input_length:]
                    response = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                else:
                    # Qwen2.5-VL/Qwen3-VL使用messages格式
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    # 处理输入
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if process_vision_info is not None:
                        image_inputs, video_inputs = process_vision_info(messages)
                    else:
                        image_inputs = [image]
                        video_inputs = None
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    # 移动到设备
                    if hasattr(self.model, 'device'):
                        inputs = inputs.to(self.model.device)
                    elif hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                        first_device = list(self.model.hf_device_map.values())[0]
                        inputs = inputs.to(first_device)
                    else:
                        target_device = self.device if isinstance(self.device, str) else "cuda:0"
                        inputs = inputs.to(target_device)
                    
                    # 生成响应
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.7,
                        top_p=1.0,
                        min_p=0.0,
                        top_k=50,
                        repetition_penalty=1.0
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                
                # 解析响应
                is_aligned, reason = self._parse_response(response, command_str)
                
        except Exception as e:
            print(f"[ERROR] Model inference failed: {e}", flush=True)
            return {
                'step': step,
                'actual_command': actual_command,
                'actual_command_str': command_str,
                'is_aligned': False,
                'reason': '',
                'error': str(e)
            }
        
        result = {
            'step': step,
            'actual_command': actual_command if isinstance(actual_command, int) else None,
            'actual_command_str': command_str,
            'is_aligned': is_aligned,
            'model_response': response,
            'reason': reason,
            'image_path': str(image_path)
        }
        
        self.step_results.append(result)
        # 检查是否需要自动保存
        self._check_and_save()
        return result
    
    def evaluate_from_image_array(self,
                                  image: np.ndarray,
                                  actual_command: Union[int, str],
                                  step: Optional[int] = None,
                                  use_last_command: bool = False) -> Dict:
        """
        从numpy数组评估command alignment
        
        Args:
            image: 图像数组 (H, W, 3) RGB格式
            actual_command: 实际command（可以是数值1-6或文本）
            step: step编号（可选）
            use_last_command: 是否使用上一个command（用于选择不同的prompt）
            
        Returns:
            result: 包含评估结果的字典
        """
        # 转换numpy数组为PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # 转换command为文本
        if isinstance(actual_command, int):
            command_str = self.COMMAND_MAP.get(actual_command, "follow the road")
        else:
            command_str = actual_command
        
        # 创建prompt
        prompt = self._create_prompt(command_str, use_last_command)
        
        # 打印prompt（如果启用）
        if self.verbose_prompt:
            print(f"\n{'=' * 80}", flush=True)
            print(f"[PROMPT] Step {step if step is not None else 'N/A'}", flush=True)
            print(f"[PROMPT] Command: {command_str}", flush=True)
            print(f"[PROMPT] Use last command: {use_last_command}", flush=True)
            print(f"[PROMPT] Prompt:", flush=True)
            print("-" * 80, flush=True)
            print(prompt, flush=True)
            print("-" * 80, flush=True)
            print(f"{'=' * 80}\n", flush=True)
        
        # 调用模型
        try:
            with torch.no_grad():
                if self.model_type == "gemma3":
                    # Gemma-3 使用 messages 格式，支持图像输入
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_image},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    # 使用 tokenizer/processor 的 apply_chat_template 处理
                    try:
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                    except Exception as e:
                        # 如果 apply_chat_template 不支持图像，尝试只使用文本
                        print(f"[WARNING] Image processing failed, using text only: {e}", flush=True)
                        text_messages = [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}]
                            }
                        ]
                        inputs = self.processor.apply_chat_template(
                            text_messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                    
                    # 移动到设备并转换数据类型
                    # 注意：input_ids 和 attention_mask 必须是整数类型（Long/Int），不能转换为 bfloat16
                    if hasattr(self.model, 'device'):
                        target_device = self.model.device
                    elif hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                        target_device = list(self.model.hf_device_map.values())[0]
                    else:
                        target_device = self.device if isinstance(self.device, str) else "cuda:0"
                    
                    # 处理每个输入，确保类型正确
                    processed_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            # input_ids 和 attention_mask 保持原类型（Long/Int），只移动设备
                            if k in ['input_ids', 'attention_mask']:
                                processed_inputs[k] = v.to(target_device)
                            # 其他张量（如 pixel_values）可以转换数据类型
                            else:
                                if self.device == "cuda":
                                    processed_inputs[k] = v.to(target_device).to(torch.bfloat16)
                                else:
                                    processed_inputs[k] = v.to(target_device).to(torch.float32)
                        else:
                            processed_inputs[k] = v
                    inputs = processed_inputs

                    # 移除 token_type_ids（Gemma不使用）
                    if 'token_type_ids' in inputs:
                        inputs.pop('token_type_ids')

                    # 生成响应
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            temperature=0.7,
                            top_p=1.0,
                            min_p=0.0,
                            top_k=50,
                            repetition_penalty=1.0
                        )
                    
                    # 提取生成的部分（去掉输入部分）
                    # 获取输入的长度
                    input_length = inputs['input_ids'].shape[1]
                    # 只解码生成的部分
                    generated_ids_trimmed = generated_ids[:, input_length:]
                    response = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                else:
                    # Qwen2.5-VL/Qwen3-VL使用messages格式
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_image},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    # 处理输入
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if process_vision_info is not None:
                        image_inputs, video_inputs = process_vision_info(messages)
                    else:
                        image_inputs = [pil_image]
                        video_inputs = None
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    # 移动到设备
                    if hasattr(self.model, 'device'):
                        inputs = inputs.to(self.model.device)
                    elif hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                        first_device = list(self.model.hf_device_map.values())[0]
                        inputs = inputs.to(first_device)
                    else:
                        target_device = self.device if isinstance(self.device, str) else "cuda:0"
                        inputs = inputs.to(target_device)
                    
                    # 生成响应
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.7,
                        top_p=1.0,
                        min_p=0.0,
                        top_k=50,
                        repetition_penalty=1.0
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                
                # 解析响应
                is_aligned, reason = self._parse_response(response, command_str)
                
        except Exception as e:
            print(f"[ERROR] Model inference failed: {e}", flush=True)
            return {
                'step': step,
                'actual_command': actual_command if isinstance(actual_command, int) else None,
                'actual_command_str': command_str,
                'is_aligned': False,
                'reason': '',
                'error': str(e)
            }
        
        result = {
            'step': step,
            'actual_command': actual_command if isinstance(actual_command, int) else None,
            'actual_command_str': command_str,
            'is_aligned': is_aligned,
            'model_response': response,
            'reason': reason
        }
        
        self.step_results.append(result)
        # 检查是否需要自动保存
        self._check_and_save()
        return result
    

    def get_statistics(self) -> Dict:
        """
        获取整体统计信息
        
        Returns:
            stats: 统计信息字典
        """
        if len(self.step_results) == 0:
            return {}
        
        total_steps = len(self.step_results)
        aligned_steps = sum(1 for r in self.step_results if r.get('is_aligned', False))
        alignment_rate = aligned_steps / total_steps if total_steps > 0 else 0.0
        
        # 按command类型统计
        command_stats = defaultdict(lambda: {'total': 0, 'aligned': 0})
        for result in self.step_results:
            cmd = result.get('actual_command')
            if cmd is not None:
                command_stats[cmd]['total'] += 1
                if result.get('is_aligned', False):
                    command_stats[cmd]['aligned'] += 1
        
        # 计算每个command的对齐率
        command_alignment_rates = {}
        for cmd, stats in command_stats.items():
            rate = stats['aligned'] / stats['total'] if stats['total'] > 0 else 0.0
            command_alignment_rates[cmd] = {
                'command_str': self.COMMAND_MAP.get(cmd, 'unknown'),
                'total': stats['total'],
                'aligned': stats['aligned'],
                'alignment_rate': rate
            }
        
        stats = {
            'total_steps': total_steps,
            'aligned_steps': aligned_steps,
            'overall_alignment_rate': alignment_rate,
            'command_alignment_rates': dict(command_alignment_rates)
        }
        
        return stats
    
    def _check_and_save(self):
        """检查是否需要保存结果（每save_interval个结果保存一次）"""
        if self.output_file is None:
            return
        
        if len(self.step_results) > 0 and len(self.step_results) % self.save_interval == 0:
            try:
                self.save_results(self.output_file)
                print(f"[INFO] Saved {len(self.step_results)} results to {self.output_file}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to save results to {self.output_file}: {e}", flush=True)
    
    def save_results(self, filepath: str):
        """保存评估结果到JSON文件"""
        results = {
            'step_results': self.step_results,
            'statistics': self.get_statistics()
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def reset(self):
        """重置评估器"""
        self.step_results = []

