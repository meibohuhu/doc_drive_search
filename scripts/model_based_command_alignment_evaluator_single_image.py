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
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils")

import re


class ModelBasedCommandAlignmentEvaluator:
    """
    使用Qwen视觉语言模型评估模型预测的waypoints是否与navigation command对齐
    支持Qwen2.5-VL和Qwen3-VL模型
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
                 model_type: str = "auto"):
        """
        Args:
            model_name: Qwen模型名称（支持Qwen2.5-VL和Qwen3-VL）
            device: 运行设备 ('cuda' or 'cpu')
            max_new_tokens: 生成的最大token数
            model_type: 模型类型 ('auto', 'qwen2.5', 'qwen3')，auto会根据model_name自动检测
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        
        # 自动检测模型类型
        if model_type == "auto":
            model_name_lower = model_name.lower()
            if "qwen3" in model_name_lower:
                model_type = "qwen3"
            elif "qwen2.5" in model_name_lower or "qwen2" in model_name_lower:
                model_type = "qwen2.5"
            else:
                # 默认尝试Qwen2.5
                model_type = "qwen2.5"
        
        print(f"[INFO] Detected model type: {model_type} from model name: {model_name}", flush=True)
        
        self.model_type = model_type
        
        print(f"[INFO] Loading Qwen model: {model_name} (type: {model_type}) on {device}...", flush=True)
        
        # 加载模型和processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            # 根据模型类型选择加载方式
            if model_type == "qwen3":
                if Qwen3VLForConditionalGeneration is None:
                    raise ImportError("Qwen3VLForConditionalGeneration not available. Please update transformers: pip install --upgrade transformers")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto" if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else device,
                    trust_remote_code=True
                )
            else:  # qwen2.5 or default
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
            print(f"[INFO] You may need to install: pip install qwen-vl-utils", flush=True)
            if model_type == "qwen3":
                print(f"[INFO] For Qwen3-VL, ensure you have the latest transformers version", flush=True)
            raise
        
        # 存储每个step的评估结果
        self.step_results = []
        
    def _create_prompt(self, command_str: str) -> str:
        """
        创建评估prompt
        
        Args:
            command_str: navigation command的文本描述
            
        Returns:
            prompt: 用于评估的prompt
        """
        prompt = f"""You are a driving trajectory evaluator. Your task is to check if the predicted path (red dots) matches the navigation command.

### Navigation Command:
{command_str}

### Evaluation Logic:
1. Identify the 'Target Direction' from the command:
   - "Go right" -> Target: RIGHT (Expect a clear rightward curve/turn)
   - "Go left" -> Target: LEFT (Expect a clear leftward curve/turn)
   - "Go straight" -> Target: LINEAR STRAIGHT (Expect a visual straight line relative to the vehicle heading)
   - "Follow the road" -> Target: ROAD CONTINUATION (Expect the dots to stay within the current road's natural extension, even if the road curves slightly)

2. Observe the 'Visual Trajectory' of the red waypoints in the image.

3. Compare:
   - If Visual Trajectory matches the intent of the Target Direction -> "Yes"
   - If Visual Trajectory contradicts the Target Direction (e.g., Command is "Follow the road" but dots perform a 90-degree "Left" turn) -> "No"

### Constraints:
- Ignore lane markings, traffic lights, and safety.
- Focus ONLY on the directional alignment between the command and the red dots.
- Output ONLY one word: "Yes" or "No".
"""
        return prompt
    
    def _parse_response(self, response: str) -> tuple[bool, str]:
        """
        解析模型响应，提取Yes/No和原因
        
        Args:
            response: 模型的原始响应
            
        Returns:
            tuple: (is_aligned: bool, reason: str)
        """
        response_original = response.strip()
        response_lower = response_original.lower()
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
        
        # 查找yes/no
        is_aligned = False
        if re.search(r'\byes\b', response_lower):
            # 确保不是"no"的一部分
            if not re.search(r'\bno\b', response_lower) or response_lower.find("yes") < response_lower.find("no"):
                is_aligned = True
        elif re.search(r'\bno\b', response_lower):
            is_aligned = False
        else:
            # 如果找不到明确的yes/no，尝试其他模式
            if any(word in response_lower for word in ["aligned", "correct", "matches", "consistent", "appropriate"]):
                is_aligned = True
            elif any(word in response_lower for word in ["not aligned", "incorrect", "doesn't match", "inconsistent", "inappropriate"]):
                is_aligned = False
            else:
                # 默认返回False（保守策略）
                print(f"[WARNING] Could not parse response: {response_original}, defaulting to False", flush=True)
                is_aligned = False
        
        return is_aligned, reason
    
    def evaluate_from_image_path(self,
                                 image_path: Union[str, Path],
                                 actual_command: Union[int, str],
                                 step: Optional[int] = None) -> Dict:
        """
        从图像路径评估command alignment
        
        Args:
            image_path: 包含waypoints可视化的图像路径
            actual_command: 实际command（可以是数值1-6或文本）
            step: step编号（可选）
            
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
            print(f"[DEBUG] Image loaded: {image.size} (W x H), mode: {image.mode}", flush=True)
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
        prompt = self._create_prompt(command_str)
        
                # 调用模型
        try:
            with torch.no_grad():
                # Qwen2.5-VL使用messages格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # 处理输入 - 使用qwen_vl_utils.process_vision_info
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                print(f"[DEBUG] Processed {len(image_inputs)} image(s), {len(video_inputs) if video_inputs else 0} video(s)", flush=True)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # 移动到设备 - 如果使用device_map="auto"，模型可能分布在多个GPU上
                # 在这种情况下，将输入移动到第一个GPU或使用模型的设备
                if hasattr(self.model, 'device'):
                    inputs = inputs.to(self.model.device)
                elif hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                    # 如果模型分布在多个设备上，使用第一个设备
                    first_device = list(self.model.hf_device_map.values())[0]
                    inputs = inputs.to(first_device)
                else:
                    # 回退到cuda:0或指定的device
                    target_device = self.device if isinstance(self.device, str) else "cuda:0"
                    inputs = inputs.to(target_device)
                print(f"[DEBUG] Input shape: {inputs.input_ids.shape if hasattr(inputs, 'input_ids') else 'N/A'}", flush=True)
                
                # 生成响应
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                num_new_tokens = len(generated_ids_trimmed[0])
                print(f"[DEBUG] Generated {num_new_tokens} new tokens (max: {self.max_new_tokens})", flush=True)
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                print(f"[DEBUG] Full model response: {response[:200]}..." if len(response) > 200 else f"[DEBUG] Full model response: {response}", flush=True)
                
                # 解析响应
                is_aligned, reason = self._parse_response(response)
                
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
        return result
    
    def evaluate_from_image_array(self,
                                  image: np.ndarray,
                                  actual_command: Union[int, str],
                                  step: Optional[int] = None) -> Dict:
        """
        从numpy数组评估command alignment
        
        Args:
            image: 图像数组 (H, W, 3) RGB格式
            actual_command: 实际command（可以是数值1-6或文本）
            step: step编号（可选）
            
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
        prompt = self._create_prompt(command_str)
        
        # 调用模型
        try:
            with torch.no_grad():
                # Qwen2.5-VL使用messages格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # 处理输入 - 使用qwen_vl_utils.process_vision_info
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # 移动到设备 - 如果使用device_map="auto"，模型可能分布在多个GPU上
                # 在这种情况下，将输入移动到第一个GPU或使用模型的设备
                if hasattr(self.model, 'device'):
                    inputs = inputs.to(self.model.device)
                elif hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                    # 如果模型分布在多个设备上，使用第一个设备
                    first_device = list(self.model.hf_device_map.values())[0]
                    inputs = inputs.to(first_device)
                else:
                    # 回退到cuda:0或指定的device
                    target_device = self.device if isinstance(self.device, str) else "cuda:0"
                    inputs = inputs.to(target_device)
                
                # 生成响应
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # 解析响应
                is_aligned, reason = self._parse_response(response)
                
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

