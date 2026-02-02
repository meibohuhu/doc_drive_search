"""
Model-based Command Alignment Evaluator using Qwen2.5-VL-3B
mh 20260202:
使用Qwen2.5-VL-3B视觉语言模型评估预测的waypoints是否与navigation command对齐
"""

import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import re


class ModelBasedCommandAlignmentEvaluator:
    """
    使用Qwen2.5-VL-3B评估模型预测的waypoints是否与navigation command对齐
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
                 model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_new_tokens: int = 10):
        """
        Args:
            model_name: Qwen2.5-VL模型名称
            device: 运行设备 ('cuda' or 'cpu')
            max_new_tokens: 生成的最大token数
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        print(f"[INFO] Loading Qwen2.5-VL model: {model_name} on {device}...", flush=True)
        
        # 加载模型和processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True
            )
            self.model.eval()
            print(f"[INFO] Model loaded successfully!", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}", flush=True)
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
        prompt = f"""You are evaluating whether a predicted driving trajectory (shown as red dots/waypoints in the image) generally aligns with the given navigation command.

Navigation command: "{command_str}"

The image shows:
- The current driving scene from a first-person perspective
- Predicted waypoints (red dots/ellipses) showing where the vehicle plans to go (typically covering about 15-20 meters ahead)
- The road, lanes, and surrounding environment

IMPORTANT evaluation criteria:
1. Distance-based commands: If the command mentions a distance (e.g., "turn left in 28 meter"), and the predicted waypoints only cover a shorter distance (e.g., 15-20 meters), it is VALID if:
   - The waypoints show the vehicle approaching the target location (e.g., going straight toward an intersection where it will turn)
   - The trajectory is consistent with preparing for the command (e.g., staying in the correct lane, approaching the intersection)
   - The waypoints do NOT contradict the command direction

2. General alignment:
   - The trajectory should GENERALLY follow the command direction
   - Minor deviations, slight tilting, or gentle curves are ACCEPTABLE
   - Even if there might be potential risks or obstacles ahead, as long as the trajectory direction matches the command, it should be considered valid
   - The trajectory does NOT need to be perfectly aligned - approximate alignment is sufficient

3. Only reject if:
   - The trajectory clearly contradicts the command (e.g., command says "turn left" but waypoints go right, or command says "go straight" but waypoints make a sharp unexpected turn)
   - The trajectory shows the vehicle moving away from where it needs to be to execute the command

Question: Does the predicted trajectory (red waypoints) generally align with the navigation command, considering the distance constraints and that minor deviations are acceptable?

Please answer with ONLY one word: "Yes" or "No".

Answer:"""
        return prompt
    
    def _parse_response(self, response: str) -> bool:
        """
        解析模型响应，提取Yes/No
        
        Args:
            response: 模型的原始响应
            
        Returns:
            bool: True if aligned, False otherwise
        """
        response_lower = response.strip().lower()
        
        # 查找yes/no
        if "yes" in response_lower:
            # 确保不是"no"的一部分
            if "no" not in response_lower or response_lower.find("yes") < response_lower.find("no"):
                return True
        if "no" in response_lower:
            return False
        
        # 如果找不到明确的yes/no，尝试其他模式
        if any(word in response_lower for word in ["aligned", "correct", "matches", "consistent", "appropriate"]):
            return True
        if any(word in response_lower for word in ["not aligned", "incorrect", "doesn't match", "inconsistent", "inappropriate"]):
            return False
        
        # 默认返回False（保守策略）
        print(f"[WARNING] Could not parse response: {response}, defaulting to False", flush=True)
        return False
    
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
                
                # 处理输入
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = self.processor.process_visual_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # 移动到设备
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # 生成响应
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # 解析响应
                is_aligned = self._parse_response(response)
                
        except Exception as e:
            print(f"[ERROR] Model inference failed: {e}", flush=True)
            return {
                'step': step,
                'actual_command': actual_command,
                'actual_command_str': command_str,
                'is_aligned': False,
                'error': str(e)
            }
        
        result = {
            'step': step,
            'actual_command': actual_command if isinstance(actual_command, int) else None,
            'actual_command_str': command_str,
            'is_aligned': is_aligned,
            'model_response': response,
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
                
                # 处理输入
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = self.processor.process_visual_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # 移动到设备
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # 生成响应
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # 解析响应
                is_aligned = self._parse_response(response)
                
        except Exception as e:
            print(f"[ERROR] Model inference failed: {e}", flush=True)
            return {
                'step': step,
                'actual_command': actual_command if isinstance(actual_command, int) else None,
                'actual_command_str': command_str,
                'is_aligned': False,
                'error': str(e)
            }
        
        result = {
            'step': step,
            'actual_command': actual_command if isinstance(actual_command, int) else None,
            'actual_command_str': command_str,
            'is_aligned': is_aligned,
            'model_response': response
        }
        
        self.step_results.append(result)
        return result
    
    def evaluate_from_step_logs(self,
                                step_logs_path: Union[str, Path],
                                images_dir: Union[str, Path],
                                image_filename_template: str = "{step}.png") -> List[Dict]:
        """
        从step_logs.json批量评估
        
        Args:
            step_logs_path: step_logs.json文件路径
            images_dir: 图像目录
            image_filename_template: 图像文件名模板，{step}会被替换为step编号
            
        Returns:
            results: 评估结果列表
        """
        # 加载step_logs
        with open(step_logs_path, 'r') as f:
            step_logs = json.load(f)
        
        images_dir = Path(images_dir)
        results = []
        
        print(f"[INFO] Evaluating {len(step_logs)} steps...", flush=True)
        
        for i, log_entry in enumerate(step_logs):
            step = log_entry.get('step', i)
            actual_command = log_entry.get('actual_command')
            
            if actual_command is None:
                print(f"[WARNING] Step {step}: no actual_command, skipping", flush=True)
                continue
            
            # 构建图像路径
            image_filename = image_filename_template.format(step=step)
            image_path = images_dir / image_filename
            
            if not image_path.exists():
                print(f"[WARNING] Step {step}: image not found at {image_path}, skipping", flush=True)
                continue
            
            # 评估
            result = self.evaluate_from_image_path(image_path, actual_command, step)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"[INFO] Processed {i + 1}/{len(step_logs)} steps", flush=True)
        
        return results
    
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

