"""
Stepwise Command Alignment Evaluator
mh 20260130: 
评估每个step的模型预测waypoints是否与navigation command对齐
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class CommandAlignmentEvaluator:
    """
    评估模型预测的waypoints是否与navigation command对齐
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
                 lateral_threshold: float = 1.5,  # 变道判断阈值（米）
                 turn_angle_threshold: float = 15.0,  # 转弯角度阈值（度）
                 lookahead_distance: float = 10.0):  # 用于判断command的lookahead距离（米）
        """
        Args:
            lateral_threshold: 判断变道的横向位移阈值（米）
            turn_angle_threshold: 判断转弯的角度阈值（度）
            lookahead_distance: 用于分析waypoints的lookahead距离（米）
        """
        self.lateral_threshold = lateral_threshold
        self.turn_angle_threshold = turn_angle_threshold
        self.lookahead_distance = lookahead_distance
        
        # 存储每个step的评估结果
        self.step_results = []
        
    def infer_command_from_waypoints(self, 
                                     waypoints: np.ndarray, 
                                     current_heading: float = 0.0,
                                     is_in_junction: bool = False) -> int:
        """
        从预测的waypoints推断模型认为应该执行的command
        
        Args:
            waypoints: 预测的waypoints，shape (N, 2) 在车辆坐标系中 (x, y)
            current_heading: 当前车辆朝向（弧度）
            is_in_junction: 是否在路口内
            
        Returns:
            inferred_command: 推断的command (1-6)
        """
        if waypoints is None or len(waypoints) == 0:
            return 4  # 默认跟随道路
        
        waypoints = np.array(waypoints)
        if waypoints.shape[0] < 2:
            return 4
        
        # 计算waypoints的累积距离，找到lookahead_distance对应的点
        distances = np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        distances = np.concatenate([[0], distances])
        
        # 找到lookahead距离内的waypoints
        lookahead_idx = np.where(distances <= self.lookahead_distance)[0]
        if len(lookahead_idx) == 0:
            lookahead_idx = [0, min(1, len(waypoints)-1)]
        else:
            lookahead_idx = [0, lookahead_idx[-1]]
        
        if lookahead_idx[-1] >= len(waypoints):
            lookahead_idx[-1] = len(waypoints) - 1
            
        # 分析waypoints的方向变化
        start_point = waypoints[0]
        end_point = waypoints[lookahead_idx[-1]]
        
        # 计算横向位移（y方向）
        lateral_displacement = end_point[1] - start_point[1]
        
        # 计算总体方向角度变化
        direction_vector = end_point - start_point
        direction_angle = np.arctan2(direction_vector[1], direction_vector[0])  # 弧度
        direction_angle_deg = np.degrees(direction_angle)
        
        # 计算waypoints的曲率（通过角度变化）
        if len(waypoints) >= 3:
            # 计算相邻waypoints的角度变化
            angles = []
            for i in range(len(waypoints) - 1):
                vec = waypoints[i+1] - waypoints[i]
                angle = np.arctan2(vec[1], vec[0])
                angles.append(angle)
            
            # 计算角度变化率（曲率指标）
            angle_changes = np.diff(angles)
            angle_changes = np.array([np.arctan2(np.sin(a), np.cos(a)) for a in angle_changes])  # 归一化到[-pi, pi]
            avg_angle_change = np.mean(np.abs(angle_changes))
            avg_angle_change_deg = np.degrees(avg_angle_change)
        else:
            avg_angle_change_deg = 0.0
        
        # 判断command类型
        # 1. 变道判断（优先级最高，因为变道有明显的横向位移）
        if abs(lateral_displacement) > self.lateral_threshold:
            if lateral_displacement > 0:  # 向左变道（y增加）
                return 5
            else:  # 向右变道（y减少）
                return 6
        
        # 2. 转弯判断（在路口或接近路口时）
        if is_in_junction or abs(direction_angle_deg) > self.turn_angle_threshold:
            if direction_angle_deg > self.turn_angle_threshold:
                return 1  # 左转
            elif direction_angle_deg < -self.turn_angle_threshold:
                return 2  # 右转
            elif abs(direction_angle_deg) <= self.turn_angle_threshold:
                return 3  # 直行
        
        # 3. 默认跟随道路
        return 4
    
    def evaluate_step(self,
                      step: int,
                      predicted_waypoints: np.ndarray,
                      actual_command: int,
                      current_heading: float = 0.0,
                      is_in_junction: bool = False,
                      metadata: Optional[Dict] = None) -> Dict:
        """
        评估单个step的command alignment
        
        Args:
            step: 当前step编号
            predicted_waypoints: 模型预测的waypoints (N, 2) 在车辆坐标系
            actual_command: 实际应该执行的command (1-6)
            current_heading: 当前车辆朝向（弧度）
            is_in_junction: 是否在路口内
            metadata: 额外的元数据（可选）
            
        Returns:
            result: 包含评估结果的字典
        """
        # 推断command
        inferred_command = self.infer_command_from_waypoints(
            predicted_waypoints, current_heading, is_in_junction
        )
        
        # 判断是否对齐
        is_aligned = (inferred_command == actual_command)
        
        # 计算waypoints的统计信息
        if predicted_waypoints is not None and len(predicted_waypoints) > 0:
            lateral_displacement = predicted_waypoints[-1][1] - predicted_waypoints[0][1]
            forward_displacement = predicted_waypoints[-1][0] - predicted_waypoints[0][0]
            total_distance = np.sum(np.linalg.norm(np.diff(predicted_waypoints, axis=0), axis=1))
        else:
            lateral_displacement = 0.0
            forward_displacement = 0.0
            total_distance = 0.0
        
        result = {
            'step': step,
            'actual_command': actual_command,
            'actual_command_str': self.COMMAND_MAP.get(actual_command, 'unknown'),
            'inferred_command': inferred_command,
            'inferred_command_str': self.COMMAND_MAP.get(inferred_command, 'unknown'),
            'is_aligned': is_aligned,
            'lateral_displacement': float(lateral_displacement),
            'forward_displacement': float(forward_displacement),
            'total_distance': float(total_distance),
            'is_in_junction': is_in_junction,
        }
        
        if metadata:
            result['metadata'] = metadata
        
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
        aligned_steps = sum(1 for r in self.step_results if r['is_aligned'])
        alignment_rate = aligned_steps / total_steps if total_steps > 0 else 0.0
        
        # 按command类型统计
        command_stats = defaultdict(lambda: {'total': 0, 'aligned': 0})
        for result in self.step_results:
            cmd = result['actual_command']
            command_stats[cmd]['total'] += 1
            if result['is_aligned']:
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
        
        # 混淆矩阵
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for result in self.step_results:
            actual = result['actual_command']
            inferred = result['inferred_command']
            confusion_matrix[actual][inferred] += 1
        
        stats = {
            'total_steps': total_steps,
            'aligned_steps': aligned_steps,
            'overall_alignment_rate': alignment_rate,
            'command_alignment_rates': dict(command_alignment_rates),
            'confusion_matrix': {k: dict(v) for k, v in confusion_matrix.items()}
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

