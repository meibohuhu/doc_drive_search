#!/usr/bin/env python3
"""
独立脚本：从已有的evaluation logs中分析command alignment

如果不想修改agent代码，可以使用这个脚本从保存的logs中分析
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from command_alignment_evaluator import CommandAlignmentEvaluator
import argparse


def load_evaluation_logs(log_dir: str) -> List[Dict]:
    """
    从evaluation logs中加载数据
    
    需要根据实际的log格式调整
    """
    log_path = Path(log_dir)
    logs = []
    
    # TODO: 根据实际log格式加载数据
    # 示例：如果logs保存在JSON文件中
    if (log_path / 'step_logs.json').exists():
        with open(log_path / 'step_logs.json', 'r') as f:
            logs = json.load(f)
    
    return logs


def analyze_command_alignment(log_dir: str, output_file: str = None):
    """
    分析command alignment
    
    Args:
        log_dir: 包含evaluation logs的目录
        output_file: 输出文件路径（可选）
    """
    evaluator = CommandAlignmentEvaluator()
    
    # 加载logs
    logs = load_evaluation_logs(log_dir)
    
    if len(logs) == 0:
        print(f"No logs found in {log_dir}")
        return
    
    # 处理每个step
    for log_entry in logs:
        step = log_entry.get('step', 0)
        predicted_waypoints = np.array(log_entry.get('predicted_waypoints', []))
        actual_command = log_entry.get('actual_command', 4)
        current_heading = log_entry.get('current_heading', 0.0)
        is_in_junction = log_entry.get('is_in_junction', False)
        
        evaluator.evaluate_step(
            step=step,
            predicted_waypoints=predicted_waypoints,
            actual_command=actual_command,
            current_heading=current_heading,
            is_in_junction=is_in_junction,
            metadata=log_entry.get('metadata', {})
        )
    
    # 获取统计信息
    stats = evaluator.get_statistics()
    
    # 打印结果
    print("\n" + "="*60)
    print("Command Alignment Evaluation Results")
    print("="*60)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Aligned steps: {stats['aligned_steps']}")
    print(f"Overall alignment rate: {stats['overall_alignment_rate']:.2%}")
    print("\nPer-command alignment rates:")
    for cmd, cmd_stats in sorted(stats['command_alignment_rates'].items()):
        print(f"  Command {cmd} ({cmd_stats['command_str']}): "
              f"{cmd_stats['aligned']}/{cmd_stats['total']} = {cmd_stats['alignment_rate']:.2%}")
    
    print("\nConfusion Matrix (Actual -> Inferred):")
    for actual_cmd in sorted(stats['confusion_matrix'].keys()):
        actual_str = evaluator.COMMAND_MAP.get(actual_cmd, 'unknown')
        print(f"  Actual {actual_cmd} ({actual_str}):")
        for inferred_cmd, count in sorted(stats['confusion_matrix'][actual_cmd].items()):
            inferred_str = evaluator.COMMAND_MAP.get(inferred_cmd, 'unknown')
            print(f"    -> Inferred {inferred_cmd} ({inferred_str}): {count}")
    print("="*60)
    
    # 保存结果
    if output_file:
        evaluator.save_results(output_file)
        print(f"\nResults saved to {output_file}")
    else:
        output_file = Path(log_dir) / 'command_alignment_eval.json'
        evaluator.save_results(str(output_file))
        print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze command alignment from evaluation logs')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing evaluation logs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (optional)')
    
    args = parser.parse_args()
    analyze_command_alignment(args.log_dir, args.output)

