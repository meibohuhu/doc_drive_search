#!/usr/bin/env python3
"""
批量评估脚本：使用ModelBasedCommandAlignmentEvaluator评估CSV中每个step的command与对应图像的alignment
mh 20260204: 处理每5步的图像和command对齐评估
"""

import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from model_based_command_alignment_evaluator import ModelBasedCommandAlignmentEvaluator
except ImportError as e:
    print(f"[ERROR] Failed to import evaluator: {e}")
    print("\n[INFO] Please ensure model_based_command_alignment_evaluator.py is in the same directory")
    sys.exit(1)


def find_image_folder(base_path: Path, route_name: str) -> Optional[Path]:
    """
    查找指定route的图像文件夹
    
    Args:
        base_path: 基础路径（应该包含RouteScenario_*_rep0_*目录）
        route_name: route名称（如RouteScenario_1711）
        
    Returns:
        图像文件夹路径，如果不存在则返回None
    """
    # 尝试多种可能的路径结构
    possible_paths = [
        # 标准路径：base_path/debug_viz/simlingo/iter_013.ckpt/RouteScenario_XXX/images
        base_path / "debug_viz" / "simlingo" / "iter_013.ckpt" / route_name / "images",
        # 如果base_path已经是debug_viz的父目录
        base_path.parent / "debug_viz" / "simlingo" / "iter_013.ckpt" / route_name / "images" if base_path.name.startswith("RouteScenario_") else None,
        # 直接路径
        base_path / route_name / "images",
        base_path / "images",
    ]
    
    # 过滤掉None值
    possible_paths = [p for p in possible_paths if p is not None]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path
    
    return None


def get_image_path(image_folder: Path, step: int) -> Optional[Path]:
    """
    获取指定step的图像路径
    
    Args:
        image_folder: 图像文件夹路径
        step: step编号
        
    Returns:
        图像路径，如果不存在则返回None
    """
    # 图像命名格式：{step}.png
    image_path = image_folder / f"{step}.png"
    if image_path.exists():
        return image_path
    return None


def load_csv_data(csv_path: Path) -> List[Dict]:
    """
    加载CSV数据
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        包含所有step数据的列表
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数据类型
            row['step'] = int(row['step'])
            row['far_command'] = int(row['far_command'])
            row['next_far_command'] = int(row['next_far_command'])
            row['last_command'] = int(row['last_command'])
            row['use_last_command'] = row['use_last_command'] == 'True'
            row['actual_command'] = int(row['actual_command'])
            row['dist_to_command'] = int(row['dist_to_command'])
            row['command_replaced'] = row['command_replaced'] == 'True'
            data.append(row)
    return data


def filter_steps_by_interval(data: List[Dict], interval: int = 5) -> List[Dict]:
    """
    过滤出每N步的数据（默认每5步）
    
    Args:
        data: 所有step数据
        interval: 步长间隔
        
    Returns:
        过滤后的数据
    """
    return [row for row in data if row['step'] % interval == 0]


def main():
    parser = argparse.ArgumentParser(description='批量评估command与图像的alignment')
    parser.add_argument('--csv', type=str, required=True,
                       help='CSV文件路径（包含step和command信息）')
    parser.add_argument('--image_base', type=str, required=True,
                       help='图像文件夹的基础路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径（默认：在CSV同目录下生成）')
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                       help='模型名称（支持Qwen和Gemma-3，例如：Qwen/Qwen3-VL-2B-Instruct, google/gemma-3-2b-it 等，默认：Qwen/Qwen3-VL-2B-Instruct）')
    parser.add_argument('--device', type=str, default="cuda",
                       help='运行设备（默认：cuda）')
    parser.add_argument('--interval', type=int, default=5,
                       help='处理间隔（默认：每5步处理一次）')
    parser.add_argument('--max_tokens', type=int, default=200,
                       help='最大生成token数（默认：20）')
    parser.add_argument('--route', type=str, default=None,
                       help='指定处理的route（默认：处理所有route）')
    parser.add_argument('--verbose_prompt', action='store_true',
                       help='打印使用的prompt（默认：False）')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    image_base = Path(args.image_base)
    
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        sys.exit(1)
    
    if not image_base.exists():
        print(f"[ERROR] Image base path not found: {image_base}")
        sys.exit(1)
    
    # 确定输出文件路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.parent / f"{csv_path.stem}_alignment_results.json"
    
    print("=" * 80)
    print("Batch Command Alignment Evaluator")
    print("=" * 80)
    print(f"CSV file: {csv_path}")
    print(f"Image base: {image_base}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Interval: Every {args.interval} steps")
    print("=" * 80)
    print()
    
    # 加载CSV数据
    print("[INFO] Loading CSV data...")
    all_data = load_csv_data(csv_path)
    print(f"[INFO] Loaded {len(all_data)} steps from CSV")
    
    # 过滤出每N步的数据
    filtered_data = filter_steps_by_interval(all_data, args.interval)
    print(f"[INFO] Filtered to {len(filtered_data)} steps (every {args.interval} steps)")
    
    # 如果指定了route，只处理该route
    if args.route:
        filtered_data = [row for row in filtered_data if row['route'] == args.route]
        print(f"[INFO] Filtered to route {args.route}: {len(filtered_data)} steps")
    
    if len(filtered_data) == 0:
        print("[ERROR] No data to process!")
        sys.exit(1)
    
    # 按route分组
    routes_data = {}
    for row in filtered_data:
        route = row['route']
        if route not in routes_data:
            routes_data[route] = []
        routes_data[route].append(row)
    
    print(f"\n[INFO] Found {len(routes_data)} route(s): {list(routes_data.keys())}")
    
    # 初始化评估器
    print(f"\n[INFO] Initializing evaluator with model: {args.model}...")
    try:
        evaluator = ModelBasedCommandAlignmentEvaluator(
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_tokens,
            output_file=str(output_path),  # 启用自动保存功能
            save_interval=10,  # 每10个结果保存一次
            verbose_prompt=args.verbose_prompt  # 是否打印prompt
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize evaluator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 处理每个route
    all_results = []
    processed_count = 0
    skipped_count = 0
    
    for route_name, route_steps in sorted(routes_data.items()):
        print(f"\n{'=' * 80}")
        print(f"Processing Route: {route_name}")
        print(f"{'=' * 80}")
        print(f"Total steps to process: {len(route_steps)}")
        
        # 查找图像文件夹
        image_folder = find_image_folder(image_base, route_name)
        if image_folder is None:
            print(f"[WARNING] Image folder not found for {route_name}, skipping...")
            skipped_count += len(route_steps)
            continue
        
        print(f"[INFO] Image folder: {image_folder}")
        
        # 处理每个step
        for i, row in enumerate(route_steps):
            step = row['step']
            command = row['command']  # 使用command字段（文本格式）
            actual_command = row['actual_command']  # 数值格式
            
            # 查找图像
            image_path = get_image_path(image_folder, step)
            if image_path is None:
                print(f"[WARNING] Step {step}: Image not found ({step}.png), skipping...")
                skipped_count += 1
                continue
            
            print(f"\n[{i+1}/{len(route_steps)}] Processing Step {step}...")
            print(f"  Image: {image_path.name}")
            print(f"  Command: '{command}' (actual_command={actual_command})")
            
            # 评估
            try:
                result = evaluator.evaluate_from_image_path(
                    image_path=image_path,
                    actual_command=command,  # 使用文本command
                    step=step,
                    use_last_command=row['use_last_command']
                )
                
                # 添加额外信息
                result['route'] = route_name
                result['far_command'] = row['far_command']
                result['next_far_command'] = row['next_far_command']
                result['last_command'] = row['last_command']
                result['use_last_command'] = row['use_last_command']
                result['next_command'] = row['next_command']
                result['dist_to_command'] = row['dist_to_command']
                result['command_replaced'] = row['command_replaced']
                
                all_results.append(result)
                processed_count += 1
                
                print(f"  Result: {'✓ Aligned' if result.get('is_aligned') else '✗ Not Aligned'}")
                if result.get('reason'):
                    reason = result['reason'][:100] + "..." if len(result['reason']) > 100 else result['reason']
                    print(f"  Reason: {reason}")
                
            except Exception as e:
                print(f"[ERROR] Failed to evaluate step {step}: {e}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
                continue
    
    # 保存结果
    print(f"\n{'=' * 80}")
    print("Saving Results")
    print(f"{'=' * 80}")
    
    # 获取统计信息
    stats = evaluator.get_statistics()
    
    output_data = {
        'config': {
            'csv_file': str(csv_path),
            'image_base': str(image_base),
            'model': args.model,
            'device': args.device,
            'interval': args.interval,
            'max_tokens': args.max_tokens,
            'route_filter': args.route
        },
        'summary': {
            'total_steps_in_csv': len(all_data),
            'filtered_steps': len(filtered_data),
            'processed_steps': processed_count,
            'skipped_steps': skipped_count,
            'routes_processed': list(routes_data.keys())
        },
        'statistics': stats,
        'step_results': all_results
    }
    
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Results saved to: {output_path}")
        
        # 验证文件是否真的被创建
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"[INFO] File created successfully, size: {file_size} bytes")
        else:
            print(f"[ERROR] File was not created at {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results to {output_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 打印统计信息
    print(f"\n{'=' * 80}")
    print("Final Statistics")
    print(f"{'=' * 80}")
    print(f"Total steps in CSV: {len(all_data)}")
    print(f"Filtered steps (every {args.interval}): {len(filtered_data)}")
    print(f"Processed successfully: {processed_count}")
    print(f"Skipped (no image): {skipped_count}")
    
    if stats:
        print(f"\nAlignment Statistics:")
        print(f"  Overall alignment rate: {stats.get('overall_alignment_rate', 0.0):.2%}")
        print(f"  Aligned steps: {stats.get('aligned_steps', 0)} / {stats.get('total_steps', 0)}")
        
        if 'command_alignment_rates' in stats:
            print(f"\n  By Command:")
            for cmd, cmd_stats in sorted(stats['command_alignment_rates'].items()):
                print(f"    Command {cmd} ({cmd_stats['command_str']}): "
                      f"{cmd_stats['alignment_rate']:.2%} "
                      f"({cmd_stats['aligned']}/{cmd_stats['total']})")
    
    print(f"\n{'=' * 80}")
    print("✅ Evaluation completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

