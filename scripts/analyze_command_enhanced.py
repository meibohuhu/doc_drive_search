"""
增强版command分析，使用更多JSON字段来判断
cd /local1/mhu/doc_drive_search && python scripts/analyze_command_enhanced.py
mh 20250130
"""

import json
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
from collections import Counter


def infer_command_enhanced(
    target_point: list,
    target_point_next: list,
    route: list,
    is_in_junction: bool = False,
    angle: float = 0.0,  # 转向角（归一化到[-1, 1]，对应[-90°, 90°]）
    steer: float = 0.0,  # 转向角（归一化到[-1, 1]）
    aim_wp: list = None,  # aim waypoint
    lateral_threshold: float = 1.5,
    turn_angle_threshold: float = 15.0,
    min_distance: float = 7.5,
    max_distance: float = 20.0,
) -> Tuple[int, Dict]:
    """
    增强版command推断，使用更多信息
    """
    target_point = np.array(target_point)
    target_point_next = np.array(target_point_next)
    route = np.array(route) if route else None
    aim_wp = np.array(aim_wp) if aim_wp else None
    
    info = {
        'lateral_displacement_route': 0.0,
        'forward_displacement_route': 0.0,
        'angle_route': 0.0,
        'angle_from_json': float(angle * 90.0),  # 转换为度数
        'steer_from_json': float(steer),
        'reasoning': []
    }
    
    # 使用route分析
    if route is not None and len(route) >= 3:
        # 计算累积距离
        route_distances = [0.0]
        for i in range(1, len(route)):
            diff = route[i] - route[i-1]
            distance = np.linalg.norm(diff[:2])
            route_distances.append(distance)
        cumulative_distances = np.cumsum(route_distances)
        
        # 找到累积距离在min_distance到max_distance范围内的点
        valid_indices = np.where((cumulative_distances >= min_distance) & 
                                 (cumulative_distances <= max_distance))[0]
        
        if len(valid_indices) > 0:
            analysis_start_idx = valid_indices[0]
            analysis_end_idx = min(valid_indices[-1] + 10, len(route) - 1)
        else:
            analysis_end_idx = min(20, len(route) - 1)
            analysis_start_idx = 0
        
        route_segment = route[analysis_start_idx:analysis_end_idx+1]
        
        if len(route_segment) >= 2:
            start_point = route_segment[0]
            end_point = route_segment[-1]
            
            # 计算X和Y的变化
            forward_displacement = end_point[0] - start_point[0]  # X方向（向前）
            lateral_displacement = end_point[1] - start_point[1]  # Y方向（向右）
            
            info['lateral_displacement_route'] = float(lateral_displacement)
            info['forward_displacement_route'] = float(forward_displacement)
            
            # 检测Y值的峰值（用于识别变道）
            # 使用整个route的前20米来分析，计算相对于route起点的Y值
            route_start_y = route[0][1] if len(route) > 0 else start_point[1]
            
            # 使用整个route的前20米，而不是route_segment
            route_distances_full = [0.0]
            for i in range(1, len(route)):
                diff = route[i] - route[i-1]
                distance = np.linalg.norm(diff[:2])
                route_distances_full.append(distance)
            cumulative_distances_full = np.cumsum(route_distances_full)
            
            # 找到前20米的点（用于变道分析）
            # 同时也分析前10米，因为变道可能在更短的距离内完成
            valid_indices_20m = np.where(cumulative_distances_full <= 20.0)[0]
            valid_indices_10m = np.where(cumulative_distances_full <= 10.0)[0]
            
            if len(valid_indices_20m) > 0:
                route_20m = route[:valid_indices_20m[-1]+1]
                y_values_20m = route_20m[:, 1]
            else:
                route_20m = route[:min(20, len(route))]
                y_values_20m = route_20m[:, 1]
            
            if len(valid_indices_10m) > 0:
                route_10m = route[:valid_indices_10m[-1]+1]
                y_values_10m = route_10m[:, 1]
            else:
                route_10m = route[:min(10, len(route))]
                y_values_10m = route_10m[:, 1]
            
            # 计算相对于route起点的Y值（使用20米和10米两个距离）
            y_values_relative_20m = y_values_20m - route_start_y
            y_values_relative_10m = y_values_10m - route_start_y
            
            # 使用20米的数据
            max_y_idx_20m = np.argmax(y_values_relative_20m)
            min_y_idx_20m = np.argmin(y_values_relative_20m)
            max_y_lateral_20m = y_values_relative_20m[max_y_idx_20m]
            min_y_lateral_20m = y_values_relative_20m[min_y_idx_20m]
            
            # 使用10米的数据（变道可能在更短距离内完成）
            max_y_idx_10m = np.argmax(y_values_relative_10m)
            min_y_idx_10m = np.argmin(y_values_relative_10m)
            max_y_lateral_10m = y_values_relative_10m[max_y_idx_10m]
            min_y_lateral_10m = y_values_relative_10m[min_y_idx_10m]
            
            # 选择更合适的值：如果10米内的值在合理范围内，优先使用10米的值
            # 否则使用20米的值
            if abs(max_y_lateral_10m) > 0.8 and abs(max_y_lateral_10m) < 4.5:
                max_y_lateral = max_y_lateral_10m
                max_y_idx = max_y_idx_10m
                max_y_value = y_values_10m[max_y_idx_10m]
            else:
                max_y_lateral = max_y_lateral_20m
                max_y_idx = max_y_idx_20m
                max_y_value = y_values_20m[max_y_idx_20m]
            
            if abs(min_y_lateral_10m) > 0.8 and abs(min_y_lateral_10m) < 4.5:
                min_y_lateral = min_y_lateral_10m
                min_y_idx = min_y_idx_10m
                min_y_value = y_values_10m[min_y_idx_10m]
            else:
                min_y_lateral = min_y_lateral_20m
                min_y_idx = min_y_idx_20m
                min_y_value = y_values_20m[min_y_idx_20m]
            
            # 检查Y值变化模式：变道应该有明显的峰值/谷值，而不是持续向一个方向移动
            # 如果最大值/最小值在起点或终点，说明是持续移动（道路弯曲），不是变道
            # 使用10米的数据来判断变道模式（更准确）
            is_lane_change_pattern = False
            if max_y_idx_10m > 0 and max_y_idx_10m < len(y_values_relative_10m) - 1:  # 最大值不在起点或终点
                # 最大值在中间，说明有向右的峰值
                is_lane_change_pattern = True
            if min_y_idx_10m > 0 and min_y_idx_10m < len(y_values_relative_10m) - 1:  # 最小值不在起点或终点
                # 最小值在中间，说明有向左的峰值
                is_lane_change_pattern = True
            
            info['max_y_lateral'] = float(max_y_lateral)
            info['min_y_lateral'] = float(min_y_lateral)
            info['max_y_value'] = float(y_values_20m[max_y_idx])
            info['min_y_value'] = float(y_values_20m[min_y_idx])
            info['is_lane_change_pattern'] = is_lane_change_pattern
            info['max_y_idx'] = int(max_y_idx)
            info['min_y_idx'] = int(min_y_idx)
            
            # 计算方向角度
            direction_vec_route = end_point - start_point
            if np.linalg.norm(direction_vec_route) > 0.1:
                angle_route = np.arctan2(direction_vec_route[1], direction_vec_route[0])
                angle_route_deg = np.degrees(angle_route)
                info['angle_route'] = float(angle_route_deg)
            else:
                angle_route_deg = 0.0
            
            # 计算X/Y比例
            if abs(forward_displacement) > 0.1:
                lateral_ratio = abs(lateral_displacement) / abs(forward_displacement)
                info['lateral_ratio'] = float(lateral_ratio)
            else:
                lateral_ratio = 0.0
        else:
            forward_displacement = 0.0
            lateral_displacement = 0.0
            angle_route_deg = 0.0
            lateral_ratio = 0.0
            info['reasoning'].append("route segment数据不足，无法分析")
    else:
        forward_displacement = 0.0
        lateral_displacement = 0.0
        angle_route_deg = 0.0
        lateral_ratio = 0.0
        info['reasoning'].append("route数据不足，无法分析")
    
    # 使用angle和steer（从JSON中读取）
    angle_from_json_deg = angle * 90.0  # 转换为度数
    info['angle_from_json'] = float(angle_from_json_deg)
    info['steer_from_json'] = float(steer)
    
    # 检查target_point的Y值（用于识别变道方向）
    target_point_y = target_point[1] if len(target_point) > 1 else 0.0
    info['target_point_y'] = float(target_point_y)
    
    # 检查aim_wp的Y值（作为变道方向的补充判断）
    # aim_wp是相对于车辆当前位置的，所以需要计算相对位置
    if aim_wp is not None and len(aim_wp) > 1 and len(route) > 0:
        vehicle_pos = route[0]
        aim_wp_relative = aim_wp - vehicle_pos
        aim_wp_y = aim_wp_relative[1]  # 相对于车辆的Y值
    else:
        aim_wp_y = 0.0
    info['aim_wp_y'] = float(aim_wp_y)
    
    # 判断command类型
    large_angle_threshold = 30.0
    
    # 1. 转弯判断（在路口时优先级最高）
    if is_in_junction:
        # 优先使用angle_from_json（更准确）
        if abs(angle_from_json_deg) > turn_angle_threshold:
            if angle_from_json_deg > turn_angle_threshold:
                info['reasoning'].append(f"在路口检测到右转：angle={angle_from_json_deg:.1f}° > {turn_angle_threshold}°")
                return 2, info
            elif angle_from_json_deg < -turn_angle_threshold:
                info['reasoning'].append(f"在路口检测到左转：angle={angle_from_json_deg:.1f}° < -{turn_angle_threshold}°")
                return 1, info
        elif abs(angle_route_deg) > turn_angle_threshold:
            if angle_route_deg > turn_angle_threshold:
                info['reasoning'].append(f"在路口检测到右转：route角度={angle_route_deg:.1f}° > {turn_angle_threshold}°")
                return 2, info
            elif angle_route_deg < -turn_angle_threshold:
                info['reasoning'].append(f"在路口检测到左转：route角度={angle_route_deg:.1f}° < -{turn_angle_threshold}°")
                return 1, info
        else:
            info['reasoning'].append(f"在路口检测到直行：角度在阈值内")
            return 3, info
    
    # 2. 如果angle和steer值较大，说明有明显的转向意图（转弯）
    # 但需要检查是否在路口，如果不在路口，需要更严格的条件（横向比例要高）
    if abs(angle_from_json_deg) > turn_angle_threshold:
        if is_in_junction:
            # 在路口时，直接根据angle判断
            if angle_from_json_deg > turn_angle_threshold:
                info['reasoning'].append(f"在路口检测到右转（angle字段）：{angle_from_json_deg:.1f}° > {turn_angle_threshold}°")
                return 2, info
            elif angle_from_json_deg < -turn_angle_threshold:
                info['reasoning'].append(f"在路口检测到左转（angle字段）：{angle_from_json_deg:.1f}° < -{turn_angle_threshold}°")
                return 1, info
        else:
            # 不在路口时，需要检查横向比例，如果横向比例较低，可能是道路弯曲而非转弯
            if 'lateral_ratio' in info:
                if info['lateral_ratio'] > 0.8:  # 横向比例>0.8才判断为转弯
                    if angle_from_json_deg > turn_angle_threshold:
                        info['reasoning'].append(f"检测到右转（angle字段+高横向比例）：{angle_from_json_deg:.1f}° > {turn_angle_threshold}°，横向比例={info['lateral_ratio']:.2f}")
                        return 2, info
                    elif angle_from_json_deg < -turn_angle_threshold:
                        info['reasoning'].append(f"检测到左转（angle字段+高横向比例）：{angle_from_json_deg:.1f}° < -{turn_angle_threshold}°，横向比例={info['lateral_ratio']:.2f}")
                        return 1, info
                else:
                    info['reasoning'].append(f"angle较大（{angle_from_json_deg:.1f}°）但横向比例较低（{info['lateral_ratio']:.2f}），可能是道路弯曲而非转弯，判断为跟随道路")
            else:
                # 如果没有横向比例信息，也判断为跟随道路（保守策略）
                info['reasoning'].append(f"angle较大（{angle_from_json_deg:.1f}°）但不在路口且无横向比例信息，判断为跟随道路")
    
    # 4. 大角度转弯判断（不在路口但角度很大时，优先于变道）
    if abs(angle_route_deg) > large_angle_threshold:
        # 但需要检查X/Y比例，如果X变化远大于Y（比例<0.8），说明主要是向前移动，不是转弯
        if 'lateral_ratio' in info:
            if info['lateral_ratio'] > 0.8:  # Y/X比例>0.8才判断为转弯（Y变化相对较大）
                if angle_route_deg > large_angle_threshold:
                    info['reasoning'].append(f"检测到右转（大角度+高横向比例）：角度={angle_route_deg:.1f}°, 横向比例={info['lateral_ratio']:.2f}")
                    return 2, info
                elif angle_route_deg < -large_angle_threshold:
                    info['reasoning'].append(f"检测到左转（大角度+高横向比例）：角度={angle_route_deg:.1f}°, 横向比例={info['lateral_ratio']:.2f}")
                    return 1, info
            else:
                info['reasoning'].append(f"角度较大但X变化大于Y（比例={info['lateral_ratio']:.2f}），且angle/steer较小，判断为跟随道路")
    
    # 5. 中等角度转弯判断
    if abs(angle_route_deg) > turn_angle_threshold:
        if 'lateral_ratio' in info and info['lateral_ratio'] > 0.8:  # Y/X比例>0.8
            if angle_route_deg > turn_angle_threshold:
                info['reasoning'].append(f"检测到右转：角度={angle_route_deg:.1f}°, 横向比例={info['lateral_ratio']:.2f}")
                return 2, info
            elif angle_route_deg < -turn_angle_threshold:
                info['reasoning'].append(f"检测到左转：角度={angle_route_deg:.1f}°, 横向比例={info['lateral_ratio']:.2f}")
                return 1, info
        elif 'lateral_ratio' in info:
            info['reasoning'].append(f"角度中等但X变化大于Y（比例={info['lateral_ratio']:.2f}），且angle/steer较小，判断为跟随道路")
    
    # 6. 使用angle和steer判断（如果值较小，说明没有明显的转向意图，判断为直行/跟随道路）
    if abs(angle_from_json_deg) < 5.0 and abs(steer) < 0.2:  # angle<5°且steer<0.2，说明转向意图不明显
        if 'lateral_ratio' in info and info['lateral_ratio'] < 0.8:  # 且横向比例<0.8
            info['reasoning'].append(f"angle和steer都很小（angle={angle_from_json_deg:.1f}°, steer={steer:.3f}），且横向比例较低（{info['lateral_ratio']:.2f}），判断为跟随道路")
            return 4, info
    
    # 7. 默认跟随道路
    info['reasoning'].append(f"默认跟随道路：角度={angle_route_deg:.1f}°，横向位移={lateral_displacement:.2f}m，angle={angle_from_json_deg:.1f}°")
    return 4, info


def analyze_command_enhanced(filepath: str) -> Dict:
    """增强版command分析"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    target_point = data.get('target_point', [0, 0])
    target_point_next = data.get('target_point_next', [0, 0])
    route = data.get('route', [])
    command = data.get('command', 4)
    is_in_junction = data.get('junction', False)
    angle = data.get('angle', 0.0)
    steer = data.get('steer', 0.0)
    aim_wp = data.get('aim_wp', None)
    
    # 推断command
    inferred_command, info = infer_command_enhanced(
        target_point, target_point_next, route, is_in_junction,
        angle, steer, aim_wp,
        lateral_threshold=1.5,
        turn_angle_threshold=15.0,
        min_distance=7.5,
        max_distance=25.0
    )
    
    result = {
        'file': filepath,
        'original_command': command,
        'inferred_command': inferred_command,
        'is_in_junction': is_in_junction,
        'angle': angle,
        'steer': steer,
        'target_point': target_point,
        'target_point_next': target_point_next,
        'aim_wp': aim_wp,
        'analysis': info,
    }
    
    return result


def main():
    """批量分析文件夹下所有JSON文件的command"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量分析JSON文件中的command')
    parser.add_argument('--folder', type=str, 
                       default='/local1/mhu/doc_drive_search/data/data_simlingo_training_split_Town12_Rep0_185_route0_01_12_01_39_39',
                       help='要分析的文件夹路径')
    parser.add_argument('--verbose', action='store_true', 
                       help='显示每个文件的详细分析结果')
    parser.add_argument('--save', type=str, default=None,
                       help='保存结果到JSON文件（可选）')
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return
    
    # 查找所有JSON文件
    json_files = sorted(folder_path.glob('*.json'))
    if len(json_files) == 0:
        print(f"错误: 在文件夹 {folder_path} 中未找到JSON文件")
        return
    
    COMMAND_MAP = {
        1: 'go left at the next intersection',
        2: 'go right at the next intersection',
        3: 'go straight at the next intersection',
        4: 'follow the road',
        5: 'do a lane change to the left',
        6: 'do a lane change to the right',
    }
    
    print("=" * 80)
    print("增强版 Command 批量分析（使用angle、steer、X/Y比例等）")
    print("=" * 80)
    print(f"文件夹: {folder_path}")
    print(f"找到 {len(json_files)} 个JSON文件")
    print("=" * 80)
    
    all_results = []
    match_count = 0
    mismatch_count = 0
    error_count = 0
    
    original_command_counter = Counter()
    inferred_command_counter = Counter()
    mismatch_details = []
    
    for filepath in json_files:
        try:
            result = analyze_command_enhanced(str(filepath))
            all_results.append(result)
            
            original_cmd = result['original_command']
            inferred_cmd = result['inferred_command']
            
            original_command_counter[original_cmd] += 1
            inferred_command_counter[inferred_cmd] += 1
            
            if original_cmd == inferred_cmd:
                match_count += 1
            else:
                mismatch_count += 1
                mismatch_details.append({
                    'file': Path(result['file']).name,
                    'original': original_cmd,
                    'inferred': inferred_cmd,
                    'original_str': COMMAND_MAP[original_cmd],
                    'inferred_str': COMMAND_MAP[inferred_cmd],
                    'is_in_junction': result['is_in_junction'],
                    'angle': result['angle'],
                    'steer': result['steer'],
                })
            
            if args.verbose:
                print(f"\n文件: {Path(result['file']).name}")
                print("-" * 80)
                print(f"原始 command: {original_cmd} ({COMMAND_MAP[original_cmd]})")
                print(f"推断 command: {inferred_cmd} ({COMMAND_MAP[inferred_cmd]})")
                print(f"是否在路口: {result['is_in_junction']}")
                print(f"angle (JSON): {result['angle']:.4f} ({result['analysis']['angle_from_json']:.1f}°)")
                print(f"steer (JSON): {result['steer']:.4f}")
                
                if 'lateral_displacement_route' in result['analysis']:
                    print(f"横向位移(Y): {result['analysis']['lateral_displacement_route']:.2f}m")
                if 'forward_displacement_route' in result['analysis']:
                    print(f"向前位移(X): {result['analysis']['forward_displacement_route']:.2f}m")
                if 'lateral_ratio' in result['analysis']:
                    print(f"横向比例(Y/X): {result['analysis']['lateral_ratio']:.2f}")
                if 'angle_route' in result['analysis']:
                    print(f"角度(route): {result['analysis']['angle_route']:.1f}°")
                if 'target_point_y' in result['analysis']:
                    print(f"target_point Y: {result['analysis']['target_point_y']:.2f}")
                if 'max_y_lateral' in result['analysis']:
                    print(f"Y峰值横向位移: {result['analysis']['max_y_lateral']:.2f}m")
                if 'min_y_lateral' in result['analysis']:
                    print(f"Y谷值横向位移: {result['analysis']['min_y_lateral']:.2f}m")
                
                print(f"\n推理过程:")
                for reason in result['analysis'].get('reasoning', []):
                    print(f"  - {reason}")
                
                if original_cmd != inferred_cmd:
                    print(f"\n⚠️  WARNING: 不匹配！")
                else:
                    print(f"\n✓ 匹配")
        except Exception as e:
            error_count += 1
            print(f"错误处理文件 {filepath.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    print(f"总文件数: {len(json_files)}")
    print(f"成功处理: {len(all_results)}")
    print(f"错误: {error_count}")
    print(f"匹配: {match_count} ({match_count/len(all_results)*100:.1f}%)")
    print(f"不匹配: {mismatch_count} ({mismatch_count/len(all_results)*100:.1f}%)")
    
    print("\n原始command分布:")
    for cmd_id, count in sorted(original_command_counter.items()):
        print(f"  {cmd_id} ({COMMAND_MAP[cmd_id]}): {count}")
    
    print("\n推断command分布:")
    for cmd_id, count in sorted(inferred_command_counter.items()):
        print(f"  {cmd_id} ({COMMAND_MAP[cmd_id]}): {count}")
    
    if mismatch_count > 0:
        print(f"\n不匹配的文件详情 ({len(mismatch_details)} 个):")
        print("-" * 80)
        for detail in mismatch_details:
            print(f"  {detail['file']}: {detail['original']} ({detail['original_str']}) -> "
                  f"{detail['inferred']} ({detail['inferred_str']}) | "
                  f"junction={detail['is_in_junction']}, angle={detail['angle']:.3f}, steer={detail['steer']:.3f}")
    
    # 保存结果
    if args.save:
        output_path = Path(args.save)
        output_data = {
            'folder': str(folder_path),
            'total_files': len(json_files),
            'processed': len(all_results),
            'errors': error_count,
            'match_count': match_count,
            'mismatch_count': mismatch_count,
            'match_rate': match_count/len(all_results) if len(all_results) > 0 else 0,
            'original_command_distribution': dict(original_command_counter),
            'inferred_command_distribution': dict(inferred_command_counter),
            'mismatch_details': mismatch_details,
            'all_results': all_results if args.verbose else None,  # 只在verbose时保存详细结果
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == '__main__':
    main()

