#!/usr/bin/env python3
"""
通过token查找对应的scene、route和instruction (使用VAD数据)

本脚本使用VAD目录下的pkl文件，可以显示更丰富的帧信息，包括:
- 控制命令 (gt_ego_fut_cmd)
- 地图位置 (map_location)
- 未来轨迹 (gt_ego_fut_trajs)
- Agent信息 (gt_agent_fut_trajs)

使用方法:
    python find_token_scene_route.py <token>
    python find_token_scene_route.py --scene <scene_number>
    
示例:
    python find_token_scene_route.py e93e98b63d3b40209056d129dc53ceee
    python find_token_scene_route.py --scene 1
"""

import pickle
import pandas as pd
import sys

# 使用VAD目录下的pkl文件
TRAIN_PKL = '/shared/rc/llm-gen-agent/mhu/fsdrive_dataset/infos/vad/vad_nuscenes_infos_temporal_train.pkl'
VAL_PKL = '/shared/rc/llm-gen-agent/mhu/fsdrive_dataset/infos/vad/vad_nuscenes_infos_temporal_val.pkl'
CSV_PATH = '/shared/rc/llm-gen-agent/mhu/fsdrive_dataset/annotated_doscenes.csv'


def build_scene_mapping():
    """建立scene_token到Scene Number的映射"""
    print('正在加载数据...')
    
    # 读取pickle文件
    with open(TRAIN_PKL, 'rb') as f:
        train_data = pickle.load(f)
    with open(VAL_PKL, 'rb') as f:
        val_data = pickle.load(f)
    
    # 读取CSV
    df = pd.read_csv(CSV_PATH)
    
    # 收集所有scene_token，按出现顺序
    scene_tokens_ordered = []
    scene_info = {}
    
    # 先处理训练集
    for info in train_data['infos']:
        scene_token = info['scene_token']
        if scene_token not in scene_info:
            scene_tokens_ordered.append(scene_token)
            scene_info[scene_token] = {
                'first_token': info['token'],
                'first_frame_idx': info['frame_idx'],
                'frame_count': 0,
                'dataset': 'train'
            }
        scene_info[scene_token]['frame_count'] += 1
    
    # 再处理验证集
    for info in val_data['infos']:
        scene_token = info['scene_token']
        if scene_token not in scene_info:
            scene_tokens_ordered.append(scene_token)
            scene_info[scene_token] = {
                'first_token': info['token'],
                'first_frame_idx': info['frame_idx'],
                'frame_count': 0,
                'dataset': 'val'
            }
        scene_info[scene_token]['frame_count'] += 1
    
    # 建立映射：scene_token -> Scene Number (从1开始)
    scene_token_to_number = {}
    for idx, scene_token in enumerate(scene_tokens_ordered):
        scene_number = idx + 1
        scene_token_to_number[scene_token] = scene_number
    
    # 建立token到scene_token的映射
    token_to_scene = {}
    for info in train_data['infos'] + val_data['infos']:
        token_to_scene[info['token']] = info['scene_token']
    
    return scene_token_to_number, scene_info, token_to_scene, df


def find_token_info(token, scene_token_to_number, scene_info, token_to_scene, df, vad_data=None):
    """通过token查找scene、route和instruction信息"""
    
    print('=' * 70)
    print('查找Token信息 (使用VAD数据)')
    print('=' * 70)
    print(f'Token: {token}')
    print()
    
    # 1. 通过token找到scene_token
    if token not in token_to_scene:
        print('❌ 未找到该token')
        return None
    
    scene_token = token_to_scene[token]
    print(f'✅ 找到对应的scene_token: {scene_token}')
    
    # 2. 通过scene_token找到Scene Number
    if scene_token not in scene_token_to_number:
        print('❌ 未找到对应的Scene Number')
        return None
    
    scene_number = scene_token_to_number[scene_token]
    print(f'✅ Scene Number: {scene_number}')
    
    # 3. 获取scene信息
    scene_data = scene_info[scene_token]
    print(f'✅ 数据集: {scene_data["dataset"]}')
    print(f'✅ 该scene的帧数: {scene_data["frame_count"]}')
    print()
    
    # 4. 从VAD数据中获取该帧的详细信息（如果提供）
    frame_info = None
    if vad_data:
        for info in vad_data['infos']:
            if info['token'] == token:
                frame_info = info
                break
        
        if frame_info:
            print('✅ 找到该帧的详细信息 (VAD数据):')
            print(f'   Frame Index: {frame_info["frame_idx"]}')
            print(f'   Prev Token: {frame_info["prev"] if frame_info["prev"] else "None"}')
            print(f'   Next Token: {frame_info["next"] if frame_info["next"] else "None"}')
            
            # 显示VAD特有字段
            if 'gt_ego_fut_cmd' in frame_info:
                cmd = frame_info['gt_ego_fut_cmd']
                cmd_names = {
                    (0.0, 1.0, 0.0): '直行',
                    (0.0, 0.0, 1.0): '右转/右变道',
                    (1.0, 0.0, 0.0): '左转/左变道'
                }
                cmd_tuple = tuple(cmd.tolist()) if hasattr(cmd, 'tolist') else tuple(cmd)
                cmd_name = cmd_names.get(cmd_tuple, f'未知命令 {cmd_tuple}')
                print(f'   控制命令: {cmd_name} {cmd_tuple}')
            
            if 'map_location' in frame_info:
                print(f'   地图位置: {frame_info["map_location"]}')
            
            if 'gt_ego_fut_trajs' in frame_info:
                traj = frame_info['gt_ego_fut_trajs']
                print(f'   未来轨迹点数: {len(traj)}')
            
            if 'gt_agent_fut_trajs' in frame_info:
                agent_trajs = frame_info['gt_agent_fut_trajs']
                print(f'   Agent数量: {agent_trajs.shape[0] if hasattr(agent_trajs, "shape") else len(agent_trajs)}')
            print()
    
    # 5. 从CSV中获取该scene的所有指令
    scene_instructions = df[df['Scene Number'] == scene_number]
    
    if len(scene_instructions) > 0:
        print(f'✅ 该scene有 {len(scene_instructions)} 条指令:')
        print()
        for idx, row in scene_instructions.iterrows():
            inst_type = row['Instruction Type'] if pd.notna(row['Instruction Type']) else 'N/A'
            instruction = row['Instruction']
            print(f'  指令 {idx+1}:')
            print(f'    类型: {inst_type}')
            print(f'    内容: {instruction}')
            print()
    else:
        print('⚠️  该scene在CSV中没有指令')
        print()
    
    return {
        'token': token,
        'scene_token': scene_token,
        'scene_number': scene_number,
        'dataset': scene_data['dataset'],
        'frame_count': scene_data['frame_count'],
        'instructions': scene_instructions.to_dict('records') if len(scene_instructions) > 0 else [],
        'frame_info': frame_info
    }


def find_by_scene_number(scene_number, scene_token_to_number, scene_info, df):
    """通过Scene Number查找信息"""
    scene_token = None
    for token, number in scene_token_to_number.items():
        if number == scene_number:
            scene_token = token
            break
    
    if scene_token is None:
        print(f'❌ 未找到Scene Number {scene_number}')
        return None
    
    print(f'✅ Scene Number {scene_number} 对应的scene_token: {scene_token}')
    print(f'   数据集: {scene_info[scene_token]["dataset"]}')
    print(f'   帧数: {scene_info[scene_token]["frame_count"]}')
    
    # 获取指令
    scene_instructions = df[df['Scene Number'] == scene_number]
    if len(scene_instructions) > 0:
        print(f'   指令数: {len(scene_instructions)}')
        for idx, row in scene_instructions.iterrows():
            print(f'     - {row["Instruction"]}')
    
    return scene_token


if __name__ == '__main__':
    # 建立映射
    scene_token_to_number, scene_info, token_to_scene, df = build_scene_mapping()
    
    # 加载VAD数据用于显示详细信息
    print('正在加载VAD数据...')
    with open(TRAIN_PKL, 'rb') as f:
        vad_train_data = pickle.load(f)
    with open(VAL_PKL, 'rb') as f:
        vad_val_data = pickle.load(f)
    # 合并VAD数据
    vad_all_data = {
        'infos': vad_train_data['infos'] + vad_val_data['infos'],
        'metadata': vad_train_data['metadata']
    }
    
    print(f'已加载 {len(scene_token_to_number)} 个scene')
    print(f'已加载 {len(token_to_scene)} 个token')
    print(f'已加载 {len(vad_all_data["infos"])} 个VAD帧')
    print(f'CSV中有 {len(df)} 条指令记录')
    print()
    
    if len(sys.argv) < 2:
        print('使用方法:')
        print('  python find_token_scene_route.py <token>')
        print('  python find_token_scene_route.py --scene <scene_number>')
        print()
        print('示例:')
        print('  python find_token_scene_route.py e93e98b63d3b40209056d129dc53ceee')
        print('  python find_token_scene_route.py --scene 1')
        sys.exit(1)
    
    if sys.argv[1] == '--scene':
        if len(sys.argv) < 3:
            print('请提供Scene Number')
            sys.exit(1)
        scene_number = float(sys.argv[2])
        find_by_scene_number(scene_number, scene_token_to_number, scene_info, df)
    else:
        token = sys.argv[1]
        result = find_token_info(token, scene_token_to_number, scene_info, token_to_scene, df, vad_all_data)
        
        if result:
            print('=' * 70)
            print('总结')
            print('=' * 70)
            print(f'Token: {result["token"]}')
            print(f'Scene Token: {result["scene_token"]}')
            print(f'Scene Number: {result["scene_number"]} (对应CSV中的Scene Number)')
            print(f'Route: scene_token就是route/scene的唯一标识')
            print(f'指令数: {len(result["instructions"])}')
            if result.get('frame_info'):
                print(f'VAD数据: ✅ 已加载该帧的详细信息')

