"""
可视化route waypoints和target point在前视图像上
参照agent_simlingo.py中的可视化方法
python scripts/visualize_route_waypoints.py --image /local1/mhu/doc_drive_search/data/simlingo_dataset/database/simlingo_extracted/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_17_route0_01_10_22_54_33/rgb/0180.jpg --measurement /local1/mhu/doc_drive_search/data/simlingo_dataset/database/simlingo_extracted/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_17_route0_01_10_22_54_33/measurements/0180.json.gz --output /local1/mhu/doc_drive_search/data/simlingo_dataset/database/simlingo_extracted/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_17_route0_01_10_22_54_33/rgb/test_180.png  --use-equal-spacing
"""

import gzip
import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
import argparse

# 添加路径以导入工具函数
sys.path.append(str(Path(__file__).parent.parent))
from team_code.simlingo_utils import (
    project_points,
    get_rotation_matrix,
    get_camera_intrinsics,
)


def load_measurement(measurement_path: str) -> dict:
    """加载measurement JSON文件（支持.gz压缩）"""
    if measurement_path.endswith('.gz'):
        with gzip.open(measurement_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(measurement_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def equal_spacing_route(points: np.ndarray, num_points: int = 20) -> np.ndarray:
    """
    对route进行等间距处理，匹配训练时的数据处理流程
    在route前面加一个(0,0)点，然后插值生成等间距的点（每1米一个点）
    
    Args:
        points: 原始route点，格式 [N, 2]，可能是从某个距离开始的（如7.49m）
        num_points: 输出的点数（默认20）
    
    Returns:
        处理后的route点，格式 [num_points, 2]，从(0,0)开始
    """
    if len(points) == 0:
        return points
    
    # 在route前面加一个(0,0)点
    route = np.concatenate((np.zeros_like(points[:1]), points))
    shift = np.roll(route, 1, axis=0)
    shift[0] = shift[1]
    
    # 计算累积距离
    dists = np.linalg.norm(route - shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(0, len(dists)) * 1e-4  # 防止距离不严格递增
    
    # 插值生成等间距的点（每1米一个点）
    x = np.arange(0, num_points, 1)
    interp_points = np.array([
        np.interp(x, dists, route[:, 0]),
        np.interp(x, dists, route[:, 1])
    ]).T
    
    return interp_points


def visualize_waypoints_on_image(
    image_path: str,
    measurement_path: str,
    output_path: str = None,
    camera_config: dict = None,
    tvec_z_offset: float = None,
    use_equal_spacing: bool = False
):
    """
    在图像上可视化route waypoints和target point
    
    Args:
        image_path: RGB图像路径
        measurement_path: measurement JSON文件路径
        output_path: 输出图像路径（可选）
        camera_config: 相机配置（可选，用于自定义相机参数）
    """
    # 加载图像（完全模拟agent_simlingo.py的处理流程）
    # 步骤1: 读取图像（模拟input_data['rgb_0'][1]）
    camera = cv2.imread(image_path)
    if camera is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    
    # 确保是BGR格式（3通道）
    if len(camera.shape) == 3:
        camera = camera[:, :, :3]
    
    # 步骤2: 保存用于可视化的图像（在裁剪之前，模拟第398行）
    camera_for_viz = camera.copy()
    
    # 步骤3: JPEG压缩/解压缩（模拟第401-402行）
    # 这确保图像处理与agent_simlingo.py一致
    _, compressed_image_i = cv2.imencode('.jpg', camera)
    camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)
    
    # 步骤4: 在agent_simlingo.py中，可视化时使用camera_for_viz（原始未裁剪图像）
    # 步骤5: BGR转RGB（模拟第778行）
    image_rgb = cv2.cvtColor(camera_for_viz, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    
    # 重要：在agent_simlingo.py中，camera_for_viz是在裁剪之前保存的
    # 可视化时使用原始未裁剪图像的尺寸来计算相机内参（第773-775行）
    # 所以这里我们使用原始图像尺寸，不进行裁剪
    
    # 加载measurement数据
    measurement = load_measurement(measurement_path)
    
    # 提取route和target_point
    route = measurement.get('route', [])
    route_original = measurement.get('route_original', route)
    target_point = measurement.get('target_point', None)
    aim_wp = measurement.get('aim_wp', None)
    
    # 如果没有route，尝试从route_original获取
    if not route and route_original:
        route = route_original
    
    # 转换为numpy数组
    if route:
        route = np.array(route)
        # 如果启用equal_spacing处理，匹配训练时的数据处理流程
        if use_equal_spacing:
            route = equal_spacing_route(route, num_points=20)
    if target_point:
        target_point = np.array(target_point)
    if aim_wp:
        aim_wp = np.array(aim_wp)
    
    # 相机配置（参照agent_simlingo.py和config_simlingo.py）
    if camera_config is None:
        # 默认相机配置（front camera，参照config_simlingo.py）
        # camera_pos_0 = [-1.5, 0.0, 2.0]  # x, y, z
        # camera_rot_0 = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        # 对于普通前视相机，不使用HD_VIZ配置，让project_points使用默认值
        camera_config = {
            'fov': 110,
            'tvec': None,  # None会使用project_points的默认值 [0.0, 2.0, 1.5]
            'rvec': None,  # None会使用project_points的默认值（零旋转）
        }
    
    # 计算相机内参
    camera_intrinsics = get_camera_intrinsics(W, H, camera_config['fov'])
    camera_intrinsics_np = camera_intrinsics.numpy()
    
    # 计算旋转矩阵和rvec（如果指定了）
    if camera_config.get('cam_rots') is not None:
        rot_matrix = get_rotation_matrix(
            -camera_config['cam_rots'][0],
            -camera_config['cam_rots'][1],
            camera_config['cam_rots'][2]
        )
        rvec = cv2.Rodrigues(rot_matrix[:3, :3])[0].flatten()
    else:
        rvec = camera_config.get('rvec')
    
    tvec = camera_config.get('tvec')
    
    # 重要：在agent_simlingo.py中，当不使用HD_VIZ时，tvec和rvec都是None
    # 这意味着project_points会使用默认值：tvec=[0.0, 2.0, 1.5], rvec=零旋转
    # 
    # 根据测试，使用原始的project_points（Y=0）时，waypoints投影到Y=324左右（图像下半部分，道路上）
    # 这与agent_simlingo.py的行为应该是一致的
    # 
    # 直接使用原始的project_points函数，与agent_simlingo.py完全一致
    from team_code.simlingo_utils import project_points as project_points_func
    
    if tvec is None:
        if tvec_z_offset is not None:
            tvec = np.array([[0.0, 2.0, tvec_z_offset]], np.float32)
        else:
            # 保持None，让project_points使用默认值[0.0, 2.0, 1.5]
            # 这与agent_simlingo.py的行为完全一致
            tvec = None
    
    # 创建PIL图像用于绘制
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # 绘制target_point（蓝色，较大）
    if target_point is not None:
        target_points_2d = project_points_func(
            [target_point],
            camera_intrinsics_np,
            tvec=tvec,
            rvec=rvec
        )
        for point_2d in target_points_2d:
            x, y = int(point_2d[0]), int(point_2d[1])
            # 绘制较大的蓝色圆点
            draw.ellipse((x-6, y-6, x+6, y+6), fill=(0, 0, 255, 255), outline=(255, 255, 255, 255), width=2)
            # 添加标签
            draw.text((x+8, y-8), "Target", fill=(255, 255, 255, 255))
    
    # 绘制aim_wp（青色）
    if aim_wp is not None:
        aim_wp_2d = project_points_func(
            [aim_wp],
            camera_intrinsics_np,
            tvec=tvec,
            rvec=rvec
        )
        for point_2d in aim_wp_2d:
            x, y = int(point_2d[0]), int(point_2d[1])
            draw.ellipse((x-5, y-5, x+5, y+5), fill=(0, 255, 255, 255), outline=(255, 255, 255, 255), width=2)
            draw.text((x+8, y-8), "Aim", fill=(255, 255, 255, 255))
    
    # 绘制route waypoints（红色，较小）
    # 只可视化前20个waypoints
    if route is not None and len(route) > 0:
        # 限制只使用前20个waypoints
        route_to_visualize = route[:20] if len(route) > 20 else route
        route_points_2d = project_points_func(
            route_to_visualize.tolist(),
            camera_intrinsics_np,
            tvec=tvec,
            rvec=rvec
        )
        
        # 绘制点（只绘制在图像范围内的点）
        valid_points = []
        for i, point_2d in enumerate(route_points_2d):
            x, y = int(point_2d[0]), int(point_2d[1])
            
            # 检查点是否在图像范围内
            if 0 <= x < W and 0 <= y < H:
                # 根据距离调整点的大小和颜色，让距离更明显
                wp_distance = route_to_visualize[i][0]  # 前方距离
                
                # 距离范围：假设前20个点在7-30米左右
                min_dist = route_to_visualize[:, 0].min() if len(route_to_visualize) > 0 else 7.0
                max_dist = route_to_visualize[:, 0].max() if len(route_to_visualize) > 0 else 30.0
                
                # 归一化距离（0=最近，1=最远）
                if max_dist > min_dist:
                    distance_normalized = (wp_distance - min_dist) / (max_dist - min_dist)
                else:
                    distance_normalized = 0.0
                distance_normalized = np.clip(distance_normalized, 0, 1)
                
                # 点的大小：近的点大（5像素），远的点小（2像素）
                point_size = int(5 - distance_normalized * 3)
                point_size = max(2, min(5, point_size))
                
                # 颜色：近的点更亮（255, 0, 0），远的点更暗（180, 0, 0）
                color_intensity = int(255 - distance_normalized * 75)
                color = (color_intensity, 0, 0, 255)
                
                # 绘制点
                draw.ellipse((x-point_size, y-point_size, x+point_size, y+point_size), fill=color)
                valid_points.append((x, y))
            # 如果点不在图像范围内，跳过（不绘制）
        
        # 调试信息：打印前几个点的坐标和使用的参数（可选，可以通过参数控制）
        # 注释掉以减少输出
        # if len(route_points_2d) > 0:
        #     print(f"\n[调试] 图像尺寸: W={W}, H={H}")
        #     print(f"[调试] 相机内参: {camera_intrinsics_np}")
        #     print(f"[调试] tvec: {tvec}, rvec: {rvec}")
        #     print(f"[调试] 前5个waypoints的投影坐标:")
        #     for i in range(min(5, len(route_points_2d))):
        #         wp = route_to_visualize[i]
        #         proj = route_points_2d[i]
        #         print(f"  Waypoint {i}: [{wp[0]:.2f}, {wp[1]:.6f}] -> [{proj[0]:.1f}, {proj[1]:.1f}]")
        
        # 绘制连线（只连接有效点）
        for i in range(1, len(valid_points)):
            prev_x, prev_y = valid_points[i-1]
            curr_x, curr_y = valid_points[i]
            draw.line(
                [(prev_x, prev_y), (curr_x, curr_y)],
                fill=(255, 100, 100, 200),
                width=2
            )
        
        # 在第一个和最后一个有效点添加标签
        if len(valid_points) > 0:
            first_x, first_y = valid_points[0]
            last_x, last_y = valid_points[-1]
            draw.text((first_x+8, first_y-8), "Route[0]", fill=(255, 255, 255, 255))
            # 显示实际绘制的最后一个点的索引（最多19，因为只绘制前20个）
            last_idx = min(19, len(route_to_visualize)-1)
            draw.text((last_x+8, last_y-8), f"Route[{last_idx}]", fill=(255, 255, 255, 255))
    
    # 添加信息文本
    info_text = []
    if target_point is not None:
        info_text.append(f"Target: [{target_point[0]:.2f}, {target_point[1]:.2f}]")
    if aim_wp is not None:
        info_text.append(f"Aim WP: [{aim_wp[0]:.2f}, {aim_wp[1]:.2f}]")
    if route is not None:
        info_text.append(f"Route points: {len(route)}")
    
    command = measurement.get('command', 4)
    command_map = {
        1: 'go left at intersection',
        2: 'go right at intersection',
        3: 'go straight at intersection',
        4: 'follow the road',
        5: 'lane change left',
        6: 'lane change right',
    }
    info_text.append(f"Command: {command} ({command_map.get(command, 'unknown')})")
    
    # 在图像底部绘制信息
    y_offset = H - 100
    for i, text in enumerate(info_text):
        draw.text((10, y_offset + i * 20), text, fill=(255, 255, 255, 255))
    
    # 添加图例
    legend_y = 10
    draw.ellipse((10, legend_y, 16, legend_y+6), fill=(0, 0, 255, 255))  # Target (blue)
    draw.text((20, legend_y-2), "Target Point", fill=(255, 255, 255, 255))
    
    draw.ellipse((10, legend_y+15, 15, legend_y+21), fill=(0, 255, 255, 255))  # Aim (cyan)
    draw.text((20, legend_y+13), "Aim WP", fill=(255, 255, 255, 255))
    
    draw.ellipse((10, legend_y+30, 13, legend_y+36), fill=(255, 0, 0, 255))  # Route (red)
    draw.text((20, legend_y+28), "Route Waypoints", fill=(255, 255, 255, 255))
    
    # 保存或显示
    if output_path:
        pil_image.save(output_path)
        print(f"可视化结果已保存到: {output_path}")
    else:
        # 如果没有指定输出路径，保存到图像同目录
        output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_visualized.png")
        pil_image.save(output_path)
        print(f"可视化结果已保存到: {output_path}")
    
    return pil_image


def main():
    parser = argparse.ArgumentParser(description='在图像上可视化route waypoints和target point')
    parser.add_argument('--image', type=str, required=True,
                       help='RGB图像路径')
    parser.add_argument('--measurement', type=str, required=True,
                       help='measurement JSON文件路径（支持.gz压缩）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径（可选，默认保存到图像同目录）')
    parser.add_argument('--fov', type=float, default=110,
                       help='相机FOV（默认110）')
    parser.add_argument('--tvec-z', type=float, default=None,
                       help='tvec的z分量偏移（用于调整waypoints高度，默认0.0）')
    parser.add_argument('--use-equal-spacing', action='store_true',
                       help='对route进行equal_spacing处理，匹配训练时的label格式（从(0,0)开始，每1米一个点）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.measurement):
        print(f"错误: measurement文件不存在: {args.measurement}")
        return
    
    # 相机配置（默认使用普通前视相机，与agent_simlingo.py一致）
    # 如果要使用HD_VIZ配置，可以取消下面的注释
    camera_config = {
        'fov': args.fov,
        'tvec': None,  # None会使用project_points的默认值 [0.0, 2.0, 1.5]
        'rvec': None,  # None会使用project_points的默认值（零旋转）
    }
    # HD_VIZ配置（如果需要，取消下面的注释并注释掉上面的配置）
    # camera_config = {
    #     'fov': args.fov,
    #     'tvec': np.array([[0.0, 3.5, 5.5]], np.float32),
    #     'cam_rots': [0.0, -15.0, 0.0],
    # }
    
    # 执行可视化
    try:
        visualize_waypoints_on_image(
            args.image,
            args.measurement,
            args.output,
            camera_config,
            tvec_z_offset=args.tvec_z,
            use_equal_spacing=args.use_equal_spacing
        )
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

