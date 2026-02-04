# Route Waypoints可视化工具

## 功能

在RGB前视图像上可视化route waypoints和target point，参照`agent_simlingo.py`中的可视化方法。

## 使用方法

### 1. 单文件可视化

```bash
python scripts/visualize_route_waypoints.py \
    --image /path/to/rgb/0001.jpg \
    --measurement /path/to/measurements/0001.json.gz \
    --output /path/to/output.png
```

### 2. 批量可视化

```bash
# 自动查找匹配的文件
python scripts/batch_visualize_route.py \
    --rgb-folder /path/to/rgb \
    --output-folder /path/to/output

# 指定measurement文件夹
python scripts/batch_visualize_route.py \
    --rgb-folder /path/to/rgb \
    --measurement-folder /path/to/measurements \
    --output-folder /path/to/output

# 限制处理文件数（用于测试）
python scripts/batch_visualize_route.py \
    --rgb-folder /path/to/rgb \
    --max-files 10
```

## 可视化内容

- **蓝色大圆点**: Target Point（目标点）
- **青色圆点**: Aim WP（瞄准航点）
- **红色小圆点**: Route Waypoints（路径航点）
- **红色连线**: 连接route waypoints，显示路径轨迹
- **信息文本**: 在图像底部显示target point坐标、route点数、command等信息
- **图例**: 在图像左上角显示图例说明

## 示例

```bash
# 可视化单个文件
cd /local1/mhu/doc_drive_search
python scripts/visualize_route_waypoints.py \
    --image data/simlingo_dataset/.../rgb/0001.jpg \
    --measurement data/simlingo_dataset/.../measurements/0001.json.gz

# 批量可视化整个文件夹
python scripts/batch_visualize_route.py \
    --rgb-folder data/simlingo_dataset/.../rgb \
    --output-folder output/visualizations
```

## 参数说明

### visualize_route_waypoints.py

- `--image`: RGB图像路径（必需）
- `--measurement`: measurement JSON文件路径，支持`.json`和`.json.gz`（必需）
- `--output`: 输出图像路径（可选，默认保存到图像同目录）
- `--fov`: 相机FOV，默认110度

### batch_visualize_route.py

- `--rgb-folder`: RGB图像文件夹路径（必需）
- `--measurement-folder`: measurement文件夹路径（可选，会自动查找）
- `--output-folder`: 输出文件夹路径（可选，默认保存到图像同目录）
- `--max-files`: 最大处理文件数（可选，用于测试）

## 技术细节

- 使用`project_points`函数将3D航点投影到2D图像坐标
- 相机内参根据图像尺寸和FOV自动计算
- 相机外参使用默认配置（参照agent_simlingo.py中的HD_VIZ配置）
- 支持压缩的`.json.gz`文件

## 注意事项

1. 确保图像和measurement文件存在且匹配（文件名对应）
2. 如果waypoints在图像范围外，可能不会显示
3. 相机参数可能需要根据实际数据调整

