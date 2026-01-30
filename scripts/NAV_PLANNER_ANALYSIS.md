# Nav Planner 分析：target_speed 和 target_point 计算

## 概述

`nav_planner.py` 提供了路径规划和控制的工具类，主要用于计算 `target_point`（目标点）和 `target_speed`（目标速度）。这些值在**训练数据收集**和**推理**两个阶段都会使用，但计算方式有所不同。

## target_speed 计算流程

### 在 autopilot.py（训练数据收集）中

```python
# 1. 从路径规划器获取速度限制
route_np, route_wp, _, distance_to_next_traffic_light, next_traffic_light, 
    distance_to_next_stop_sign, next_stop_sign, speed_limit = \
    self._waypoint_planner.run_step(ego_position)

# 2. 基础目标速度
target_speed = min(speed_limit * self.config.ratio_target_speed_limit, 72. / 3.6)

# 3. 检查前方是否有路口，降低速度
for i in range(min(self.config.max_lookahead_to_check_for_junction, len(route_wp))):
    if route_wp[i].is_junction:
        target_speed = min(target_speed, self.config.max_speed_in_junction)
        break

# 4. 处理路线障碍物场景（车辆、交通灯、停止标志等）
target_speed_route_obstacle, keep_driving, speed_reduced_by_obj = \
    self._manage_route_obstacle_scenarios(
        target_speed, ego_speed, route_wp, vehicles, route_np
    )

# 5. 使用 IDM（Intelligent Driver Model）和运动学自行车模型
brake, target_speed, speed_reduced_by_obj = self.get_brake_and_target_speed(
    plant, route_np, distance_to_next_traffic_light, next_traffic_light,
    distance_to_next_stop_sign, next_stop_sign, vehicles, actors, 
    target_speed, speed_reduced_by_obj
)

# 6. 取最小值（最保守的速度）
target_speed = min(target_speed, target_speed_route_obstacle)
```

## 坐标单位说明

### 问题：JSON 和 XML 文件中的坐标数字单位是什么？是厘米还是米？

**答案：所有坐标单位都是米（meters，m）**

### 1. XML 文件中的坐标（CARLA 世界坐标系）

```xml
<position x="-497.6" y="3672.9" z="364.9"/>
```

- **坐标系**：CARLA 世界坐标系（全局坐标系）
- **单位**：**米（m）**
- **含义**：
  - `x="-497.6"` 表示在世界坐标系中 x 方向 -497.6 米
  - `y="3672.9"` 表示在世界坐标系中 y 方向 3672.9 米
  - `z="364.9"` 表示高度 364.9 米

### 2. JSON 文件中的坐标（车辆坐标系）

```json
{
    "pos_global": [-1130.51, 3738.84],        // 全局位置（米）
    "target_point": [51.54, 0.004],           // 目标点（车辆坐标系，米）
    "route": [
        [7.49, 0.0006],                       // 路径点（车辆坐标系，米）
        [8.49, 0.0007],
        ...
    ]
}
```
- **坐标系**：车辆坐标系（以车辆当前位置为原点）
- **单位**：**米（m）**
- **含义**：
  - `target_point: [51.54, 0.004]` 表示目标点相对于车辆当前位置，前方 51.54 米，左侧 0.004 米
  - `route: [7.49, 0.0006]` 表示路径点相对于车辆当前位置（`pos_global`），前方 7.49 米，左侧 0.0006 米
  - 路径点间距约 1 米（符合路径规划中 1 米间距的设置）
  
**重要**：`route` 中的坐标是**相对于车辆当前位置（`pos_global`）**的距离和方向，不是世界坐标系中的绝对位置。

### 3. 代码证据
```python
# 在 agent_simlingo.py 中
self.logger_region_of_interest = 30.0  # Meters around the car

# 在 prompt 中
prompt = f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
# speed 单位是 m/s（米每秒）

# 在 transfuser_utils.py 中
# Multiply position and extent by pixels_per_meter to convert the unit from meters to pixels
# Divide position and extent by pixels_per_meter to convert the unit from pixels to meters
# 明确说明基础单位是 meters（米）
```

### 4. 数值合理性验证

从 JSON 文件中的数值可以验证单位是米：

- `target_point: [51.54, 0.004]`
  - 51.54 米前方：合理的目标点距离
  - 0.004 米侧向偏移：约 4 毫米，合理的微小偏移

- `route: [[7.49, 0.0006], [8.49, 0.0007], ...]`
  - 路径点间距约 1 米（8.49 - 7.49 = 1.0）：符合路径规划中 1 米间距的设置
  - 如果单位是厘米，间距会是 1 厘米，这太小了

- `speed: 0.174` 和 `target_speed: 1.371`
  - 单位是 m/s（米每秒）
  - 0.174 m/s ≈ 0.6 km/h（很慢，可能是起步阶段）
  - 1.371 m/s ≈ 4.9 km/h（低速行驶）

### 总结表

| 文件类型 | 坐标系 | 单位 | 示例 |
|---------|--------|------|------|
| **XML** | CARLA 世界坐标系 | **米（m）** | `x="-497.6"` = -497.6 米 |
| **JSON - pos_global** | CARLA 世界坐标系 | **米（m）** | `[-1130.51, 3738.84]` |
| **JSON - target_point** | 车辆坐标系 | **米（m）** | `[51.54, 0.004]` = 前方 51.54m，左侧 0.004m |
| **JSON - route** | 车辆坐标系 | **米（m）** | `[7.49, 0.0006]` = 前方 7.49m，左侧 0.0006m |
| **JSON - speed** | - | **米/秒（m/s）** | `0.174` = 0.174 m/s |

## 示例：target_point 的坐标转换

```python
# 1. 全局坐标系中的目标点（单位：米）
target_point = [x_global, y_global, z_global]  # CARLA 世界坐标（米）

# 2. 转换为车辆坐标系（单位：米）
ego_target_point = inverse_conversion_2d(
    target_point[:2],      # [x, y] 单位：米
    current_gps,           # 车辆当前位置 [x, y] 单位：米
    current_compass        # 车辆当前朝向（弧度）
)

# 3. ego_target_point 是相对于车辆的位置（单位：米）
#    [x_relative, y_relative]
#    x_relative: 车辆前方为正（米）
#    y_relative: 车辆左侧为正（米）
#    例如：[51.54, 0.004] 表示前方 51.54 米，左侧 0.004 米
```

## Route 坐标的含义

### 问题：route 里面的值指的是离当前位置（pos_global）往前 7.49 米吗？

**答案：是的！** `route` 中的坐标是相对于车辆当前位置（`pos_global`）的距离和方向。

### 详细说明

```json
{
    "pos_global": [-1130.51, 3738.84],    // 车辆在世界坐标系中的位置（米）
    "theta": 1.571,                       // 车辆朝向角度（弧度）
    "route": [
        [7.49, 0.0006],                   // 相对于 pos_global，前方 7.49 米，左侧 0.0006 米
        [8.49, 0.0007],                   // 相对于 pos_global，前方 8.49 米，左侧 0.0007 米
        ...
    ]
}
```

### 坐标转换过程

```python
# 在 autopilot.py 的 save() 方法中
# 1. 原始路径点在世界坐标系中
checkpoint = [x_world, y_world]  # 世界坐标系中的路径点

# 2. 转换为车辆坐标系
ego_position = tick_data["gps"][:2]      # 车辆当前位置 [x, y]（对应 pos_global）
ego_orientation = tick_data["compass"]   # 车辆朝向（对应 theta）

# 3. 使用 inverse_conversion_2d 转换
dense_route.append(
    t_u.inverse_conversion_2d(
        checkpoint[:2],           # 世界坐标系中的路径点
        ego_position[:2],         # 车辆当前位置（pos_global）
        ego_orientation           # 车辆朝向（theta）
    )
)
# 结果：[x_relative, y_relative] 相对于车辆的位置
```

### 车辆坐标系定义

- **原点**：车辆当前位置（`pos_global`）
- **x 轴**：指向车辆前方（沿车辆朝向方向）
- **y 轴**：指向车辆左侧（垂直于 x 轴，左侧为正）
- **单位**：米（m）

### 示例解读

对于 `route: [7.49, 0.0006]`：

1. **原始位置**：路径点在世界坐标系中的某个位置
2. **转换后**：相对于车辆当前位置（`pos_global: [-1130.51, 3738.84]`）
   - **前方 7.49 米**：沿车辆朝向方向（`theta: 1.571` 弧度）前进 7.49 米
   - **左侧 0.0006 米**：垂直于车辆朝向，向左偏移 0.0006 米（约 0.6 毫米）

### 可视化理解

```
车辆当前位置 (pos_global)
    ↓ (车辆朝向，theta)
    |
    | 前方 7.49 米
    |
    ● ← 路径点 [7.49, 0.0006]
    |
    | 左侧 0.0006 米
```

### 关键点总结

1. ✅ `route` 中的坐标是**相对于车辆当前位置（`pos_global`）**的
2. ✅ 第一个值（7.49）表示**前方距离**（沿车辆朝向方向）
3. ✅ 第二个值（0.0006）表示**左侧距离**（垂直于车辆朝向，左侧为正）
4. ✅ 单位都是**米（m）**
5. ✅ 路径点间距约 1 米（8.49 - 7.49 = 1.0 米）

## 模型预测的坐标系统

### 问题：模型预测的也是相对位置吗？

**答案：是的！** 模型预测的 route waypoints 和 speed waypoints 都是**相对于车辆当前位置的相对位置**。

### 模型输出格式

```python
# 模型预测输出
pred_route = [[-0.001, 0.001], [1.0, -0.034], [2.0, -0.059], ...]  # [1, 20, 2]
pred_speed_wps = [[2.27, -0.12], [4.47, -0.17], [6.63, -0.25], ...]  # [1, 10, 2]
```

### 预测值的含义

- **pred_route**: 预测的 20 个 route waypoints
  - 第一个点接近 `[0, 0]`：表示车辆当前位置（原点）
  - 后续点：`[1.0, ...]`, `[2.0, ...]` 表示前方 1 米、2 米的位置
  - 坐标系：**车辆坐标系**（相对于车辆当前位置）

- **pred_speed_wps**: 预测的 10 个 speed waypoints
  - 表示车辆在未来时刻的位置
  - 坐标系：**车辆坐标系**（相对于车辆当前位置）

### 在控制中的使用

```python
# 在 agent_simlingo.py 中
pred_speed_wps, pred_route, language = self.model(model_input)

# 直接使用预测的相对坐标计算控制命令
steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)

# 在 control_pid() 中
route_interp = self.interpolate_waypoints(route_waypoints.squeeze())
steer = self.turn_controller.step(route_interp, speed)
# 直接使用相对坐标，无需转换
```



### 完整流程

```
1. XML 路由文件（bench2drive_2403.xml）
   ↓
   <waypoints>
     <position x="-3418.8" y="1174.6" z="345.3" />
     <position x="-3419.3" y="1176.6" z="345.3" />
     ... (稀疏的关键点，通常 20-100 个)
   </waypoints>
   
2. RouteParser.parse_routes_file()
   ↓
   解析 XML，提取 keypoints (carla.Location 列表)
   
3. interpolate_trajectory(waypoints_trajectory, hop_resolution=1.0)
   ↓
   使用 GlobalRoutePlanner 在每两个关键点之间插值
   - hop_resolution=1.0 表示插值点间距 1 米
   - 生成密集的路径点序列（可能有数百到数千个点）
   
4. RoutePlanner.set_route(global_plan, gps=True)
   ↓
   设置全局路径，转换为 GPS 坐标
   
5. 每个推理 step:
   RoutePlanner.run_step(current_gps)
   ↓
   根据当前 GPS 位置，动态选择目标点
   - 移除已通过的路径点
   - 返回当前应追踪的路径点序列
   
6. target_point = waypoint_route[1]
   ↓
   选择下一个目标点，转换为车辆坐标系
```

### XML 文件结构示例

```xml
<?xml version='1.0' encoding='utf-8'?>
<routes>
   <route id="2403" town="Town12">
      <waypoints>
         <position x="-3418.8" y="1174.6" z="345.3" />
         <position x="-3419.3" y="1176.6" z="345.3" />
         <!-- 稀疏的关键点，通常 20-100 个 -->
      </waypoints>
      <scenarios>...</scenarios>
      <weathers>...</weathers>
   </route>
</routes>
```

## Ground Truth 数据来源

### 问题：模型输出 route waypoints 和 speed waypoints，训练数据中哪些字段用作 ground truth？

### 1. Speed Waypoints (速度路径点) 的 Ground Truth

**来源**：`ego_matrix` 字段

**提取过程**：
```python
# 在 dataset_base.py 的 get_waypoints() 方法中
def get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
    """从 measurements 中提取未来时刻的车辆位置作为 waypoints"""
    origin = measurements[0]  # 当前时刻
    origin_matrix = np.array(origin['ego_matrix'])[:3]  # 4x4 变换矩阵的前3行
    origin_translation = origin_matrix[:, 3:4]  # 平移部分
    origin_rotation = origin_matrix[:, :3]     # 旋转部分
    
    waypoints = []
    for index in range(len(measurements)):  # 遍历未来时刻
        waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
        # 转换为以当前车辆为原点的坐标系
        waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
        waypoints.append(waypoint_ego_frame[:2, 0])  # 只取 x, y (BEV)
    
    return waypoints
```

**说明**：
- `ego_matrix` 是一个 4x4 齐次变换矩阵，表示车辆在世界坐标系中的位姿
- 每个 measurement 文件（对应一个时间步）都有一个 `ego_matrix`
- 通过提取未来时刻的 `ego_matrix` 的平移部分，得到车辆在未来时刻的位置
- 这些位置转换为以当前车辆为原点的坐标系后，就是 **speed waypoints 的 ground truth**

**数据格式**：
- 输入：`loaded_measurements[self.hist_len - 1:]`（当前时刻及未来时刻的 measurements）
- 输出：`waypoints[1:-1]`（去除第一个和最后一个点后的路径点序列）
- 形状：`[N, 2]`，其中 N 是未来路径点的数量，2 是 (x, y) 坐标

### 2. Route Waypoints (路线路径点) 的 Ground Truth

**来源**：`route_original` 或 `route` 字段

**提取过程**：
```python
# 在 dataset_base.py 的 load_route() 方法中
def load_route(self, data, current_measurement, aug_translation=0.0, aug_rotation=0.0):
    # 使用 route_original（原始路径点）
    route = current_measurement['route_original']
    route = self.augment_route(route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
    
    # 或使用 route（调整后的路径点）
    route_adjusted = np.array(current_measurement['route'])
    route_adjusted = self.augment_route(route_adjusted, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
    
    # 等间距采样
    route = self.equal_spacing_route(route)
    route_adjusted = self.equal_spacing_route(route_adjusted)
    
    data['route'] = route                    # 原始路径点（等间距采样后）
    data['route_adjusted'] = route_adjusted  # 调整后的路径点（等间距采样后）
```

**说明**：
- `route_original`：原始路径点序列（来自 `PrivilegedRoutePlanner` 的原始路径）
- `route`：调整后的路径点序列（可能根据障碍物等情况调整过）
- 根据模型配置，可能使用 `route_original` 或 `route_adjusted` 作为 ground truth
- 路径点会进行等间距采样（`equal_spacing_route()`），确保路径点间距一致

**数据格式**：
- 输入：`current_measurement['route_original']` 或 `current_measurement['route']`
- 输出：`route` 或 `route_adjusted`（等间距采样后）
- 形状：`[M, 2]`，其中 M 是路径点数量（通常由 `num_route_points` 配置决定），2 是 (x, y) 坐标

### 3. 在模型训练中的使用

**在 `DrivingAdaptor.compute_loss()` 中**：

```python
# 对于 route waypoints
if self.predict_route_as_wps:
    label_route = label.path  # 或 label.route_adjusted（取决于配置）
else:
    label_route = None

# 对于 speed waypoints
if self.speed_wps_mode == '2d':
    label_speed_wps = label.waypoints[:, : self.future_waypoints + 1]
elif self.speed_wps_mode == '1d':
    label_speed_wps = label.waypoints_1d
else:
    label_speed_wps = None

# 计算损失
prediction = self.heads[input_type](features_tmp).cumsum(1)
loss = F.smooth_l1_loss(prediction, label, reduction="none").sum(-1)
```

### 总结表

| 模型输出 | Ground Truth 来源 | JSON 字段 | 提取方法 | 数据形状 |
|---------|-----------------|-----------|---------|---------|
| **Speed Waypoints** | 未来时刻的车辆位置 | `ego_matrix` | `get_waypoints()` | `[N, 2]` |
| **Route Waypoints** | 路径点序列 | `route` | `load_route()` | `[20, 2]` |

**关键点**：
1. **Speed waypoints** 来自未来时刻的 `ego_matrix`，表示车辆实际行驶轨迹
2. **Route waypoints** 来自 `route` 字段（不是 `route_original`），表示调整后的规划路径
3. 两者都转换为以当前车辆为原点的坐标系（车辆坐标系）
4. 路径点会进行等间距采样，确保训练数据的一致性

## Route Waypoints 预测详解

### 问题：模型输出 `pred_route shape=torch.Size([1, 20, 2])`，说明需要预测 20 个 future route waypoints 吗？

**答案：是的！** 模型需要预测 **20 个未来的 route waypoints**。

### 模型配置

```python
# 在 DrivingAdaptor 中
if predict_route_as_wps:
    self.future_waypoints = 20  # 预测 20 个 route waypoints
    self.query_embeds_wps = nn.Parameter(0.02 * torch.randn((1, 20, hidden_size)))
    self.route_head = nn.Sequential(...)  # 输出 [B, 20, 2]
```

### Ground Truth 来源

**Ground Truth 来自 `route` 字段（不是 `route_original`）**

```python
# 在 dataset_driving.py 中
path = data['route_adjusted']  # 来自 JSON 的 'route' 字段

# 在 load_route() 中
route_adjusted = np.array(current_measurement['route'])  # 从 JSON 读取
route_adjusted = self.equal_spacing_route(route_adjusted)  # 等间距采样
data['route_adjusted'] = route_adjusted  # 保存为 path
```

### 数据处理流程

```
1. JSON 文件中的 'route' 字段
   ↓
   [多个路径点，可能数量不固定]
   
2. load_route() 处理
   ↓
   route_adjusted = np.array(current_measurement['route'])
   route_adjusted = equal_spacing_route(route_adjusted)
   # 等间距采样，确保路径点间距一致
   
3. 截断或填充到 20 个点
   ↓
   if len(route_adjusted) < 20:
       # 填充最后一个点
   else:
       route_adjusted = route_adjusted[:20]
   
4. 作为 Ground Truth
   ↓
   label.path = route_adjusted  # [20, 2]
```

### Loss 计算

```python
# 在 DrivingAdaptor.compute_loss() 中
if self.predict_route_as_wps:
    label_route = label.path  # [B, 20, 2] 来自 route_adjusted

# 预测值
prediction = self.heads['route'](features_tmp).cumsum(1)  # [B, 20, 2]

# 计算损失
loss = F.smooth_l1_loss(prediction, label_route, reduction="none").sum(-1)
```

### 关键点总结

1. **预测数量**：模型预测 **20 个 route waypoints**（`[B, 20, 2]`）
2. **Ground Truth 来源**：来自 JSON 文件中的 **`route` 字段**（不是 `route_original`）
3. **数据处理**：
   - `route` 字段经过 `equal_spacing_route()` 等间距采样
   - 如果路径点数量不足 20，会填充最后一个点
   - 如果超过 20，会截断到前 20 个点
4. **Loss 计算**：使用 `smooth_l1_loss` 比较预测值和 `label.path`（即 `route_adjusted`）

### 为什么使用 `route` 而不是 `route_original`？

- `route_original`：原始路径点，来自 `PrivilegedRoutePlanner` 的原始规划
- `route`：调整后的路径点，可能根据障碍物、交通情况等进行了调整
- 使用 `route` 可以让模型学习到更实际的路径规划（考虑了动态调整）

### RoadOption 命令值
```python
# 在 agents/navigation/local_planner.py 中定义
class RoadOption(IntEnum):
    VOID = -1
    LEFT = 1      # 左转
    RIGHT = 2     # 右转
    STRAIGHT = 3  # 直行
    LANEFOLLOW = 4  # 跟随车道
    CHANGELANELEFT = 5   # 向左变道
    CHANGELANERIGHT = 6  # 向右变道
```

