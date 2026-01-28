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

**关键因素**：
1. **速度限制**：从地图获取的道路速度限制
2. **路口减速**：接近路口时降低速度
3. **障碍物处理**：根据前方车辆、交通灯、停止标志调整速度
4. **IDM 模型**：使用智能驾驶员模型计算安全速度

### 在 agent_simlingo.py（推理）中

**注意**：当前 `agent_simlingo.py` 中**不直接计算 `target_speed`**，而是：
- 模型预测控制动作（steering, throttle, brake）
- 或者模型预测 waypoints，然后使用 `LateralPIDController` 计算 steering

## 训练 vs 推理模式的区别

### 1. 路径点密度

| 模式 | 路径点间距 | 说明 |
|------|-----------|------|
| **训练模式** | 10cm | 专家策略使用密集采样，用于生成高质量训练数据 |
| **推理模式** | 1m | 模型预测的 checkpoint 间距，需要适配 |

### 2. Lookahead 距离计算

**训练模式**（`inference_mode=False`）：
```python
n_lookahead = int(min(
    np.clip(speed_scale * current_speed + speed_offset, 24, 105),
    route_np.shape[0] - 1
))
# 单位：米（m），范围 [24, 105] 米
# 路径点间距 10cm，所以索引范围 [240, 1050]
```

**推理模式**（`inference_mode=True`）：
```python
n_lookahead = np.clip(speed_scale * current_speed + speed_offset, 24, 105) / 10
n_lookahead = n_lookahead - 2
n_lookahead = int(min(n_lookahead, route_np.shape[0] - 1))
# 单位：索引（index），对应 1m 间距的路径点
# 除以 10 是因为路径点间距从 10cm 变为 1m
# 减 2 是为了调整索引偏移
```

### 3. target_point 选择

**相同点**：
- 都使用 `RoutePlanner.run_step()` 获取当前路径点序列
- 都选择 `waypoint_route[1]` 作为 `target_point`
- 都转换为车辆坐标系

**不同点**：
- **训练模式**：路径点更密集，`target_point` 更精确
- **推理模式**：路径点较稀疏，`target_point` 可能跳跃较大

### 4. target_speed 计算

**训练模式**（autopilot.py）：
- 完整的专家策略
- 考虑速度限制、路口、障碍物、IDM 模型
- 生成高质量的训练数据

**推理模式**（agent_simlingo.py）：
- 模型直接预测控制动作
- 或者模型预测 waypoints，然后使用控制器
- **不直接计算 `target_speed`**（如果需要，可以从预测的 waypoints 推导）

## 代码位置总结

### 训练数据收集
- **路径规划**：`team_code/autopilot.py` → `PrivilegedRoutePlanner`
- **target_speed 计算**：`team_code/autopilot.py` → `_get_control()` 方法
- **target_point 计算**：`team_code/autopilot.py` → `_command_planner.run_step()`
- **控制器**：`team_code/nav_planner.py` → `LateralPIDController(inference_mode=False)`

### 推理
- **路径规划**：`team_code/agent_simlingo.py` → `RoutePlanner`
- **target_point 计算**：`team_code/agent_simlingo.py` → `tick()` 方法（第 469-497 行）
- **控制器**：`team_code/nav_planner.py` → `LateralPIDController(inference_mode=False)`（当前设置）


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

### 为什么使用相对坐标？

1. **训练一致性**：Ground truth 是相对坐标，模型预测也应该是相对坐标
2. **控制便利性**：PID 控制器直接使用相对坐标计算转向角度，无需坐标转换
3. **数值稳定性**：相对坐标数值较小，有利于模型训练和数值稳定性

### 训练 vs 推理的一致性

| 阶段 | 输入 | 输出 | 坐标系 |
|------|------|------|--------|
| **训练** | 图像 + 相对 target_point | Ground truth: 相对 route/waypoints | 车辆坐标系 |
| **推理** | 图像 + 相对 target_point | 预测: 相对 route/waypoints | 车辆坐标系 |

**关键**：训练和推理使用相同的坐标系（车辆坐标系），确保模型学习的是相对位置预测。

## 注意事项

1. **当前代码设置**：`agent_simlingo.py` 中 `LateralPIDController` 的 `inference_mode=False`，这意味着即使是在推理时，也使用训练模式的 lookahead 计算方式。

2. **路径点更新**：`RoutePlanner.run_step()` 会根据车辆当前位置动态更新路径点，移除已经经过的点。

3. **坐标系统**：
   - 全局坐标：CARLA 世界坐标系
   - 车辆坐标：以车辆为中心，x 轴指向前方，y 轴指向左侧

4. **命令（RoadOption）**：
   - `target_point` 伴随一个 `RoadOption` 命令（如 LANEFOLLOW, LEFT, RIGHT 等）
   - 这个命令用于生成语言提示（prompt）

## XML 路由文件对 target_point 的影响

### 问题：XML 路由文件会影响推理时的 target_point 计算吗？

**答案：会！** XML 文件中的 waypoints 是计算 target_point 的基础。

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

### 关键点说明

1. **稀疏关键点 → 密集路径点**：
   - XML 中的 waypoints 是**稀疏的关键点**（keypoints）
   - 通常只有 20-100 个点，定义路径的主要转折点
   - 通过 `interpolate_trajectory()` 插值成**密集路径点**（间距 1m）
   - 最终可能有数百到数千个路径点

2. **插值过程**：
   ```python
   # 在 leaderboard/utils/route_manipulation.py 中
   def interpolate_trajectory(waypoints_trajectory, hop_resolution=1.0):
       grp = GlobalRoutePlanner(CarlaDataProvider.get_map(), hop_resolution)
       for i in range(len(waypoints_trajectory) - 1):
           waypoint = waypoints_trajectory[i]
           waypoint_next = waypoints_trajectory[i + 1]
           interpolated_trace = grp.trace_route(waypoint, waypoint_next)
           # 在两点之间生成密集的路径点
   ```

3. **target_point 的选择**：
   - `RoutePlanner.run_step()` 根据当前 GPS 位置动态更新路径点
   - 选择 `waypoint_route[1]` 作为 `target_point`
   - 如果 XML 中的关键点位置改变，插值后的路径点也会改变
   - 因此 `target_point` 也会改变

### 影响程度

| 因素 | 影响程度 | 说明 |
|------|---------|------|
| **关键点位置** | ⭐⭐⭐⭐⭐ | 直接影响插值后的路径，从而影响 target_point |
| **关键点数量** | ⭐⭐⭐ | 影响路径的平滑度，但插值会填充中间点 |
| **关键点顺序** | ⭐⭐⭐⭐⭐ | 顺序错误会导致完全错误的路径 |
| **Town/地图** | ⭐⭐⭐⭐⭐ | 不同地图的路径规划结果不同 |

### 实际影响示例

**场景 1：关键点位置偏移**
```xml
<!-- 原始 -->
<position x="-3418.8" y="1174.6" z="345.3" />

<!-- 偏移后 -->
<position x="-3420.0" y="1175.0" z="345.3" />
```
**影响**：插值后的路径会整体偏移，`target_point` 也会相应偏移。

**场景 2：关键点缺失**
```xml
<!-- 如果某个关键点被删除 -->
<!-- 插值会在剩余点之间直接连接，可能穿过建筑物 -->
```
**影响**：路径可能不合理，`target_point` 可能指向错误方向。

**场景 3：关键点顺序错误**
```xml
<!-- 如果关键点顺序颠倒 -->
<!-- 车辆会沿着相反方向行驶 -->
```
**影响**：完全错误的路径，`target_point` 完全错误。

### 代码位置

- **XML 解析**：`Bench2Drive/leaderboard/leaderboard/utils/route_parser.py`
- **路径插值**：`Bench2Drive/leaderboard/leaderboard/utils/route_manipulation.py` → `interpolate_trajectory()`
- **路径设置**：`team_code/agent_simlingo.py` → `setup()` → `self._route_planner.set_route()`
- **target_point 计算**：`team_code/agent_simlingo.py` → `tick()` → `self._route_planner.run_step()`

### 总结

**XML 路由文件直接影响推理时的 target_point 计算**，因为：

1. ✅ XML 中的 waypoints 定义了全局路径的关键点
2. ✅ 这些关键点被插值成密集的路径点序列
3. ✅ `RoutePlanner` 使用这些路径点计算 `target_point`
4. ✅ 如果 XML 文件改变，`target_point` 也会改变

因此，**确保 XML 路由文件的正确性非常重要**，它直接决定了车辆应该追踪的路径。

## 训练数据收集时的 route、target_point 和 command 生成

### 问题：训练数据收集时没有 XML 路由文件，route、target_point 和 command 是怎么生成的？

**答案**：训练数据收集时，路径是从外部传入的（可能来自 XML 或其他来源），然后通过 `PrivilegedRoutePlanner` 和 `RoutePlanner` 生成 route、target_point 和 command。

### 完整流程

```
1. 外部路径来源（可能是 XML、scenario runner 或其他）
   ↓
2. AutonomousAgent.set_global_plan(global_plan_gps, global_plan_world_coord)
   ↓
   下采样路径点（downsample_route，每 200 米一个点）
   ↓
   self._global_plan (GPS 坐标，稀疏)
   self._global_plan_world_coord (世界坐标，稀疏)
   self.org_dense_route_world_coord (密集路径点，来自外部)
   
3. PrivilegedRoutePlanner.setup_route()
   ↓
   使用密集路径点生成平滑、插值的路径
   - 平滑路径点
   - 超采样（supersample）
   - 计算交通灯、停止标志距离
   - 生成 self.route_points (密集路径点)
   
4. RoutePlanner.set_route() (作为 _command_planner)
   ↓
   使用稀疏路径点设置命令规划器
   - 用于生成 target_point 和 command
   
5. 每个 step:
   a) PrivilegedRoutePlanner.run_step() 
      → 生成 route_np (密集路径点，用于控制)
   
   b) RoutePlanner.run_step() (作为 _command_planner)
      → 生成 command_route
      → target_point, far_command = command_route[1]
      → next_target_point, next_far_command = command_route[2]
   
6. save() 方法保存数据
   ↓
   转换为车辆坐标系并保存到 JSON 文件
   - target_point: 车辆坐标系中的目标点
   - target_point_next: 下一个目标点
   - command: RoadOption 命令（1-6）
   - route: 密集路径点序列（车辆坐标系）
```

### 关键代码位置

#### 1. 路径设置（`autopilot.py`）

```python
def _init(self, hd_map):
    # 使用密集路径点设置路径规划器
    self._waypoint_planner = PrivilegedRoutePlanner(self.config)
    self._waypoint_planner.setup_route(
        self.org_dense_route_world_coord,  # 密集路径点
        self._world, 
        self.world_map,
        starts_with_parking_exit, 
        self._vehicle.get_location()
    )
    
    # 使用稀疏路径点设置命令规划器
    self._command_planner = RoutePlanner(...)
    self._command_planner.set_route(self._global_plan_world_coord)  # 稀疏路径点
```

#### 2. target_point 和 command 生成（`autopilot.py`）

```python
def _get_control(self, input_data, plant):
    # 从命令规划器获取目标点和命令
    command_route = self._command_planner.run_step(ego_position)
    
    if len(command_route) > 2:
        target_point, far_command = command_route[1]
        next_target_point, next_far_command = command_route[2]
    elif len(command_route) > 1:
        target_point, far_command = command_route[1]
        next_target_point, next_far_command = command_route[1]
    else:
        target_point, far_command = command_route[0]
        next_target_point, next_far_command = command_route[0]
    
    # 保存数据
    driving_data = self.save(
        target_point, 
        next_target_point, 
        steer, throttle, brake, 
        control_brake, 
        target_speed,
        speed_limit, 
        tick_data, 
        speed_reduced_by_obj
    )
```

#### 3. 数据保存（`autopilot.py`）

```python
def save(self, target_point, next_target_point, ...):
    # 转换为车辆坐标系
    ego_target_point = t_u.inverse_conversion_2d(
        target_point[:2], 
        ego_position, 
        ego_orientation
    ).tolist()
    
    ego_next_target_point = t_u.inverse_conversion_2d(
        next_target_point[:2], 
        ego_position, 
        ego_orientation
    ).tolist()
    
    # 获取剩余路径点（车辆坐标系）
    dense_route = []
    for checkpoint in remaining_route:
        dense_route.append(
            t_u.inverse_conversion_2d(
                checkpoint[:2], 
                ego_position[:2], 
                ego_orientation
            ).tolist()
        )
    
    # 保存到 JSON 文件
    data = {
        "target_point": ego_target_point,        # [x, y] 车辆坐标系
        "target_point_next": ego_next_target_point,
        "command": self.commands[-2],            # RoadOption 值 (1-6)
        "next_command": self.next_commands[-2],
        "route": dense_route,                    # 路径点序列（车辆坐标系）
        "route_original": dense_route_original,
        "target_speed": target_speed,
        "speed_limit": speed_limit,
        # ... 其他数据
    }
```

### 训练数据 JSON 文件结构

```json
{
    "pos_global": [-1130.51, 3738.84],           // 全局位置
    "theta": 1.571,                              // 朝向角度
    "speed": 0.174,                              // 当前速度 (m/s)
    "target_speed": 1.371,                       // 目标速度 (m/s)
    "speed_limit": 13.889,                      // 速度限制 (m/s)
    "target_point": [51.54, 0.004],             // 目标点（车辆坐标系）
    "target_point_next": [67.13, 0.005],        // 下一个目标点
    "command": 4,                                // RoadOption: 4 = LANEFOLLOW
    "next_command": 4,                           // 下一个命令
    "aim_wp": [7.39, 0.0006],                   // 转向目标点
    "route": [                                   // 路径点序列（车辆坐标系，调整后的）
        [7.49, 0.0006],
        [8.49, 0.0007],
        [9.49, 0.0007],
        ...
    ],
    "route_original": [...],                     // 原始路径点（车辆坐标系）
    "ego_matrix": [[...], [...], [...], [...]], // 4x4 变换矩阵（车辆位姿）
    "steer": 0.0,                                // 转向角度
    "throttle": 0.0,                             // 油门
    "brake": false,                              // 刹车
    ...
}
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

### 关键区别：训练 vs 推理

| 方面 | 训练数据收集 | 推理 |
|------|------------|------|
| **路径来源** | 外部传入（`set_global_plan()`） | XML 路由文件 |
| **路径规划器** | `PrivilegedRoutePlanner` + `RoutePlanner` | `RoutePlanner` |
| **路径密度** | 密集（10cm 间距）+ 稀疏（命令） | 稀疏（1m 间距，插值后） |
| **target_point** | 从 `_command_planner.run_step()` 获取 | 从 `_route_planner.run_step()` 获取 |
| **command** | 从 `_command_planner` 获取 `RoadOption` | 从 `_route_planner` 获取 `RoadOption` |
| **route** | 从 `PrivilegedRoutePlanner` 获取密集路径点 | 从 `RoutePlanner` 获取路径点 |

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

### 总结

1. **训练数据收集时**：
   - 路径从外部传入（通过 `set_global_plan()`）
   - 使用 `PrivilegedRoutePlanner` 生成密集、平滑的路径点
   - 使用 `RoutePlanner` (作为 `_command_planner`) 生成 `target_point` 和 `command`
   - 所有数据转换为车辆坐标系后保存到 JSON 文件

2. **推理时**：
   - 路径从 XML 文件解析
   - 使用 `RoutePlanner` 生成路径点和 `target_point`
   - 路径点间距较稀疏（1m），通过插值生成

3. **关键点**：
   - 训练数据中的 `route`、`target_point` 和 `command` 都是**专家策略（expert policy）**生成的
   - 这些数据用于监督学习，模型学习模仿专家的行为
   - 推理时，模型需要根据当前状态预测这些值

## 参考

- `team_code/nav_planner.py`: 核心路径规划和控制代码
- `team_code/agent_simlingo.py`: 推理时的使用
- `team_code/autopilot.py`: 训练数据收集时的使用
- `team_code/privileged_route_planner.py`: 特权路径规划器（训练时使用）
- `Bench2Drive/leaderboard/leaderboard/utils/route_parser.py`: XML 路由文件解析
- `Bench2Drive/leaderboard/leaderboard/utils/route_manipulation.py`: 路径插值
- `leaderboard/leaderboard/autoagents/autonomous_agent.py`: 基类，包含 `set_global_plan()` 方法

