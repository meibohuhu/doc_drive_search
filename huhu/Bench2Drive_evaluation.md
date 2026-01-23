# Bench2Drive 评估流程详解

## 1. 评估代码是否根据 XML 进行评估？

**是的！** `run_eval_local.sh` 中的评估代码会根据 `bench2drive_mini_10.xml` 进行评估：

```bash
ROUTES=${WORK_DIR}/leaderboard/data/bench2drive_mini_10.xml
```

`leaderboard_evaluator.py` 会解析该 XML 文件，对每个 route 进行评估。

## 2. XML 文件包含的数据

`bench2drive_mini_10.xml` 包含 6 个 route，每个 route 包含：

### a) Waypoints（路径点）
- 一系列 3D 坐标点 `(x, y, z)`
- 定义车辆应该行驶的**粗略路径规划**
- 例如第一个 route 有 73 个 waypoints，从 `(1362.2, 5317.5, 370.5)` 到 `(1362.6, 5451.5, 370.1)`
- **注意：** 这些是粗略的关键点，系统会在它们之间插值生成密集的路径点

### b) Scenarios（场景）
- 定义路径上会遇到的挑战场景
- 例如：
  - `AccidentTwoWays_1`：双向事故
  - `Accident_1`：事故场景
  - `ControlLoss_1`：控制丢失
  - `DynamicObjectCrossing_1`：动态物体穿越
  - `HardBreakRoute_1`：急刹车
  - `OppositeVehicleTakingPriority_1`：对向车辆优先
- 每个 scenario 包含触发点（trigger_point）和参数（如距离、方向、速度等）

### c) Weathers（天气条件）
- 定义天气参数（云量、雾、降水、风速等）
- 通常包含起始和结束天气条件

## 3. XML Waypoints 的作用是什么？

### XML Waypoints 的三个主要作用

虽然 Model 不会直接接收 XML 的所有 waypoints，但 XML waypoints 在整个系统中起到关键作用：

#### 作用 1：全局路径规划（导航地图）
- XML waypoints 定义了**整个 route 的路径**
- 告诉系统"应该走哪条路"
- 类似于 GPS 导航中的路线规划

#### 作用 2：生成密集路径点和命令
```python
# route_scenario.py 第121行
self.gps_route, self.route = interpolate_trajectory(config.keypoints)
```

**`interpolate_trajectory` 做了什么：**
1. 接收 XML 的粗略 waypoints（73个点）
2. 使用 CARLA 的 `GlobalRoutePlanner` 在 waypoints 之间插值
3. 生成**密集的路径点序列**（可能有数百个点）
4. 同时为每个路径点生成 **RoadOption（命令）**：
   - `LANEFOLLOW` (4) - 跟随道路
   - `LEFT` (1) - 左转
   - `RIGHT` (2) - 右转
   - `STRAIGHT` (3) - 直行
   - `CHANGELANELEFT` (5) - 左变道
   - `CHANGELANERIGHT` (6) - 右变道

**命令是如何生成的？**
- CARLA 的 GlobalRoutePlanner 根据路径的几何形状和道路网络自动推断
- 例如：如果路径在路口向左转，对应的 waypoint 会被标记为 `LEFT`

#### 作用 3：为 RoutePlanner 提供数据源
- RoutePlanner 使用插值后的密集路径点和命令
- 根据车辆当前位置，从这些路径点中选择下一个目标点
- 同时返回对应的命令（用于生成 prompt）

## 4. 模型如何根据 XML 数据进行预测？

### 重要理解：XML Waypoints 不会直接传给 Model

**关键点：**
- XML 中的 waypoints 是**全局路径规划**（告诉你要去哪里）
- Model **不会**直接接收 XML 的所有 waypoints
- 而是从 waypoints 中**提取下一个目标点（target_point）**传给 model
- Model 根据图像和目标点，**预测如何到达目标点的路径点**

### 详细流程

#### 步骤 1：XML Waypoints → 密集路径点和命令
```python
# route_scenario.py 第121行
self.gps_route, self.route = interpolate_trajectory(config.keypoints)
```
- XML 的粗略 waypoints（73个点）被插值成密集路径点（数百个点）
- 每个路径点都有对应的 RoadOption 命令
- 结果存储在 `self.route` 中：`[(transform, RoadOption), ...]`

#### 步骤 2：Route 信息传递给 Agent
```python
# leaderboard_evaluator.py 第387行
self.agent_instance.set_global_plan(self.route_scenario.gps_route, self.route_scenario.route)
```
- 插值后的密集路径点和命令被传递给 agent
- 存储在 `self._global_plan` 中

#### 步骤 3：RoutePlanner 初始化并存储路径点和命令
```python
# agent_simlingo.py 第299-301行
self._route_planner = RoutePlanner(...)
self._route_planner.set_route(self._global_plan, True)
```
- RoutePlanner 将密集路径点和命令存储为全局路径
- 这些数据保存在 `self._route_planner.route` 中：`[(position, command), ...]`

#### 步骤 4：每个 Step 从路径点中选择目标点和命令
```python
# agent_simlingo.py 第448行
waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))
target_point, far_command = waypoint_route[1]  # 选择下一个目标点和命令
```

**RoutePlanner.run_step() 做了什么：**
1. 根据车辆当前位置（GPS）
2. 从存储的密集路径点中找到距离合适的目标点（通常在 7.5-50 米范围内）
3. 返回包含目标点和**命令（RoadOption）**的列表
4. **命令（far_command）** 是从 XML waypoints 插值后自动生成的，例如：
   - `RoadOption.LEFT` (1) → "go left at the next intersection"
   - `RoadOption.LANEFOLLOW` (4) → "follow the road"

**关键转换：**
```python
# agent_simlingo.py 第472行
ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
```
- 将世界坐标系的目标点转换为**车辆坐标系**（ego坐标系）
- 这样 model 就知道"目标点在我前方哪个方向"

#### 步骤 5：生成 Prompt
根据配置（`eval_route_as`），模型会收到不同的 prompt：

- 如果 `eval_route_as == 'target_point'`：
  ```python
  # agent_simlingo.py 第495行
  prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
  # <TARGET_POINT> 会被替换为实际的坐标值
  ```

- 如果 `eval_route_as == 'command'`：
  ```python
  # agent_simlingo.py 第524-538行
  # 根据far_command生成命令，如：
  # "Command: follow the road in 44 meter."
  # "Command: go left at the next intersection in 30 meter."
  ```

#### 步骤 6：构建 Model 输入
```python
# agent_simlingo.py 第666-673行
self.DrivingInput["camera_images"] = processed_image.to(self.device).bfloat16()
self.DrivingInput["vehicle_speed"] = result['speed']
self.DrivingInput["target_point"] = result['target_point'].to(self.device)  # 从waypoints提取的目标点
self.DrivingInput["prompt"] = ll  # 包含目标点或命令的文本prompt
```

**Model 的完整输入：**
1. **相机图像（RGB）**：当前看到的场景
2. **当前速度**：车辆当前速度
3. **目标点（target_point）**：从 XML waypoints 中提取的下一个目标点（车辆坐标系）
4. **Prompt（文本）**：包含目标点坐标或命令的文本描述
5. **相机内外参**：用于图像处理

**注意：** Model **不接收** XML 的所有 waypoints，只接收**下一个目标点**！

#### 步骤 7：Model 预测
```python
# agent_simlingo.py 第693行
pred_speed_wps, pred_route, language = self.model(model_input)
```

**Model 预测的是什么？**
- **`pred_route`**：预测的**未来路径点序列**（waypoints）
  - 这些路径点用于**到达目标点**
  - 不是 XML 的 waypoints，而是 model 根据图像和目标点预测的路径
- **`pred_speed_wps`**：速度相关的路径点
- **`language`**：语言输出（如果启用）

**Model 的任务：**
- 输入：图像 + 目标点（"我要去那里"）
- 输出：预测如何到达目标点的路径点序列（"这样走"）

#### 步骤 8：控制信号生成
```python
# agent_simlingo.py 第811-840行
steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)
```

使用 PID 控制器将 model 预测的 waypoints 转换为：
- `steer`：转向角
- `throttle`：油门
- `brake`：刹车

## 5. 示例：target_point 是如何选择的？

### 重要理解：target_point 不是 XML 中固定的某个点

**target_point 是动态选择的，取决于车辆当前位置！**

### 示例：XML Route 1852

假设 XML 中有这些 waypoints（简化示例）：
```xml
<position x="1362.2" y="5317.5" z="370.5" />  <!-- 起点 -->
<position x="1362.2" y="5319.5" z="370.5" />
<position x="1362.2" y="5321.5" z="370.5" />
...
<position x="1362.6" y="5451.5" z="370.1" />  <!-- 终点 -->
```

### 选择逻辑

1. **XML waypoints 被插值成密集路径点**
   - 73 个粗略点 → 数百个密集点（每个点间隔约 1 米）

2. **每个 Step，RoutePlanner.run_step() 动态选择 target_point**
   ```python
   # 选择条件：
   min_distance = 7.5 米  # 最小距离
   max_distance = 50.0 米  # 最大距离
   
   # 算法：
   # 1. 从路径点中找到距离车辆在 7.5-50 米范围内的点
   # 2. 选择距离最远的那个点作为 target_point
   # 3. 返回 waypoint_route[1]（第二个点，第一个是当前位置）
   ```

3. **具体例子**

   **情况 A：车辆在起点附近**
   - 车辆位置：`(1362.2, 5317.5)` （接近第一个 waypoint）
   - RoutePlanner 选择：距离车辆约 7.5-50 米范围内的点
   - target_point 可能是：`(1362.2, 5325.5)` 或 `(1362.2, 5330.5)` 等
   - **不是固定的！** 取决于车辆精确位置

   **情况 B：车辆在中间位置**
   - 车辆位置：`(1362.3, 5360.0)` （中间某个位置）
   - RoutePlanner 选择：前方 7.5-50 米范围内的点
   - target_point 可能是：`(1362.4, 5400.0)` 或 `(1362.4, 5405.0)` 等

   **情况 C：车辆接近终点**
   - 车辆位置：`(1362.5, 5440.0)` （接近最后一个 waypoint）
   - RoutePlanner 选择：前方剩余的点
   - target_point 可能是：`(1362.6, 5451.5)` （最后一个 waypoint）

### 关键点

1. **target_point 不是 XML 中固定的某个点**
   - XML 中的 73 个 waypoints 都是"候选点"
   - 实际选择哪个点作为 target_point 取决于车辆当前位置

2. **选择是动态的**
   - 每个 Step（每 0.05 秒）都会重新选择
   - 随着车辆前进，target_point 会不断更新

3. **选择范围**
   - 目标点必须在车辆前方 7.5-50 米范围内
   - 选择距离最远的点（但不超过 50 米）

4. **为什么这样设计？**
   - 太近（<7.5米）：目标点太近，车辆反应不过来
   - 太远（>50米）：目标点太远，不够精确
   - 动态选择：适应车辆当前速度，保持合适的"前瞻距离"

## 6. 什么时候车辆会停止？Route 的终止条件

对于每个 route，车辆会在以下情况下停止：

### 成功完成 Route

**条件：RouteCompletionTest**
```python
# 需要同时满足两个条件：
PERCENTAGE_THRESHOLD = 90%  # 完成度 > 90%
DISTANCE_THRESHOLD = 10.0 米  # 距离终点 < 10米
```

- 车辆完成了 **90% 以上的路径**
- 并且距离最后一个 waypoint **小于 10 米**
- Route 标记为 **SUCCESS**，车辆停止

### 失败终止（提前停止）

以下任一条件满足时，route 会**立即终止**（标记为 FAILURE）：

#### 1. **碰撞（CollisionTest）**
- 车辆与其他车辆、行人或静态物体发生碰撞

#### 2. **偏离路线（OutsideRouteLanesTest）**
- 车辆偏离了规划的路线车道

#### 3. **严重偏离路线（InRouteTest）**
- 车辆偏离路线超过 **30 米**
- `terminate_on_failure=True`，会立即终止

#### 4. **闯红灯（RunningRedLightTest）**
- 车辆在红灯时通过路口

#### 5. **闯停车标志（RunningStopTest）**
- 车辆在停车标志前没有停车

#### 6. **速度太慢（MinimumSpeedRouteTest）**
- 车辆在多个检查点速度都太慢

#### 7. **车辆被阻塞（ActorBlockedTest）**
- 车辆速度 < 0.1 m/s 持续超过 **60 秒**
- `terminate_on_failure=True`，会立即终止

#### 8. **超时（RouteTimeoutBehavior）**
- 初始超时：**300 秒**（5 分钟）
- 动态调整：根据车辆进度和道路限速动态增加超时时间
- 计算公式：
  ```python
  timeout_speed = max_speed * 10%  # 限速的 10%
  timeout += 行驶距离 / timeout_speed
  ```
- 如果实际用时超过动态计算的超时时间，route 终止

### 示例：Route 1852

对于你选中的 Route 1852：
- **起点**：`(1362.2, 5317.5, 370.5)`
- **终点**：`(1362.6, 5451.5, 370.1)`
- **距离**：约 134 米（73 个 waypoints，每个间隔约 2 米）

**成功完成条件：**
- 完成度 > 90%（约 120 米）
- 距离终点 < 10 米

**失败条件：**
- 碰撞、偏离路线、闯红灯、被阻塞超过 60 秒、超时等

### 关键点

1. **Route 不会无限运行**
   - 有明确的成功和失败条件
   - 成功：完成 90%+ 且接近终点
   - 失败：违反交通规则、碰撞、超时等

2. **提前终止机制**
   - 某些失败条件（如偏离 30 米、被阻塞 60 秒）会立即终止
   - 避免浪费计算资源

3. **超时是动态的**
   - 不是固定的 300 秒
   - 根据车辆进度和道路限速动态调整
   - 如果车辆开得快，超时时间会增加

## 总结

### 数据流图

```
XML Waypoints (全局路径)
    ↓
RoutePlanner.set_route() (存储所有waypoints)
    ↓
每个 Step:
    RoutePlanner.run_step() (根据当前位置选择下一个目标点)
    ↓
    target_point (从waypoints中提取的单个目标点)
    ↓
    Model 输入:
        - Image (相机图像)
        - target_point (目标点坐标)
        - speed (当前速度)
        - prompt (文本描述)
    ↓
    Model 预测:
        - pred_route (预测的路径点序列，用于到达target_point)
    ↓
    PID 控制器:
        - steer, throttle, brake (车辆控制信号)
```

### 关键理解

1. **XML Waypoints 的作用**
   - XML waypoints 提供**全局路径规划**（粗略的关键点，73个点）
   - 通过插值生成**密集路径点和命令**（数百个点，每个点都有命令）
   - 为 RoutePlanner 提供数据源，用于选择目标点和命令

2. **XML Waypoints ≠ Model 输入**
   - XML 的 waypoints 不直接传给 model
   - Model 只接收**下一个目标点**（1个点）和**命令**（如果使用 command 模式）
   - 但目标点和命令都是从 XML waypoints 派生出来的

3. **Model 预测的是什么？**
   - Model 预测的是**如何到达目标点的路径点序列**
   - 不是直接使用 XML 的 waypoints
   - 而是根据图像和目标点，预测"这样走可以到达目标点"

4. **为什么需要 Model？**
   - XML waypoints 只是粗略的路径规划
   - 实际驾驶需要考虑：
     - 当前路况（图像）
     - 障碍物
     - 交通规则
     - 平滑的轨迹
   - Model 根据这些信息预测更精细的路径点

5. **完整流程**
   ```
   XML Waypoints (73个粗略点)
       ↓
   插值生成密集路径点和命令 (数百个点，每个点有命令)
       ↓
   RoutePlanner 存储路径点和命令
       ↓
   每个 Step: RoutePlanner 选择下一个目标点和命令
       ↓
   Model 输入: 图像 + 目标点 + 命令（可选）
       ↓
   Model 预测: 到达目标点的路径点序列
       ↓
   PID 控制器: 将路径点转换为控制信号（steer/throttle/brake）
   ```

6. **总结：XML Waypoints 的用途**
   - ✅ 提供全局路径规划（导航地图）
   - ✅ 生成密集路径点和命令（通过插值）
   - ✅ 为 RoutePlanner 提供数据源（选择目标点和命令）
   - ❌ 不直接传给 Model（Model 只接收目标点和命令）

