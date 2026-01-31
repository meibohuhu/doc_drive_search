# Command Alignment Evaluation 数据格式说明

## 概述

`run_command_alignment_eval.py` 脚本用于从已保存的evaluation logs中分析command alignment。本文档说明脚本期望的数据格式。

## 当前 `metric_info.json` 的格式

当前的 `metric_info.json` 只包含车辆状态信息，格式如下：

```json
{
    "1": {
        "acceleration": [x, y, z],
        "angular_velocity": [x, y, z],
        "forward_vector": [x, y, z],
        "right_vector": [x, y, z],
        "location": [x, y, z],
        "rotation": [pitch, yaw, roll]
    },
    "2": { ... },
    ...
}
```

**问题**：这个格式缺少以下关键信息：
- ❌ `predicted_waypoints`: 模型预测的waypoints
- ❌ `actual_command`: 实际执行的command (1-6)
- ❌ `is_in_junction`: 是否在junction内
- ❌ `current_heading`: 当前heading（可以从rotation推导）

## 脚本期望的数据格式

脚本期望一个名为 `step_logs.json` 的文件，格式如下：
predicted_waypoints: 模型预测的waypoints（车辆坐标系）
actual_command: 实际command（从self.last_command_tmp获取）
current_heading: 当前heading（从compass获取，已归一化到[-π, π]）
is_in_junction: 是否在junction内（command 1,2,3表示在junction附近）
metadata: 包含location、speed、target_point等额外信息

```json
[
    {
        "step": 1,
        "predicted_waypoints": [
            [x1, y1],
            [x2, y2],
            ...
            [xN, yN]
        ],
        "actual_command": 4,
        "current_heading": 1.57,
        "is_in_junction": false,
        "metadata": {
            "location": [x, y, z],
            "rotation": [pitch, yaw, roll],
            ...
        }
    },
    {
        "step": 2,
        ...
    },
    ...
]
```

### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `step` | int | ✅ | Step编号（从1开始） |
| `predicted_waypoints` | List[List[float]] | ✅ | 模型预测的waypoints数组，每个waypoint是 `[x, y]` 坐标（车辆坐标系，单位：米） |
| `actual_command` | int | ✅ | 实际执行的command，取值范围：1-6<br/>1=左转, 2=右转, 3=直行, 4=跟随道路, 5=左变道, 6=右变道 |
| `current_heading` | float | ✅ | 当前车辆heading（弧度），范围 [-π, π] |
| `is_in_junction` | bool | ✅ | 是否在junction内 |
| `metadata` | dict | ❌ | 可选的元数据，可以包含其他信息（如location, rotation等） |

### Waypoints格式说明

- **坐标系**：车辆坐标系（vehicle frame）
  - x轴：车辆前进方向
  - y轴：车辆左侧方向
- **单位**：米（meters）
- **数量**：通常为10-20个waypoints

### Command映射

| Command值 | 含义 | 说明 |
|-----------|------|------|
| 1 | 左转 | Turn left |
| 2 | 右转 | Turn right |
| 3 | 直行 | Go straight |
| 4 | 跟随道路 | Follow lane |
| 5 | 左变道 | Change lane left |
| 6 | 右变道 | Change lane right |

## 如何生成符合格式的logs

### 方法1：使用 `agent_simlingo_with_eval.py`

在evaluation时使用 `agent_simlingo_with_eval.py`，它会自动保存符合格式的logs。

### 方法2：手动保存logs

在 `agent_simlingo.py` 的 `run_step()` 方法中添加日志保存逻辑：

```python
def run_step(self, input_data, timestamp):
    # ... 现有代码 ...
    
    # 获取预测的waypoints
    predicted_waypoints = self.model_output  # 需要根据实际代码调整
    
    # 获取实际command
    actual_command = self.current_command  # 需要根据实际代码调整
    
    # 获取当前heading
    current_heading = self.get_current_heading()  # 需要根据实际代码调整
    
    # 检查是否在junction内
    is_in_junction = self.is_in_junction()  # 需要根据实际代码调整
    
    # 保存log entry
    log_entry = {
        "step": self.step_count,
        "predicted_waypoints": predicted_waypoints.tolist(),  # 转换为list
        "actual_command": actual_command,
        "current_heading": current_heading,
        "is_in_junction": is_in_junction,
        "metadata": {
            "location": self.vehicle_location,
            "rotation": self.vehicle_rotation,
            # ... 其他信息
        }
    }
    
    # 追加到logs列表
    self.evaluation_logs.append(log_entry)
```

在 `destroy()` 方法中保存logs：

```python
def destroy(self):
    # ... 现有代码 ...
    
    # 保存evaluation logs
    if hasattr(self, 'evaluation_logs') and len(self.evaluation_logs) > 0:
        log_file = Path(self.save_path) / 'step_logs.json'
        with open(log_file, 'w') as f:
            json.dump(self.evaluation_logs, f, indent=2)
```

## 使用脚本

一旦有了符合格式的 `step_logs.json` 文件，可以运行：

```bash
python scripts/run_command_alignment_eval.py \
    --log_dir /path/to/evaluation/results \
    --output /path/to/output.json
```

脚本会在指定目录下查找 `step_logs.json` 文件并进行分析。

## 总结

- ✅ **需要**：`step_logs.json` 包含每个step的预测waypoints和实际command
- ❌ **当前**：`metric_info.json` 只包含车辆状态，不包含预测和command信息
- 🔧 **解决方案**：需要在evaluation时额外保存 `step_logs.json` 文件

