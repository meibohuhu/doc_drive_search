# Bench2Drive 评估结果评分分析

## 评分系统概述

Bench2Drive 评估使用三个主要指标来评估自动驾驶代理的性能：

### 1. 核心评分指标

#### **score_composed (综合驾驶分数)**
- **范围**: 0-100
- **含义**: 综合考虑路线完成度和违规惩罚的最终分数
- **计算公式**: `score_composed = score_route × score_penalty`
- **100分**: 完美完成路线且无严重违规

#### **score_route (路线完成分数)**
- **范围**: 0-100
- **含义**: 完成路线的百分比
- **100分**: 成功完成整个路线

#### **score_penalty (违规惩罚分数)**
- **范围**: 0-1
- **含义**: 违规惩罚系数
- **1.0**: 无惩罚（完美）
- **< 1.0**: 根据违规类型和严重程度扣分

## 违规类型说明

### 严重违规（会显著降低 score_penalty）

根据 `statistics_manager.py` 中的 `PENALTY_VALUE_DICT`，以下违规会直接乘以惩罚系数：

1. **collisions_pedestrian** - 与行人碰撞
   - **惩罚系数**: 0.5
   - **严重性**: 极高
   - **扣分**: `score_penalty *= 0.5`（每次碰撞）

2. **collisions_vehicle** - 与车辆碰撞
   - **惩罚系数**: 0.6
   - **严重性**: 高
   - **扣分**: `score_penalty *= 0.6`（每次碰撞）

3. **collisions_layout** - 与静态物体碰撞
   - **惩罚系数**: 0.65
   - **严重性**: 高
   - **扣分**: `score_penalty *= 0.65`（每次碰撞）

4. **red_light** - 闯红灯
   - **惩罚系数**: 0.7
   - **严重性**: 高
   - **扣分**: `score_penalty *= 0.7`（每次违规）

5. **stop_infraction** - 违反停止标志
   - **惩罚系数**: 0.8
   - **严重性**: 中-高
   - **扣分**: `score_penalty *= 0.8`（每次违规）

6. **scenario_timeouts** - 场景超时
   - **惩罚系数**: 0.7
   - **严重性**: 中
   - **扣分**: `score_penalty *= 0.7`

7. **yield_emergency_vehicle_infractions** - 未让行紧急车辆
   - **惩罚系数**: 0.7
   - **严重性**: 中
   - **扣分**: `score_penalty *= 0.7`

### 其他违规类型

8. **outside_route_lanes** - 偏离路线车道
   - **类型**: 百分比惩罚（`PENALTY_PERC_DICT`）
   - **严重性**: 中
   - **扣分**: 根据偏离程度动态计算

9. **route_dev** - 路线偏差
   - **严重性**: 中
   - **影响**: 可能导致路线失败（`failure_message = "Agent deviated from the route"`）

10. **route_timeout** - 路线超时
    - **严重性**: 中
    - **影响**: 如果超时则 `score_route = 0`

11. **vehicle_blocked** - 车辆被阻塞
    - **严重性**: 低
    - **影响**: 通常不影响分数，但会记录

### ⚠️ 重要：min_speed_infractions 不扣分

12. **min_speed_infractions** - 最小速度违规
    - **状态**: **'unused'**（在代码中标记为 `[0.7, 'unused']`）
    - **严重性**: 低（仅记录）
    - **扣分**: **不影响 `score_penalty`**
    - **含义**: 车辆速度与周围交通速度的差异
    - **示例**: 速度是周围交通的 193%-414% 倍
    - **在 Bench2Drive 评估中，min_speed_infractions 只作为记录，不扣分**

**代码证据**：
- `statistics_manager.py` 第 37 行：`TrafficEventType.MIN_SPEED_INFRACTION: [0.7, 'unused']`
- `merge_route_json.py` 第 24-26 行：在判断 success_flag 时明确排除 `min_speed_infractions`

## 评分公式

```
score_composed = max(score_route × score_penalty, 0.0)

其中：
- score_route = (已完成路线长度 / 总路线长度) × 100
- score_penalty = 初始值为 1.0，然后根据违规类型乘以相应的惩罚系数
```

**惩罚系数累积**：
- 每次违规都会乘以对应的惩罚系数
- 例如：如果发生一次车辆碰撞，`score_penalty = 1.0 × 0.6 = 0.6`
- 如果又发生一次闯红灯，`score_penalty = 0.6 × 0.7 = 0.42`

## 示例结果分析

### 示例 1: 完美分数（100分）

```json
{
    "score_composed": 100.0,
    "score_route": 100.0,
    "score_penalty": 1.0,
    "infractions": {
        "collisions_layout": 0.0,
        "collisions_pedestrian": 0.0,
        "collisions_vehicle": 0.0,
        "red_light": 0.0,
        "stop_infraction": 0.0,
        "outside_route_lanes": 0.0,
        "min_speed_infractions": 238.663,  // ⚠️ 仅记录，不扣分
        "yield_emergency_vehicle_infractions": 0.0,
        "scenario_timeouts": 0.0,
        "route_dev": 0.0,
        "vehicle_blocked": 0.0,
        "route_timeout": 0.0
    }
}
```

**分析**：
- ✅ **Driving Score: 100.0** - 完美分数
- ✅ **Route Completion: 100%** - 成功完成整个路线
- ✅ **Penalty: 1.0** - 无严重违规
- ⚠️ **Min Speed Infractions: 238.663** - 有速度违规记录，但**不影响分数**
  - 车辆速度比周围交通快很多（193%-414%）
  - 这可能是模型过于激进，但 Bench2Drive 评估中不扣分

### 示例 2: 有违规的结果

```json
{
    "score_composed": 99.285864,
    "score_route": 100.0,
    "score_penalty": 0.992859,
    "infractions": {
        "collisions_layout": 0.178,
        "outside_route_lanes": 0.004,
        "min_speed_infractions": 189.929  // 不影响分数
    }
}
```

**分析**：
- ⚠️ **Driving Score: 99.29** - 略低于完美分数
- ✅ **Route Completion: 100%** - 成功完成路线
- ⚠️ **Penalty: 0.993** - 有轻微违规扣分
- 可能原因：有轻微的静态物体碰撞（0.178）和路线偏离（0.004）

## 评估结果文件结构

评估结果保存在 JSON 文件中，包含以下信息：

```json
{
    "_checkpoint": {
        "global_record": {
            "scores_mean": {
                "score_composed": 100.0,    // 综合分数
                "score_route": 100.0,        // 路线完成度
                "score_penalty": 1.0         // 惩罚系数
            },
            "infractions": {
                // 各种违规的统计（数值表示累计影响）
            }
        },
        "records": [
            {
                "scores": {
                    "score_route": 100,
                    "score_penalty": 1.0,
                    "score_composed": 100.0
                },
                "infractions": {
                    // 详细的违规列表（数组形式）
                }
            }
        ]
    },
    "values": [
        "100.0",  // Avg. driving score (score_composed)
        "100.0",  // Avg. route completion (score_route)
        "1.0",    // Avg. infraction penalty (score_penalty)
        "0.0",    // Collisions with pedestrians
        "0.0",    // Collisions with vehicles
        "0.0",    // Collisions with layout
        "0.0",    // Red lights infractions
        "0.0",    // Stop sign infractions
        "0.0",    // Off-road infractions
        "0.0",    // Route deviations
        "0.0",    // Route timeouts
        "0.0",    // Agent blocked
        "0.0",    // Yield emergency vehicles infractions
        "0.0",    // Scenario timeouts
        "238.663" // Min speed infractions (仅记录，不扣分)
    ],
    "labels": [
        "Avg. driving score",
        "Avg. route completion",
        "Avg. infraction penalty",
        "Collisions with pedestrians",
        "Collisions with vehicles",
        "Collisions with layout",
        "Red lights infractions",
        "Stop sign infractions",
        "Off-road infractions",
        "Route deviations",
        "Route timeouts",
        "Agent blocked",
        "Yield emergency vehicles infractions",
        "Scenario timeouts",
        "Min speed infractions"
    ]
}
```

## 如何解读结果

### 完美结果（100分）
- `score_composed = 100.0`
- `score_route = 100.0`
- `score_penalty = 1.0`
- 所有严重违规 = 0.0
- `min_speed_infractions` 可能有值，但不影响分数

### 有违规的结果
- `score_composed < 100.0` 表示有违规扣分
- 检查 `infractions` 部分查看具体违规类型
- `min_speed_infractions` 通常不影响分数

### 未完成路线
- `score_route < 100.0` 表示未完成整个路线
- 可能原因：超时、碰撞导致停止、路线偏差过大

## Success Rate 计算

根据 `merge_route_json.py` 的逻辑，Success Rate 的计算方式：

```python
success_flag = True
for k, v in rd['infractions'].items():
    if len(v) > 0 and k != 'min_speed_infractions':  # 排除 min_speed_infractions
        success_flag = False
        break
```

**成功条件**：
- 状态为 `'Completed'` 或 `'Perfect'`
- 除了 `min_speed_infractions` 外，没有其他违规

**重要**：`min_speed_infractions` 在计算 Success Rate 时也被排除！

## 注意事项

1. **min_speed_infractions 不扣分**：
   - 虽然记录了很多速度违规，但这些在 Bench2Drive 评估中不影响最终分数
   - 在计算 Success Rate 时也被排除
   - 仅作为行为分析的数据记录

2. **分数计算**：
   - `score_composed = score_route × score_penalty`
   - 即使路线完成度是 100%，如果有严重违规，综合分数也会降低
   - 惩罚系数是累积的（每次违规都会乘以对应的系数）

3. **评估标准**：
   - Bench2Drive 主要关注安全性（避免碰撞、遵守交通规则）和路线完成度
   - 速度控制相对次要（min_speed_infractions 不扣分）

4. **结果文件位置**：
   - 单个路由结果：`eval_results_rc/Bench2Drive/simlingo/bench2drive/{seed}/res/{route_id}_res.json`
   - 可视化结果：`eval_results_rc/Bench2Drive/simlingo/bench2drive/{seed}/viz/{route_id}/`
   - 合并结果：使用 `Bench2Drive/tools/merge_route_json.py` 生成 `merged.json`

## 参考

- Bench2Drive 评估代码：`Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py`
- 评分管理代码：`Bench2Drive/leaderboard/leaderboard/utils/statistics_manager.py`
- 结果合并工具：`Bench2Drive/tools/merge_route_json.py`
- 评估结果示例：`eval_results_rc/Bench2Drive/simlingo/bench2drive/1/res/1_res.json`

## 惩罚系数总结表

| 违规类型 | 惩罚系数 | 说明 |
|---------|---------|------|
| collisions_pedestrian | 0.5 | 与行人碰撞 |
| collisions_vehicle | 0.6 | 与车辆碰撞 |
| collisions_layout | 0.65 | 与静态物体碰撞 |
| red_light | 0.7 | 闯红灯 |
| scenario_timeouts | 0.7 | 场景超时 |
| yield_emergency_vehicle | 0.7 | 未让行紧急车辆 |
| stop_infraction | 0.8 | 违反停止标志 |
| outside_route_lanes | 动态 | 偏离路线车道（百分比惩罚） |
| min_speed_infractions | **不扣分** | 速度违规（仅记录） |

