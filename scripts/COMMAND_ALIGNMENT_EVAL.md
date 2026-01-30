# Stepwise Command Alignment Evaluation Guide

## 概述

这个工具用于评估模型在每个step预测的waypoints是否与navigation command对齐。

## 问题背景

当前的评估指标（Driving score和success rate）是route-level的，只能知道整个trajectory是否成功完成，但无法知道：
- 模型在每个step是否理解了navigation command
- 模型预测的waypoints是否符合预期的command（左转、右转、变道等）
- 哪些command类型模型更容易出错

## 解决方案

### 1. Command Alignment Evaluator

`CommandAlignmentEvaluator`类提供了stepwise的command alignment评估：

- **输入**：每个step的
  - 模型预测的waypoints（车辆坐标系）
  - 实际应该执行的command（1-6）
  - 当前车辆状态（heading, 是否在junction等）

- **输出**：
  - 从waypoints推断的command
  - 是否与actual command对齐
  - 详细的统计信息

### 2. Command定义

```python
COMMAND_MAP = {
    1: 'go left at the next intersection',      # 左转
    2: 'go right at the next intersection',      # 右转
    3: 'go straight at the next intersection',   # 直行
    4: 'follow the road',                        # 跟随道路
    5: 'do a lane change to the left',          # 左变道
    6: 'do a lane change to the right',         # 右变道
}
```

### 3. Command推断方法

从waypoints推断command的逻辑：

1. **变道判断**（优先级最高）
   - 如果横向位移（y方向）> 阈值（默认1.5米）
   - 左变道：y增加 → command 5
   - 右变道：y减少 → command 6

2. **转弯判断**（在路口或接近路口时）
   - 分析waypoints的方向角度变化
   - 左转：角度 > 阈值（默认15度） → command 1
   - 右转：角度 < -阈值 → command 2
   - 直行：角度在阈值内 → command 3

3. **默认**
   - 跟随道路 → command 4

## 使用方法

### 方法1：在Agent中集成（推荐）

修改`agent_simlingo.py`，添加评估功能：

```python
# 1. 在setup()中初始化
from team_code.command_alignment_evaluator import CommandAlignmentEvaluator
self.command_evaluator = CommandAlignmentEvaluator()

# 2. 在run_step()中每个step评估
if pred_route is not None:
    pred_waypoints_ego = pred_route[0].detach().cpu().numpy()
    current_command = self.last_command_tmp.value
    
    eval_result = self.command_evaluator.evaluate_step(
        step=self.step,
        predicted_waypoints=pred_waypoints_ego,
        actual_command=current_command,
        current_heading=tick_data['compass'],
        is_in_junction=False,  # TODO: 从实际数据获取
    )

# 3. 在destroy()中保存结果
if hasattr(self, 'command_evaluator'):
    eval_filepath = os.path.join(self.save_path, 'command_alignment_eval.json')
    self.command_evaluator.save_results(eval_filepath)
    stats = self.command_evaluator.get_statistics()
    print(f"Overall alignment rate: {stats['overall_alignment_rate']:.2%}")
```

### 方法2：从Logs分析（如果不想修改agent）

如果evaluation时已经保存了step-level的logs，可以使用独立脚本：

```bash
python scripts/run_command_alignment_eval.py \
    --log_dir eval_results/Bench2Drive/simlingo_mini/route_0 \
    --output command_alignment_eval.json
```

## 输出结果

### 1. Step-level结果

每个step的评估结果包含：
- `actual_command`: 实际command
- `inferred_command`: 从waypoints推断的command
- `is_aligned`: 是否对齐
- `lateral_displacement`: 横向位移
- `forward_displacement`: 纵向位移

### 2. 统计信息

- **Overall alignment rate**: 整体对齐率
- **Per-command alignment rates**: 每个command类型的对齐率
- **Confusion matrix**: 混淆矩阵（actual vs inferred）

### 3. 示例输出

```
================================================================
Command Alignment Evaluation Results
================================================================
Total steps: 1250
Aligned steps: 1080
Overall alignment rate: 86.40%

Per-command alignment rates:
  Command 1 (go left at the next intersection): 45/50 = 90.00%
  Command 2 (go right at the next intersection): 42/48 = 87.50%
  Command 3 (go straight at the next intersection): 38/45 = 84.44%
  Command 4 (follow the road): 850/950 = 89.47%
  Command 5 (do a lane change to the left): 65/80 = 81.25%
  Command 6 (do a lane change to the right): 40/77 = 51.95%

Confusion Matrix (Actual -> Inferred):
  Actual 1 (go left at the next intersection):
    -> Inferred 1: 45
    -> Inferred 4: 5
  ...
================================================================
```

## 参数调整

可以根据实际情况调整`CommandAlignmentEvaluator`的参数：

```python
evaluator = CommandAlignmentEvaluator(
    lateral_threshold=1.5,      # 变道判断阈值（米）
    turn_angle_threshold=15.0,   # 转弯角度阈值（度）
    lookahead_distance=10.0     # 用于判断command的lookahead距离（米）
)
```

## 注意事项

1. **坐标系**：waypoints需要在车辆坐标系（ego coordinates）中
2. **Junction判断**：需要准确判断是否在junction内，这会影响转弯判断
3. **阈值选择**：不同场景可能需要不同的阈值，建议根据实际数据调整
4. **边界情况**：waypoints为空或数量很少时的处理

## 未来改进

1. 使用更复杂的几何分析（曲率、路径规划等）来推断command
2. 考虑时间序列信息（多个step的waypoints）
3. 结合HD map信息提高判断准确性
4. 可视化misalignment的cases用于分析

