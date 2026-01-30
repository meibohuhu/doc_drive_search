"""
Modified agent_simlingo.py with stepwise command alignment evaluation

在agent中添加command alignment评估功能
"""

# 在agent_simlingo.py的基础上添加评估功能
# 主要修改点：
# 1. 在setup()中初始化CommandAlignmentEvaluator
# 2. 在run_step()中每个step调用evaluator
# 3. 在destroy()中保存评估结果

# 需要导入
from team_code.command_alignment_evaluator import CommandAlignmentEvaluator

# 在LingoAgent.setup()中添加：
# self.command_evaluator = CommandAlignmentEvaluator()

# 在LingoAgent.run_step()中，模型预测后添加：
"""
# 评估command alignment
if hasattr(self, 'command_evaluator'):
    # 获取当前command
    current_command = self.last_command_tmp.value if hasattr(self.last_command_tmp, 'value') else self.last_command_tmp
    
    # 获取当前heading（从compass或IMU）
    current_heading = tick_data['compass']
    
    # 判断是否在junction（可以从measurement或route planner获取）
    is_in_junction = False  # TODO: 从实际数据获取
    
    # 转换waypoints到车辆坐标系（如果需要）
    if pred_route is not None:
        pred_waypoints_ego = pred_route[0].detach().cpu().numpy()  # (20, 2)
        
        # 评估
        eval_result = self.command_evaluator.evaluate_step(
            step=self.step,
            predicted_waypoints=pred_waypoints_ego,
            actual_command=current_command,
            current_heading=current_heading,
            is_in_junction=is_in_junction,
            metadata={
                'speed': float(tick_data['speed']),
                'target_point': tick_data.get('target_point', None),
            }
        )
        
        # 可选：打印对齐情况
        if not eval_result['is_aligned']:
            print(f"[Step {self.step}] Command misalignment: "
                  f"Actual={eval_result['actual_command_str']}, "
                  f"Inferred={eval_result['inferred_command_str']}")
"""

# 在LingoAgent.destroy()中添加：
"""
if hasattr(self, 'command_evaluator'):
    # 保存评估结果
    eval_filepath = os.path.join(self.save_path, 'command_alignment_eval.json')
    self.command_evaluator.save_results(eval_filepath)
    
    # 打印统计信息
    stats = self.command_evaluator.get_statistics()
    print("\n" + "="*60)
    print("Command Alignment Evaluation Results")
    print("="*60)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Aligned steps: {stats['aligned_steps']}")
    print(f"Overall alignment rate: {stats['overall_alignment_rate']:.2%}")
    print("\nPer-command alignment rates:")
    for cmd, cmd_stats in stats['command_alignment_rates'].items():
        print(f"  Command {cmd} ({cmd_stats['command_str']}): "
              f"{cmd_stats['aligned']}/{cmd_stats['total']} = {cmd_stats['alignment_rate']:.2%}")
    print("="*60)
"""

