# Bench2Drive 本地评估指南（不使用 SLURM）

## 📋 快速开始
1. CARLA提供: GPS位置, Speed, RGB图像, IMU
   ↓
2. RoutePlanner.run_step() → 计算target_point和command
   ↓ target_point 和 command 不是 CARLA 直接提供，而是根据当前 GPS 位置和全局路径动态计算
3. 构建模型输入: image + speed + (target_point或command)
   ↓
4. 模型预测: pred_route (20 waypoints) + pred_speed_wps (10 waypoints)
   ↓
5. control_pid() → 转换为steer, throttle, brake
   ↓ 用 pred_route（20个waypoints）通过 LateralPIDController 计算 steer
   ↓ 用 pred_speed_wps（10个waypoints）计算 desired_speed，再通过 PIDController 计算 throttle/brake
7. 循环回到步骤1

### 1. 准备路由文件

Bench2Drive-mini 包含的是数据文件（tar.gz），路由 XML 文件需要单独获取。

**选项A：从 Bench2Drive GitHub 获取（推荐）**

```bash
# 克隆 Bench2Drive 仓库（如果还没有）
cd /local1/mhu/doc_drive_search
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git Bench2Drive_repo

# 路由文件应该在 leaderboard/data/ 目录下
# 检查是否有 bench2drive220.xml 或其他路由文件
ls Bench2Drive_repo/leaderboard/data/*.xml

# 复制到项目目录
cp Bench2Drive_repo/leaderboard/data/bench2drive220.xml \
   Bench2Drive/leaderboard/data/bench2drive_mini.xml
```

**选项B：使用单个路由文件测试**

如果只有单个路由，可以创建一个简单的 XML 文件。

### 2. 检查模型文件

```bash
# 确认模型文件存在
ls -lh /local1/mhu/doc_drive_search/pretrained/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

### 3. 运行本地评估

```bash
cd /local1/mhu/doc_drive_search
conda activate simlingo
source carla_exports.sh

# 运行评估
bash run_eval_local.sh
```

## 🔧 自定义配置

编辑 `run_eval_local.sh` 修改以下参数：

- `ROUTES`: 路由 XML 文件路径
- `TEAM_CONFIG`: 模型检查点路径
- `CHECKPOINT_ENDPOINT`: 结果输出路径
- `SAVE_PATH`: 可视化输出路径
- `GPU_RANK`: GPU 编号（0 或 1）
- `BASE_PORT`: CARLA 端口（默认 2000）
- `BASE_TM_PORT`: Traffic Manager 端口（默认 8000）

## 📊 评估结果

评估完成后，结果保存在：
- JSON 结果：`eval_results/Bench2Drive/simlingo_mini.json`
- 可视化输出：`eval_results/Bench2Drive/simlingo_mini/`

### 查看结果

```bash
# 查看 JSON 结果
cat eval_results/Bench2Drive/simlingo_mini.json

# 合并结果（如果有多个路由）
python Bench2Drive/tools/merge_route_json.py -f eval_results/Bench2Drive/
```










