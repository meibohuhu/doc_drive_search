# Bench2Drive 本地评估指南（不使用 SLURM）

## 📋 快速开始

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

## 🆘 常见问题

### 问题1：找不到路由文件

**解决**：
1. 从 Bench2Drive GitHub 下载路由文件
2. 或使用 `prepare_routes_mini.py` 生成简化版本

### 问题2：端口被占用

**解决**：修改 `run_eval_local.sh` 中的端口号：
```bash
BASE_PORT=2001  # 改为其他端口
BASE_TM_PORT=8001
```

### 问题3：CARLA 启动失败

**解决**：
- 检查 CARLA_ROOT 是否正确
- 确保 GPU 可用：`nvidia-smi`
- 清理旧的 CARLA 进程：`bash Bench2Drive/tools/clean_carla.sh`

### 问题4：模型加载失败

**解决**：
- 检查模型路径是否正确
- 确认模型文件完整：`ls -lh pretrained/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`

## 📝 注意事项

1. **GPU 内存**：每个评估任务需要约 15-20 GB 显存
2. **CARLA 自动启动**：脚本会自动启动 CARLA 服务器，无需手动启动
3. **评估时间**：单个路由评估可能需要几分钟到十几分钟
4. **并行评估**：如果需要并行评估多个路由，需要修改脚本使用不同端口

## 🔄 评估多个路由

如果需要评估多个路由，可以创建一个循环脚本：

```bash
#!/bin/bash
# 评估多个路由

ROUTES_DIR="/local1/mhu/doc_drive_search/Bench2Drive/leaderboard/data"
for route_file in ${ROUTES_DIR}/*.xml; do
    echo "评估路由: $route_file"
    # 修改 run_eval_local.sh 中的 ROUTES 变量
    # 或创建新的评估脚本
done
```









