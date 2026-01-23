# Bench2Drive 评估指南

## 📋 准备工作

### 1. 下载模型

从 Hugging Face 下载 SimLingo 模型：

```bash
cd /local1/mhu/doc_drive_search
conda activate simlingo
python download_model.py --output_dir pretrained/simlingo
```

模型将下载到：`pretrained/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`

### 2. 准备 Bench2Drive 路由文件

**选项A：使用 Bench2Drive-mini（推荐用于快速测试）**

如果你已经有 `Bench2Drive-mini` 文件夹，需要准备路由 XML 文件：

```bash
# 方法1：从 Bench2Drive 官方下载路由文件
# 路由文件需要单独下载，不在数据集中
# 可以从 Bench2Drive GitHub 仓库获取

# 方法2：使用准备脚本（简化版）
python prepare_routes_mini.py \
    --mini_dir Bench2Drive/Bench2Drive-mini \
    --output Bench2Drive/leaderboard/data/bench2drive_mini.xml
```

**选项B：下载完整路由文件**

从 Bench2Drive 官方获取路由 XML 文件：
- GitHub: https://github.com/Thinklab-SJTU/Bench2Drive
- 路由文件应该在 `leaderboard/data/` 目录下

**注意**：路由文件格式为 XML，需要放在 `Bench2Drive/leaderboard/data/` 目录下。

### 3. 配置评估脚本

编辑 `start_eval_simlingo.py`，确保以下路径正确：

- ✅ `checkpoint`: 模型检查点路径（已更新）
- ✅ `carla_root`: CARLA 安装路径（已更新）
- ✅ `repo_root`: 项目根目录（已更新）
- ⚠️ `route_path`: Bench2Drive 路由文件路径（需要下载路由文件）
- ⚠️ `username`: SLURM 用户名（需要修改为你的用户名）
- ⚠️ `partition_name`: SLURM 分区名称（需要修改为你的分区）

## 🚀 运行评估

### 前提条件

1. **环境已激活**：
   ```bash
   conda activate simlingo
   source /local1/mhu/doc_drive_search/carla_exports.sh
   ```

2. **SLURM 集群可用**（脚本使用 SLURM 提交任务）

**注意**：✅ **不需要手动启动 CARLA 服务器**。评估脚本会自动为每个任务启动独立的 CARLA 实例（使用 `-RenderOffScreen` 无头模式），每个任务使用不同的端口。

### 运行评估

**方法1：本地运行（不使用 SLURM，推荐）**

```bash
cd /local1/mhu/doc_drive_search
conda activate simlingo
source carla_exports.sh

# 运行本地评估脚本
bash run_eval_local.sh
```

**方法2：使用 SLURM（集群环境）**

```bash
cd /local1/mhu/doc_drive_search
conda activate simlingo
source carla_exports.sh
python start_eval_simlingo.py
```

## 📊 评估结果

评估结果将保存在：
- `eval_results/Bench2Drive/simlingo/bench2drive/{seed}/res/` - JSON 结果文件
- `eval_results/Bench2Drive/simlingo/bench2drive/{seed}/viz/` - 可视化输出
- `eval_results/Bench2Drive/simlingo/bench2drive/{seed}/out/` - 日志文件
- `eval_results/Bench2Drive/simlingo/bench2drive/{seed}/err/` - 错误日志

### 合并结果

评估完成后，使用工具合并结果：

```bash
cd /local1/mhu/doc_drive_search
python Bench2Drive/tools/merge_route_json.py
```

## ⚙️ 配置说明

### 关键配置项

- **seeds**: 评估种子列表，论文使用一个评估种子在三个训练种子上运行
- **tries**: 失败重试次数
- **max_num_jobs**: 最大并行任务数（在 `max_num_jobs.txt` 中配置）
- **partition**: SLURM 分区名称

### 端口配置

脚本会自动分配 CARLA 端口：
- World ports: 10000-20000 (步长 50)
- Streaming ports: 20000-30000 (步长 50)
- Traffic Manager ports: 30000-40000 (步长 50)

## 🆘 故障排除

### 问题1：模型文件未找到
- **解决**：运行 `python download_model.py` 下载模型

### 问题2：路由文件未找到
- **解决**：下载 Bench2Drive 路由文件到指定目录

### 问题3：SLURM 提交失败
- **检查**：`username` 和 `partition_name` 是否正确
- **检查**：SLURM 集群是否可用

### 问题4：CARLA 连接失败
- **说明**：评估脚本会自动启动 CARLA，如果连接失败可能是：
  - CARLA 启动时间过长（脚本会等待60秒并重试）
  - 端口冲突（脚本会自动查找空闲端口）
  - GPU 资源不足（每个 CARLA 实例需要 GPU）

## 📚 参考

- Bench2Drive 官方文档：`Bench2Drive/README.md`
- 项目指南：`huhu/PROJECT_GUIDE.md`
- Hugging Face 模型：https://huggingface.co/RenzKa/simlingo
- Bench2Drive 数据集：https://huggingface.co/datasets/rethinklab/Bench2Drive

