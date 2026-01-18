# 数据集下载与训练完整指南

## 📋 概述

本指南将帮助你：
1. ✅ 从 Hugging Face 下载 SimLingo 数据集
2. ✅ 配置训练环境
3. ✅ 开始训练模型

**注意**：本指南不需要 CARLA，只需要数据集和训练代码。

---

## 阶段一：环境准备

### 1.1 创建 Conda 环境

```bash
# 进入项目目录
cd /home/mh2803/projects/simlingo

# 创建环境
conda env create -f environment.yaml
conda activate simlingo

# 安装 PyTorch（确保CUDA版本正确）
pip install torch==2.2.0

# 安装 flash-attn（可选，用于加速训练，但不是必需的）
# ⚠️ 如果遇到CUDA环境问题，可以安全跳过此步骤
# 训练仍然可以正常运行，只是速度会稍慢一些
# 
# 如果需要安装flash-attn，可以运行：
#   bash install_flash_attn.sh
# 或者手动设置CUDA环境后安装：
#   module load cuda/7.5  # 根据你的系统调整
#   export CUDA_HOME=/usr/local/cuda-7.5
#   pip install flash-attn==2.7.0.post2
#
# 如果跳过flash-attn，直接继续下一步即可 ✅
```

### 1.2 验证环境（可选）

```bash
# 验证PyTorch和CUDA是否正常工作
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### 1.3 安装 Git LFS（用于下载大文件）

```bash
# Ubuntu/Debian
sudo apt install git-lfs

# 初始化 Git LFS
git lfs install
```

### 1.4 配置 Wandb（训练日志）

```bash
# 登录 Wandb（训练需要）
wandb login
# 输入你的 API key（如果没有账号，先去 https://wandb.ai 注册）
```

---

## 阶段二：下载数据集

### 2.1 下载完整数据集

数据集包含：
- 驾驶数据（RGB图像、轨迹等）
- VQA 标签
- Commentary 标签
- Dreamer 数据
- Bucket 文件（用于数据采样）

```bash
# 创建数据集目录
mkdir -p database

# 克隆数据集仓库（使用 Git LFS）
cd database
git clone https://huggingface.co/datasets/RenzKa/simlingo

# 进入数据集目录
cd simlingo

# 拉取 LFS 文件（这一步会下载所有大文件，可能需要较长时间）
git lfs pull

# 返回项目根目录
cd ../..
```

**预计下载时间**：取决于网络速度，可能需要数小时（数据集很大）

### 2.2 解压数据集

```bash
# 进入数据集目录
cd database/simlingo

# 解压所有 tar.gz 文件到统一目录
mkdir -p ../simlingo_extracted
for file in *.tar.gz; do
    echo "正在解压 $file ..."
    tar -xzf "$file" -C ../simlingo_extracted/
done

# 返回项目根目录
cd ../..
```

**注意**：确保有足够的磁盘空间（至少 100-200 GB）

### 2.3 验证数据集结构

解压后，数据集应该包含以下结构：

```
database/
├── simlingo_extracted/          # 解压后的主数据集
│   ├── driving_data/            # 驾驶数据
│   ├── vqa_labels/              # VQA标签
│   ├── commentary_labels/        # Commentary标签
│   └── dreamer_data/            # Dreamer数据
└── bucketsv2_simlingo/          # Bucket文件（如果单独下载）
```

如果 bucket 文件在数据集中，检查是否有 `bucketsv2_simlingo` 目录。

---

## 阶段三：配置训练

### 3.1 检查数据集路径

训练配置使用 Hydra，配置文件位于 `simlingo_training/config/`。

主要配置文件：
- `simlingo_training/config/experiment/simlingo_seed1.yaml` - 完整模型训练
- `simlingo_training/config/experiment/debug.yaml` - 调试/小规模训练

### 3.2 修改数据集路径

编辑配置文件，修改数据集路径：

```bash
# 编辑训练配置
vim simlingo_training/config/experiment/simlingo_seed1.yaml
```

找到并修改以下部分：

```yaml
data_module:
  base_dataset:
    data_path: database/simlingo_extracted  # 修改为你的数据集路径
    bucket_path: database/bucketsv2_simlingo  # 修改为你的bucket路径
```

**重要**：
- `data_path`：指向解压后的数据集主目录
- `bucket_path`：指向bucket文件目录（如果bucket文件在数据集中，可能需要调整路径）

### 3.3 调整训练参数（可选）

根据你的硬件配置调整：

```yaml
data_module:
  batch_size: 6        # 根据GPU显存调整（8GB显存建议2-4，16GB建议6-8）
  num_workers: 8       # 数据加载线程数（建议等于CPU核心数）

gpus: 8                # 使用的GPU数量（单卡改为1）
max_epochs: 15         # 训练轮数
```

### 3.4 单GPU训练配置示例

如果只有单GPU，可以基于 `debug.yaml` 创建配置：

```bash
# 复制debug配置作为起点
cp simlingo_training/config/experiment/debug.yaml simlingo_training/config/experiment/my_training.yaml
```

编辑 `my_training.yaml`：

```yaml
data_module:
  batch_size: 2        # 单GPU建议2-4
  num_workers: 4       # 根据CPU核心数调整
  base_dataset:
    data_path: database/simlingo_extracted
    bucket_path: database/bucketsv2_simlingo

gpus: 1                # 单GPU
max_epochs: 15
```

---

## 阶段四：开始训练

### 4.1 单GPU训练（本地）

```bash
# 激活环境
conda activate simlingo

# 设置工作目录（如果需要）
export WORK_DIR=/home/mh2803/projects/simlingo
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}

# 进入训练目录
cd simlingo_training

# 开始训练（使用debug配置）
python train.py experiment=debug

# 或使用自定义配置
python train.py experiment=my_training
```

### 4.2 多GPU训练（本地）

```bash
cd simlingo_training

# 使用simlingo_seed1配置（默认8 GPU）
python train.py experiment=simlingo_seed1 gpus=2  # 修改为你的GPU数量
```

### 4.3 SLURM集群训练

如果使用SLURM集群，可以修改 `train_simlingo_seed1.sh`：

```bash
# 编辑SLURM脚本
vim train_simlingo_seed1.sh

# 修改以下内容：
# - 第7-8行：输出路径
# - 第9行：SLURM分区名称
# - 第15行：conda环境路径
# - 第19行：WORK_DIR路径
# - 第25行：GPU数量

# 提交任务
sbatch train_simlingo_seed1.sh
```

### 4.4 训练命令参数覆盖

可以在命令行直接覆盖配置参数：

```bash
python train.py \
    experiment=simlingo_seed1 \
    data_module.batch_size=4 \
    gpus=1 \
    max_epochs=10 \
    name=my_experiment
```

---

## 阶段五：监控训练

### 5.1 Wandb 监控

训练日志会自动上传到 Wandb：
- 访问 https://wandb.ai
- 登录你的账号
- 查看项目 "simlingo"

### 5.2 本地日志

训练日志和检查点保存在：
```
outputs/
└── [wandb_name]_[name]/
    ├── checkpoints/        # 模型检查点
    └── .hydra/             # Hydra配置备份
```

### 5.3 检查点

模型检查点按epoch保存：
- `checkpoints/epoch=000.ckpt`
- `checkpoints/epoch=001.ckpt`
- ...
- `checkpoints/last.ckpt`（最新）

---

## 🔧 常见问题排查

### Q1: 数据集路径错误

**错误**：`FileNotFoundError` 或找不到数据文件

**解决**：
1. 检查 `data_path` 是否正确指向解压后的数据集
2. 验证数据集目录结构是否正确
3. 检查路径是绝对路径还是相对路径

```bash
# 验证路径
ls -la database/simlingo_extracted/
```

### Q2: Bucket文件找不到

**错误**：找不到bucket文件

**解决**：
1. 检查bucket文件是否在数据集中
2. 如果bucket文件单独下载，确保路径正确
3. 或者使用 `carla_no_buckets.yaml` 配置（不使用bucket采样）

```yaml
# 在experiment配置中修改
defaults:
  - /data_module: carla_no_buckets  # 不使用bucket
```

### Q3: GPU显存不足

**错误**：`CUDA out of memory`

**解决**：
1. 减小 `batch_size`（例如从6改为2）
2. 使用混合精度训练（默认已启用 `precision: 16-mixed`）
3. 减少 `num_workers`

### Q4: Wandb登录问题

**错误**：Wandb认证失败

**解决**：
```bash
# 重新登录
wandb login

# 或离线模式（不上传日志）
export WANDB_MODE=offline
```

### Q5: 数据加载慢

**解决**：
1. 增加 `num_workers`（但不要超过CPU核心数）
2. 确保数据集在SSD上（而不是HDD）
3. 检查磁盘I/O性能

---

## 📊 训练配置说明

### 数据集配置选项

```yaml
base_dataset:
  data_path: database/simlingo_extracted    # 数据集路径
  bucket_path: database/bucketsv2_simlingo # Bucket路径
  use_commentary: True                      # 使用Commentary数据
  use_qa: True                              # 使用VQA数据
  qa_augmentation: True                     # VQA数据增强
  commentary_augmentation: True              # Commentary数据增强
  use_safety_flag: True                     # 使用安全标志
  cut_bottom_quarter: True                  # 裁剪图像底部1/4
  pred_len: 11                              # 预测长度
  hist_len: 1                               # 历史长度
```

### 模型配置

```yaml
model:
  lr: 3e-5                                  # 学习率
  vision_model:
    variant: 'OpenGVLab/InternVL2-1B'       # 视觉模型
  language_model:
    variant: 'OpenGVLab/InternVL2-1B'       # 语言模型
    lora: True                               # 使用LoRA
```

---

## 🎯 快速开始（最小配置）

如果你想快速测试训练流程：

```bash
# 1. 下载数据集（至少部分数据用于测试）
cd database
git clone https://huggingface.co/datasets/RenzKa/simlingo
cd simlingo
git lfs pull  # 只下载部分文件用于测试

# 2. 解压测试数据
cd ../..
# 只解压一个小的tar.gz文件用于测试

# 3. 修改debug.yaml中的data_path

# 4. 开始小规模训练
cd simlingo_training
python train.py experiment=debug gpus=1 batch_size=1 max_epochs=1
```

---

## 📝 下一步

训练完成后，你可以：

1. **评估模型**：使用 `simlingo_training/eval.py` 评估语言能力
2. **Bench2Drive评估**：在CARLA中评估闭环驾驶性能（需要CARLA）
3. **继续训练**：从检查点恢复训练

---

## 🔗 相关资源

- **数据集**：https://huggingface.co/datasets/RenzKa/simlingo
- **模型**：https://huggingface.co/RenzKa/simlingo
- **Wandb**：https://wandb.ai
- **Hydra文档**：https://hydra.cc/

---

## ⚠️ 重要提示

1. **磁盘空间**：确保有至少 200 GB 可用空间
2. **网络**：数据集很大，需要稳定的网络连接
3. **时间**：完整训练可能需要数天（取决于GPU数量）
4. **检查点**：定期保存检查点，避免训练中断丢失进度

