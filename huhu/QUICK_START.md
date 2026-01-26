# 快速开始指南（跳过 flash-attn）

## ✅ 简化安装步骤

如果你遇到 flash-attn 安装问题，可以安全跳过它。训练仍然可以正常运行！

### 1. 创建环境并安装基础依赖

```bash
cd /home/mh2803/projects/simlingo

# 创建conda环境（使用简化版配置文件）
conda env create -f environment_simplified.yaml
conda activate simlingo

# 注意：PyTorch 2.2.0 已包含在 environment_simplified.yaml 中，无需单独安装

# ⚠️ 跳过 flash-attn（可选，用于加速但不是必需的）
# 训练可以正常运行，只是速度会稍慢一些
```


### 3. 配置 Wandb

```bash
wandb login
# 输入你的API key（如果没有账号，去 https://wandb.ai 注册）
```

### 4. 下载数据集

#### 方法1：使用下载脚本（推荐，更快）

```bash
# 下载训练数据（只下载训练文件，不包括验证数据）
bash download_training_data.sh

# 解压数据（如果目录已有内容会自动跳过）
bash extract_training_data.sh
```

脚本功能：
- `download_training_data.sh`: 下载训练数据文件（~545 GB压缩）
  - 激活simlingo conda环境
  - 使用huggingface-cli下载（比git lfs更快）
  - 文件保存到：`/shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo`
- `extract_training_data.sh`: 解压数据文件
  - 自动检测目标目录是否已有内容，如有则跳过
  - 解压到：`/shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo_extracted`

#### 方法2：手动下载（如果脚本失败）

```bash
# 激活环境
conda activate simlingo

# 安装huggingface-cli（如果未安装）
pip install -U "huggingface_hub[cli]"

# 创建数据集目录
mkdir -p /shared/rc/llm-gen-agent/mhu/simlingo_dataset/database
cd /shared/rc/llm-gen-agent/mhu/simlingo_dataset/database

# 使用huggingface-cli下载（推荐，更快）
huggingface-cli download RenzKa/simlingo \
    --repo-type dataset \
    --local-dir simlingo \
    --local-dir-use-symlinks False

# 解压数据（使用解压脚本，如果目录已有内容会自动跳过）
bash extract_training_data.sh

# 或者手动解压：
# mkdir -p simlingo_extracted
# cd simlingo
# for file in *.tar.gz; do
#     echo "解压 $file ..."
#     tar -xzf "$file" -C ../simlingo_extracted/
# done
```

### 5. 配置训练路径

编辑 `simlingo_training/config/experiment/simlingo_seed1.yaml`：

```yaml
data_module:
  base_dataset:
    data_path: /shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo_extracted  # 修改为你的路径
    bucket_path: /shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/bucketsv2_simlingo  # 修改为你的bucket路径
```

**注意**：如果bucket文件在数据集中，路径可能需要调整。

### 6. 开始训练

```bash
cd simlingo_training

# 单GPU训练
python train.py experiment=simlingo_seed1 gpus=1 batch_size=2

# 多GPU训练
python train.py experiment=simlingo_seed1 gpus=2 batch_size=4
```

---

## 📝 关于 flash-attn

**flash-attn 是什么？**
- 一个优化的 attention 实现，可以加速 transformer 模型的训练
- 主要用于大模型训练，可以显著提升速度

**为什么可以跳过？**
- flash-attn 不是训练代码的硬依赖
- 如果没有 flash-attn，PyTorch 会使用标准的 attention 实现
- 训练仍然可以正常运行，只是速度会稍慢一些（通常慢 10-30%）

**什么时候需要安装？**
- 如果你有 CUDA 开发环境（nvcc、CUDA_HOME）
- 如果你想要最快的训练速度
- 如果你训练非常大的模型

**如何后续安装？**
如果之后想安装 flash-attn，可以：
1. 运行 `bash install_flash_attn.sh`（自动检测CUDA环境）
2. 或参考 `DATASET_TRAINING_GUIDE.md` 中的 Q1 部分

---

## ⚠️ 重要提示

1. **跳过 flash-attn 是安全的**：训练代码不依赖它
2. **训练速度**：可能会慢 10-30%，但仍然可以正常训练
3. **内存使用**：不使用 flash-attn 可能会使用更多内存
4. **批次大小**：如果内存不足，可以减小 `batch_size`

---

## 🎯 下一步

训练开始后，你可以：
- 在 Wandb 上查看训练进度
- 检查点保存在 `outputs/[experiment_name]/checkpoints/`
- 训练完成后可以评估模型性能


