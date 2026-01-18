# 快速开始指南（跳过 flash-attn）

## ✅ 简化安装步骤

如果你遇到 flash-attn 安装问题，可以安全跳过它。训练仍然可以正常运行！

### 1. 创建环境并安装基础依赖

```bash
cd /home/mh2803/projects/simlingo

# 创建conda环境
conda env create -f environment.yaml
conda activate simlingo

# 安装PyTorch
pip install torch==2.2.0

# ⚠️ 跳过 flash-attn（可选，用于加速但不是必需的）
# 训练可以正常运行，只是速度会稍慢一些
```

### 2. 安装 Git LFS

```bash
# Ubuntu/Debian
sudo apt install git-lfs
git lfs install
```

### 3. 配置 Wandb

```bash
wandb login
# 输入你的API key（如果没有账号，去 https://wandb.ai 注册）
```

### 4. 下载数据集

```bash
# 创建数据集目录
mkdir -p database
cd database

# 克隆数据集
git clone https://huggingface.co/datasets/RenzKa/simlingo
cd simlingo

# 下载数据（这需要较长时间）
git lfs pull

# 解压数据
cd ../..
mkdir -p database/simlingo_extracted
cd database/simlingo
for file in *.tar.gz; do
    echo "解压 $file ..."
    tar -xzf "$file" -C ../simlingo_extracted/
done
cd ../..
```

### 5. 配置训练路径

编辑 `simlingo_training/config/experiment/simlingo_seed1.yaml`：

```yaml
data_module:
  base_dataset:
    data_path: database/simlingo_extracted  # 修改为你的路径
    bucket_path: database/bucketsv2_simlingo  # 修改为你的bucket路径
```

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


