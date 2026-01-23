# 本地训练指南

## 概述

`train_simlingo_local.sh` 是一个不使用 SLURM 的本地训练脚本，用于训练小规模数据集。

**注意**：此脚本使用 `simlingo_base_training` 进行训练（而不是 `simlingo_training`）。

## 数据集路径

脚本默认使用以下数据集路径（相对于项目根目录）：
- **数据路径**: `data/simlingo_dataset/database/simlingo_extracted`
- **Bucket路径**: `data/simlingo_dataset/database/bucketsv2_simlingo`

## 使用方法

### 1. 基本使用（使用默认配置）

```bash
bash train_simlingo_local.sh
```

这将使用 `simlingo_base_training/config/experiment/local_training.yaml` 配置文件。

### 2. 使用自定义参数

```bash
# 修改GPU数量
bash train_simlingo_local.sh experiment=local_training gpus=1

# 修改batch size
bash train_simlingo_local.sh experiment=local_training data_module.batch_size=4

# 修改多个参数
bash train_simlingo_local.sh experiment=local_training gpus=1 data_module.batch_size=4 data_module.num_workers=8
```

### 3. 使用debug配置（更小的batch size）

```bash
bash train_simlingo_local.sh experiment=debug
```

### 4. 完全自定义数据集路径

```bash
bash train_simlingo_local.sh \
    experiment=local_training \
    data_module.base_dataset.data_path=your/custom/path \
    data_module.base_dataset.bucket_path=your/custom/bucket/path
```

## 配置说明

### 主要参数

- **gpus**: GPU数量（默认：1）
- **batch_size**: 批次大小（默认：2，根据GPU显存调整）
- **num_workers**: 数据加载线程数（默认：4，根据CPU核心数调整）
- **max_epochs**: 训练轮数（默认：15）

### GPU显存建议

- **8GB显存**: `batch_size=2`
- **16GB显存**: `batch_size=4-6`
- **24GB+显存**: `batch_size=8+`

### CPU线程数建议

- **4核CPU**: `num_workers=2-4`
- **8核CPU**: `num_workers=4-8`
- **16核+CPU**: `num_workers=8-16`

## 修改配置

如果需要修改默认配置，可以编辑：
- `simlingo_base_training/config/experiment/local_training.yaml`

主要修改项：
```yaml
data_module:
  batch_size: 2  # 修改batch size
  num_workers: 4  # 修改线程数
  base_dataset:
    data_path: data/simlingo_dataset/database/simlingo_extracted  # 修改数据路径
    bucket_path: data/simlingo_dataset/database/bucketsv2_simlingo  # 修改bucket路径
```

## 环境要求

1. **Conda环境**: 确保已激活 `simlingo` 环境
2. **数据集**: 确保数据集已下载并解压到指定路径
3. **GPU**: 建议使用至少8GB显存的GPU

## 训练输出

训练输出将保存在：
- `outputs/${wandb_name}_${name}/`

包含：
- 模型checkpoints
- 训练日志
- TensorBoard日志（如果启用）

## 故障排除

### 1. 显存不足（OOM）

减小batch size：
```bash
bash train_simlingo_local.sh data_module.batch_size=1
```

### 2. 数据加载慢

增加num_workers：
```bash
bash train_simlingo_local.sh data_module.num_workers=8
```

### 3. 找不到数据集

检查数据集路径是否正确：
```bash
ls data/simlingo_dataset/database/simlingo_extracted
ls data/simlingo_dataset/database/bucketsv2_simlingo
```

### 4. Conda环境问题

确保环境已正确激活：
```bash
conda activate simlingo
which python  # 应该指向simlingo环境中的python
```

## 示例命令

```bash
# 单GPU，小batch size训练
bash train_simlingo_local.sh experiment=local_training gpus=1 data_module.batch_size=2

# 多GPU训练（如果有多个GPU）
bash train_simlingo_local.sh experiment=local_training gpus=2 data_module.batch_size=4

# 快速测试（使用debug配置）
bash train_simlingo_local.sh experiment=debug max_epochs=1
```

