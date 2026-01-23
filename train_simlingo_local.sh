#!/bin/bash
# 本地训练脚本（不使用 SLURM）
# 用于训练小规模数据集

# 设置工作目录
export WORK_DIR=/local1/mhu/doc_drive_search
cd ${WORK_DIR}

# 激活 conda 环境
source ~/.bashrc
conda activate simlingo  # 根据你的环境名称修改

# 设置 Python 路径
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH}"
if [ -n "${CARLA_ROOT}" ]; then
    export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${PYTHONPATH}"
fi

# 设置环境变量
export OMP_NUM_THREADS=8  # 根据你的 CPU 核心数调整
export OPENBLAS_NUM_THREADS=1

# 设置 Hugging Face token（如果需要访问 gated 模型）
# 如果你有 Hugging Face token，取消下面的注释并设置你的 token
# export HF_TOKEN=your_huggingface_token_here
# 或者使用 huggingface-cli login 命令登录

# 检查是否设置了 HF_TOKEN
if [ -z "${HF_TOKEN}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "警告: 未检测到 HF_TOKEN 或 HUGGING_FACE_HUB_TOKEN 环境变量"
    echo "如果遇到 gated repo 错误，请："
    echo "1. 申请 Llama 模型访问权限: https://huggingface.co/meta-llama/Llama-2-7b-hf"
    echo "2. 设置: export HF_TOKEN=your_token"
    echo "3. 或使用: huggingface-cli login"
fi

# 训练参数
# 默认使用 local_training 配置，可以通过命令行参数覆盖
# 例如：bash train_simlingo_local.sh experiment=local_training gpus=1 batch_size=4
TRAIN_ARGS="experiment=local_training"

# 如果提供了命令行参数，使用命令行参数
if [ $# -gt 0 ]; then
    TRAIN_ARGS="$@"
fi
# 设置 wandb API key（新格式支持 wandb_v1_ 开头的长 key）
export WANDB_API_KEY=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


echo "=========================================="
echo "开始本地训练"
echo "=========================================="
echo "工作目录: ${WORK_DIR}"
echo "数据集路径: data/simlingo_dataset/database/simlingo_extracted"
echo "Bucket路径: data/simlingo_dataset/database/bucketsv2_simlingo"
echo "训练参数: ${TRAIN_ARGS}"
echo ""
echo "使用说明："
echo "1. 默认配置：bash train_simlingo_local.sh"
echo "2. 自定义参数：bash train_simlingo_local.sh experiment=local_training gpus=1 data_module.batch_size=4"
echo "3. 使用debug配置：bash train_simlingo_local.sh experiment=debug"
echo ""
echo "注意：使用 simlingo_base_training 进行训练"
echo "=========================================="

# 运行训练
cd simlingo_base_training
python train.py ${TRAIN_ARGS}

echo "训练完成！"

