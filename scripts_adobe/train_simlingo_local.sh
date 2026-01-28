#!/bin/bash
# 本地训练脚本（不使用 SLURM）
# 用于训练小规模数据集 bash train_simlingo_local.sh experiment=adobe_training gpus=0,1,2,3

# 设置工作目录
export WORK_DIR=/code/doc_drive_search
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
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 训练参数（默认在 GPU 配置前解析，以支持 gpus= 参数）
# 默认使用 adobe_training 配置，可以通过命令行参数覆盖
# 例如：bash train_simlingo_local.sh experiment=adobe_training gpus=1 batch_size=4
TRAIN_ARGS="experiment=adobe_training"

# 如果提供了命令行参数，使用命令行参数
if [ $# -gt 0 ]; then
    TRAIN_ARGS="$@"
fi

# GPU 配置：支持三种方式优先级（从高到低）:
# 1. 将环境变量 GPU_IDS 设置为具体索引列表 (例如 export GPU_IDS="0,2")
# 2. 在 TRAIN_ARGS 中传入 gpus=0,2 或 gpus=2 (列表或数量)
# 3. 默认使用 GPU 0

# 先尝试从 TRAIN_ARGS 中解析 gpus= 参数（支持数字或逗号分隔列表）
GPU_IDS=""
if echo "${TRAIN_ARGS}" | grep -q -E 'gpus=[0-9,]+'; then
    GPU_PARAM=$(echo "${TRAIN_ARGS}" | grep -o -E 'gpus=[0-9,]+' | head -n1 | cut -d'=' -f2)
    if echo "${GPU_PARAM}" | grep -q ','; then
        # 用户提供了显式索引列表，例如 gpus=0,2,3
        GPU_IDS="${GPU_PARAM}"
        NUM_DEVICES=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
    else
        # 用户提供了数字，例如 gpus=2 -> 使用前 N 个 GPU (0..N-1)
        GPU_COUNT=${GPU_PARAM}
        if [ -n "${GPU_COUNT}" ] && [ "${GPU_COUNT}" -gt 0 ] 2>/dev/null; then
            GPU_IDS=$(seq -s, 0 $((GPU_COUNT-1)))
            NUM_DEVICES=${GPU_COUNT}
        fi
    fi
    # 确保 TRAIN_ARGS 中的 gpus= 设置为设备数量（PyTorch-Lightning 期望整数）
    if [ -n "${NUM_DEVICES}" ]; then
        TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed -E "s/gpus=[0-9,]+/gpus=${NUM_DEVICES}/")
    fi
fi

# 如果外部环境变量 CUDA_VISIBLE_DEVICES 被设置，则用它覆盖（支持 "0,2" 列表）
GPU_LIST_ENV=${CUDA_VISIBLE_DEVICES:-}
if [ -n "${GPU_LIST_ENV}" ]; then
    GPU_IDS="${GPU_LIST_ENV}"
    NUM_DEVICES=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
    # 同步 TRAIN_ARGS 中的 gpus 数量
    TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed -E "s/gpus=[0-9,]+/gpus=${NUM_DEVICES}/")
fi

# 最后回退到单 GPU (0) 当没有任何设置时
if [ -z "${GPU_IDS}" ]; then
    GPU_IDS="0"
    NUM_DEVICES=1
    TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed -E "s/gpus=[0-9,]+/gpus=${NUM_DEVICES}/")
fi

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
if [ -z "${NUM_DEVICES}" ]; then
    NUM_DEVICES=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
fi

# 如果 TRAIN_ARGS 中没有指定 gpus=，追加一个以覆盖配置文件中的默认值
if ! echo "${TRAIN_ARGS}" | grep -q -E '(^|\s)gpus='; then
    TRAIN_ARGS="${TRAIN_ARGS} gpus=${NUM_DEVICES}"
fi

# Launcher and master port for potential distributed runs
export LAUNCHER=${LAUNCHER:-pytorch}
export MASTER_PORT=${MASTER_PORT:-29500}

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


# 设置 wandb API key（新格式支持 wandb_v1_ 开头的长 key）
export WANDB_API_KEY=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


echo "=========================================="
echo "开始本地训练"
echo "=========================================="
echo "工作目录: ${WORK_DIR}"
echo "数据集路径: data/simlingo_dataset/database/simlingo_extracted"
echo "Bucket路径: data/simlingo_dataset/database/bucketsv2_simlingo"
echo "训练参数: ${TRAIN_ARGS}"
echo "GPU_IDS: ${GPU_IDS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Number of devices: ${NUM_DEVICES}"
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

