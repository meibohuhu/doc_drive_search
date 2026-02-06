#!/bin/bash
# 训练 simlingo_training（包含语言模型的完整版本）

export WORK_DIR=/code/doc_drive_search
cd ${WORK_DIR}

# source ~/.bashrc
# # 初始化 conda（如果还没有初始化）
# if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/anaconda3/etc/profile.d/conda.sh"
# elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/miniconda3/etc/profile.d/conda.sh"
# fi
# conda activate simlingo

export PYTHONPATH="${WORK_DIR}:${PYTHONPATH}"
[ -n "${CARLA_ROOT}" ] && export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${PYTHONPATH}"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 训练参数（默认在 GPU 配置前解析，以支持 gpus= 参数）
# 默认使用 local_training 配置，可以通过命令行参数覆盖
# 例如：bash train_simlingo_full_local.sh experiment=local_training gpus=1 batch_size=4
TRAIN_ARGS="${@:-experiment=adobe_training_full_targetpoint_command}"

# GPU 配置：支持三种方式优先级（从高到低）:
# 1. 环境变量 CUDA_VISIBLE_DEVICES 指定具体索引列表 (例如 export CUDA_VISIBLE_DEVICES="0,2")
# 2. 在 TRAIN_ARGS 中传入 gpus=0,2 或 gpus=2 (列表或数量)
# 3. 默认使用 GPU 0

# 先尝试从 TRAIN_ARGS 中解析 gpus= 参数（支持数字或逗号分隔列表）
GPU_IDS=""
if echo "${TRAIN_ARGS}" | grep -q -E 'gpus=[0-9,]+'; then
    GPU_PARAM=$(echo "${TRAIN_ARGS}" | grep -o -E 'gpus=[0-9,]+' | head -n1 | cut -d'=' -f2)
    if echo "${GPU_PARAM}" | grep -q ','; then
        GPU_IDS="${GPU_PARAM}"
        NUM_DEVICES=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
    else
        GPU_COUNT=${GPU_PARAM}
        if [ -n "${GPU_COUNT}" ] && [ "${GPU_COUNT}" -gt 0 ] 2>/dev/null; then
            GPU_IDS=$(seq -s, 0 $((GPU_COUNT-1)))
            NUM_DEVICES=${GPU_COUNT}
        fi
    fi
    if [ -n "${NUM_DEVICES}" ]; then
        TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed -E "s/gpus=[0-9,]+/gpus=${NUM_DEVICES}/")
    fi
fi

# 如果外部环境变量 CUDA_VISIBLE_DEVICES 被设置，则用它覆盖（支持 "0,2" 列表）
GPU_LIST_ENV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [ -n "${GPU_LIST_ENV}" ]; then
    GPU_IDS="${GPU_LIST_ENV}"
    NUM_DEVICES=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
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

# 设置 wandb API key（新格式支持 wandb_v1_ 开头的长 key）
export WANDB_API_KEY=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
export MASTER_PORT=29510
echo "=========================================="
echo "开始本地训练 (full)"
echo "=========================================="
echo "工作目录: ${WORK_DIR}"
echo "训练参数: ${TRAIN_ARGS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Number of devices: ${NUM_DEVICES}"
echo "=========================================="

cd simlingo_training
python train.py ${TRAIN_ARGS}

