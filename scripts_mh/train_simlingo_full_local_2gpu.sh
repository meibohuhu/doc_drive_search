#!/bin/bash
# 训练 simlingo_training（包含语言模型的完整版本）- 双GPU版本

export WORK_DIR=/local1/mhu/doc_drive_search
cd ${WORK_DIR}

source ~/.bashrc
# 初始化 conda（如果还没有初始化）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate simlingo

export PYTHONPATH="${WORK_DIR}:${PYTHONPATH}"
[ -n "${CARLA_ROOT}" ] && export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${PYTHONPATH}"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

# 设置分布式训练环境变量（用于多GPU训练）
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
# 设置 NCCL 超时时间以避免挂起（单位：秒）
export NCCL_TIMEOUT=1800

# 设置 wandb API key（新格式支持 wandb_v1_ 开头的长 key）
export WANDB_API_KEY=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

# 指定GPU: CUDA_VISIBLE_DEVICES=0,1 bash train_simlingo_full_local_2gpu.sh
# 或通过参数: bash train_simlingo_full_local_2gpu.sh experiment=local_training gpus=2
# 默认使用2个GPU
TRAIN_ARGS="${@:-experiment=local_training gpus=2}"

cd simlingo_training
python train.py ${TRAIN_ARGS}

echo ""
echo "✅ Training completed!"

