#!/bin/bash
# 训练 simlingo_training（包含语言模型的完整版本）

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

# 设置 wandb API key（新格式支持 wandb_v1_ 开头的长 key）
export WANDB_API_KEY=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

# 指定GPU: CUDA_VISIBLE_DEVICES=0 bash train_simlingo_full_local.sh
# 或通过参数: bash train_simlingo_full_local.sh gpus=1

TRAIN_ARGS="${@:-experiment=local_training}"

cd simlingo_training
python train.py ${TRAIN_ARGS}

