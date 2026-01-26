#!/bin/bash -l
# NOTE the -l flag!
#
# SimLingo Training on Cluster (SLURM)
# Based on local_training.yaml configuration

#SBATCH --job-name=simlingo_training
#SBATCH --error=/home/mh2803/projects/simlingo/scripts/cluster_logs/err_%j.txt
#SBATCH --output=/home/mh2803/projects/simlingo/scripts/cluster_logs/out_%j.txt
#SBATCH --account=llm-gen-agent
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=64G

# Load spack modules (adjust based on your cluster setup)
spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths
export PATH="/home/mh2803/miniconda3/envs/simlingo/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/simlingo:${PYTHONPATH}"
if [ -n "${CARLA_ROOT}" ]; then
    export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${PYTHONPATH}"
fi

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1

# Set Hugging Face token (if needed for gated models)
# export HF_TOKEN=your_huggingface_token_here
# or use: huggingface-cli login

# Check if HF_TOKEN is set
if [ -z "${HF_TOKEN}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "警告: 未检测到 HF_TOKEN 或 HUGGING_FACE_HUB_TOKEN 环境变量"
    echo "如果遇到 gated repo 错误，请设置: export HF_TOKEN=your_token"
fi

# Set Wandb API key
export WANDB_API_KEY=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

# Change to project directory
cd /home/mh2803/projects/simlingo

# Training configuration
# Default uses cluster_training.yaml config
# Can override via command line: sbatch train_simlingo_cluster.sh experiment=cluster_training gpus=1 data_module.batch_size=8
TRAIN_ARGS="experiment=cluster_training"

# If command line arguments provided, use them
if [ $# -gt 0 ]; then
    TRAIN_ARGS="$@"
fi

echo "=========================================="
echo "🚀 Starting SimLingo Training on Cluster"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Working Directory: $(pwd)"
echo "Dataset Path: /shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo_extracted_mini"
echo "Bucket Path: /shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/bucketsv2_simlingo"
echo "Training Args: ${TRAIN_ARGS}"
echo ""

# Run training
cd simlingo_base_training
python train.py ${TRAIN_ARGS}

echo ""
echo "✅ Training completed!"

