#!/bin/bash -l
# NOTE the -l flag!
#
# SimLingo Training on Cluster (SLURM)
# Based on local_training.yaml configuration

#SBATCH --job-name=simlingo_training
#SBATCH --error=/home/mh2803/projects/simlingo/scripts/cluster_logs/err_%j.txt
#SBATCH --output=/home/mh2803/projects/simlingo/scripts/cluster_logs/out_%j.txt
#SBATCH --account=llm-gen-agent
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --partition tier3
#SBATCH --mem=128G

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

# Set up distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
# Set NCCL timeout to avoid hanging (in seconds)
export NCCL_TIMEOUT=1800

# Set Wandb API key
export WANDB_API_KEY=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

# Change to project directory
cd /home/mh2803/projects/simlingo

# Training configuration
# Default uses cluster_training.yaml config with 2 GPUs
# Use ddp strategy instead of deepspeed_stage_2 for better multi-GPU stability
# Can override via command line: sbatch train_simlingo_cluster_2gpu.sh experiment=cluster_training gpus=2 data_module.batch_size=16
TRAIN_ARGS="experiment=cluster_training gpus=2 strategy=ddp"

# If command line arguments provided, use them
if [ $# -gt 0 ]; then
    TRAIN_ARGS="$@"
fi

# Run training
# PyTorch Lightning will automatically handle multi-GPU training
cd simlingo_base_training
python train.py ${TRAIN_ARGS}

echo ""
echo "✅ Training completed!"

