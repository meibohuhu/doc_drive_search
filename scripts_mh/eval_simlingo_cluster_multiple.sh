#!/bin/bash -l
# NOTE the -l flag!
#
# SimLingo Evaluation on Cluster (SLURM)
# Evaluates SimLingo model on Bench2Drive benchmark

#SBATCH --job-name=simlingo_eval
#SBATCH --error=/home/mh2803/projects/simlingo/scripts/cluster_logs/eval_err_%j.txt
#SBATCH --output=/home/mh2803/projects/simlingo/scripts/cluster_logs/eval_out_%j.txt
#SBATCH --account=llm-gen-agent
#SBATCH --nodes=1
#SBATCH --time=00:20:00  # 30 minutes for testing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=tier3
#SBATCH --mem=64G

# Configuration - modify these variables as needed
ROUTE_FILE="${1:-/home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/bench2drive_changelane.xml}"
ROUTE_ID="${2:-bench2drive_multiple_changelane_command}"  # Route ID from filename (e.g., bench2drive_1.xml -> "1")
SEED="${3:-1}"
# /shared/rc/llm-gen-agent/mhu/pretrained_models/simlingo
CHECKPOINT="${4:-/shared/rc/llm-gen-agent/mhu/pretrained_models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
AGENT_FILE="/home/mh2803/projects/simlingo/team_code/agent_simlingo.py"
REPO_ROOT="/home/mh2803/projects/simlingo"
CARLA_ROOT="${CARLA_ROOT:-/home/mh2803/software/carla0915}"  # Adjust if CARLA is installed elsewhere
VIZ_PATH="${REPO_ROOT}/eval_results_rc/Bench2Drive/simlingo/bench2drive/${SEED}/viz/${ROUTE_ID}"
RESULT_FILE="${REPO_ROOT}/eval_results_rc/Bench2Drive/simlingo/bench2drive/${SEED}/res/${ROUTE_ID}_res.json"

# Ports for CARLA (adjust if needed, should be unique per job)
PORT="${5:-20002}"
TM_PORT="${6:-30002}"

echo "JOB ID $SLURM_JOB_ID"
echo "Route: ${ROUTE_FILE}"
echo "Route ID: ${ROUTE_ID}"
echo "Seed: ${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
echo "CARLA Port: ${PORT}"
echo "Traffic Manager Port: ${TM_PORT}"

# Load spack modules (adjust based on your cluster setup)
spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Try to load libjpeg-turbo which may provide compatible libjpeg.so.8
# This helps with CARLA's libjpeg dependency
spack load libjpeg-turbo@2.1.4 2>/dev/null || echo "libjpeg-turbo not available via spack"

# Set up environment paths
export PATH="/home/mh2803/miniconda3/envs/simlingo/bin:$PATH"
export CONDA_PREFIX="/home/mh2803/miniconda3/envs/simlingo"  # Set conda prefix explicitly
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Set up CARLA paths
export CARLA_ROOT="${CARLA_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/Bench2Drive/leaderboard"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/Bench2Drive/scenario_runner"
export SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"

# Set WORK_DIR for leaderboard evaluator (needed for weather.xml and other data files)
# WORK_DIR should point to the Bench2Drive directory (contains leaderboard folder)
export WORK_DIR="${REPO_ROOT}/Bench2Drive"

# Set up library paths for CARLA dependencies
# CARLA needs libjpeg.so.8, but system may have libjpeg.so.62
# Add system library paths and CARLA's own libraries
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CARLA_ROOT}/CarlaUE4/Binaries/Linux"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CARLA_ROOT}/Engine/Binaries/ThirdParty/PhysX3/Linux/x86_64-unknown-linux-gnu"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib64:/usr/lib"

# Handle libjpeg.so.8 dependency
# CARLA 0.9.15 requires libjpeg.so.8 with LIBJPEG_8.0 symbols
# Try multiple approaches to find compatible library
TEMP_LIB_DIR="${HOME}/.local/lib/carla_deps"
mkdir -p "${TEMP_LIB_DIR}"

# Try to find compatible libjpeg.so.8
# Option 1: Check spack-loaded libjpeg-turbo
SPACK_JPEG=$(spack location -i libjpeg-turbo@2.1.4 2>/dev/null || echo "")
if [ -n "${SPACK_JPEG}" ] && [ -f "${SPACK_JPEG}/lib/libjpeg.so.8" ]; then
    export LD_LIBRARY_PATH="${SPACK_JPEG}/lib:${LD_LIBRARY_PATH}"
    echo "Using spack libjpeg-turbo from ${SPACK_JPEG}/lib"
# Option 2: Check conda environment for libjpeg
elif [ -f "${CONDA_PREFIX}/lib/libjpeg.so.8" ] 2>/dev/null; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    echo "Using conda libjpeg from ${CONDA_PREFIX}/lib"
# Option 3: Try to install libjpeg-turbo via conda if available
elif command -v conda &> /dev/null && [ -n "${CONDA_PREFIX}" ]; then
    echo "Installing libjpeg-turbo via conda (this may take a moment)..."
    # Activate conda environment and install
    source /home/mh2803/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate simlingo 2>/dev/null || true
    conda install -y -c conda-forge libjpeg-turbo 2>&1 | tail -5
    if [ -f "${CONDA_PREFIX}/lib/libjpeg.so.8" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
        echo "Successfully installed and using conda libjpeg-turbo from ${CONDA_PREFIX}/lib"
    else
        echo "Warning: libjpeg-turbo installation may have failed"
        echo "Try manually: conda activate simlingo && conda install -c conda-forge libjpeg-turbo"
    fi
fi

# Clean up old symlink if it exists (it doesn't work with version symbols)
rm -f "${TEMP_LIB_DIR}/libjpeg.so.8" 2>/dev/null || true

# Verify libjpeg.so.8 is available
if [ -z "$(echo ${LD_LIBRARY_PATH} | tr ':' '\n' | xargs -I {} find {} -name 'libjpeg.so.8' 2>/dev/null | head -1)" ]; then
    echo "Warning: libjpeg.so.8 not found in LD_LIBRARY_PATH"
    echo "CARLA may fail to load. Install manually with:"
    echo "  conda activate simlingo"
    echo "  conda install -c conda-forge libjpeg-turbo"
fi

# Create output directories
mkdir -p "${VIZ_PATH}"
mkdir -p "$(dirname ${RESULT_FILE})"

export SAVE_PATH="${VIZ_PATH}"

# Change to project directory
cd "${REPO_ROOT}"

# Run evaluation
echo "Starting evaluation..."
python -u "${REPO_ROOT}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py" \
    --routes="${ROUTE_FILE}" \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint="${RESULT_FILE}" \
    --timeout=600 \
    --agent="${AGENT_FILE}" \
    --agent-config="${CHECKPOINT}" \
    --traffic-manager-seed="${SEED}" \
    --port="${PORT}" \
    --traffic-manager-port="${TM_PORT}"

echo ""
echo "✅ Evaluation completed!"
echo "Results saved to: ${RESULT_FILE}"
echo "Visualizations saved to: ${VIZ_PATH}"

