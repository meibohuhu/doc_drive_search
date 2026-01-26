#!/bin/bash -l
# NOTE the -l flag!
#
# SimLingo Evaluation on Cluster (SLURM) - Multi-Route Parallel Execution
# Evaluates SimLingo model on 4 routes simultaneously using 1 GPU
# Each route runs in parallel with its own CARLA server instance

#SBATCH --job-name=simlingo_eval_4routes
#SBATCH --error=/home/mh2803/projects/simlingo/scripts/cluster_logs/eval_4routes_err_%j.txt
#SBATCH --output=/home/mh2803/projects/simlingo/scripts/cluster_logs/eval_4routes_out_%j.txt
#SBATCH --account=llm-gen-agent
#SBATCH --nodes=1
#SBATCH --time=00:30:00  # 1 hour for 4 routes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32  # More CPUs for parallel execution
#SBATCH --partition=tier3
#SBATCH --mem=128G  # More memory for multiple CARLA instances

# Configuration - modify these variables as needed
# Route files to evaluate (provide 4 routes)
ROUTE_FILES=(
    "${1:-/home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/bench2drive_69.xml}"
    "${2:-/home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/bench2drive_2403.xml}"
    "${3:-/home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/bench2drive_1711.xml}"
    "${4:-/home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/bench2drive_28035.xml}"
)

# Extract route IDs from filenames
ROUTE_IDS=()
for route_file in "${ROUTE_FILES[@]}"; do
    route_id=$(basename "$route_file" | sed 's/bench2drive_//' | sed 's/\.xml//')
    ROUTE_IDS+=("$route_id")
done

SEED="${5:-1}"
CHECKPOINT="${6:-/shared/rc/llm-gen-agent/mhu/pretrained_models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
AGENT_FILE="/home/mh2803/projects/simlingo/team_code/agent_simlingo.py"
REPO_ROOT="/home/mh2803/projects/simlingo"
CARLA_ROOT="${CARLA_ROOT:-/home/mh2803/software/carla0915}"

# Port configuration - each route gets unique ports
PORT_BASE=20000
TM_PORT_BASE=30000
PORTS=()
TM_PORTS=()
for i in {0..3}; do
    PORTS+=($((PORT_BASE + i * 10)))
    TM_PORTS+=($((TM_PORT_BASE + i * 10)))
done

echo "=========================================="
echo "JOB ID: $SLURM_JOB_ID"
echo "Evaluating ${#ROUTE_FILES[@]} routes in parallel"
echo "Seed: ${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
echo "=========================================="
for i in "${!ROUTE_FILES[@]}"; do
    echo "Route $((i+1)): ${ROUTE_IDS[$i]} (Port: ${PORTS[$i]}, TM Port: ${TM_PORTS[$i]})"
done
echo "=========================================="

# Load spack modules
spack load /lhqcen5
spack load cuda@12.4.0/obxqih4
spack load libjpeg-turbo@2.1.4 2>/dev/null || echo "libjpeg-turbo not available via spack"

# Set up environment paths
export PATH="/home/mh2803/miniconda3/envs/simlingo/bin:$PATH"
export CONDA_PREFIX="/home/mh2803/miniconda3/envs/simlingo"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Set up CARLA paths
export CARLA_ROOT="${CARLA_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/Bench2Drive/leaderboard"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/Bench2Drive/scenario_runner"
export SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"
export WORK_DIR="${REPO_ROOT}/Bench2Drive"

# Set up library paths for CARLA dependencies
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CARLA_ROOT}/CarlaUE4/Binaries/Linux"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CARLA_ROOT}/Engine/Binaries/ThirdParty/PhysX3/Linux/x86_64-unknown-linux-gnu"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib64:/usr/lib"

# Handle libjpeg.so.8 dependency
SPACK_JPEG=$(spack location -i libjpeg-turbo@2.1.4 2>/dev/null || echo "")
if [ -n "${SPACK_JPEG}" ] && [ -f "${SPACK_JPEG}/lib/libjpeg.so.8" ]; then
    export LD_LIBRARY_PATH="${SPACK_JPEG}/lib:${LD_LIBRARY_PATH}"
    echo "Using spack libjpeg-turbo from ${SPACK_JPEG}/lib"
elif [ -f "${CONDA_PREFIX}/lib/libjpeg.so.8" ] 2>/dev/null; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    echo "Using conda libjpeg from ${CONDA_PREFIX}/lib"
fi

# Change to project directory
cd "${REPO_ROOT}"

# Function to start CARLA server for a route
start_carla_server() {
    local port=$1
    local route_id=$2
    local carla_log="/tmp/carla_server_${port}.log"
    
    echo "[Route ${route_id}] Starting CARLA server on port ${port}..."
    "${CARLA_ROOT}/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port=${port} -graphicsadapter=0 > "${carla_log}" 2>&1 &
    local carla_pid=$!
    echo "[Route ${route_id}] CARLA server PID: ${carla_pid}"
    
    # Wait for CARLA server to be ready
    echo "[Route ${route_id}] Waiting for CARLA server to be ready..."
    local max_wait=90
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if timeout 2 python -c "import carla; client = carla.Client('localhost', ${port}); client.set_timeout(2.0); client.get_world()" 2>/dev/null; then
            echo "[Route ${route_id}] CARLA server is ready!"
            echo "${carla_pid}"  # Return PID
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    
    echo "[Route ${route_id}] ⚠️  CARLA server failed to start within ${max_wait} seconds"
    echo "${carla_pid}"  # Return PID anyway for cleanup
    return 1
}

# Function to run evaluation for a single route
run_evaluation() {
    local route_file=$1
    local route_id=$2
    local port=$3
    local tm_port=$4
    local seed=$5
    local checkpoint=$6
    
    local viz_path="${REPO_ROOT}/eval_results_rc/Bench2Drive/simlingo/bench2drive/${seed}/viz/${route_id}"
    local result_file="${REPO_ROOT}/eval_results_rc/Bench2Drive/simlingo/bench2drive/${seed}/res/${route_id}_res.json"
    
    # Create output directories
    mkdir -p "${viz_path}"
    mkdir -p "$(dirname ${result_file})"
    
    export SAVE_PATH="${viz_path}"
    
    echo "[Route ${route_id}] Starting evaluation on port ${port}..."
    python -u "${REPO_ROOT}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py" \
        --routes="${route_file}" \
        --repetitions=1 \
        --track=SENSORS \
        --checkpoint="${result_file}" \
        --timeout=600 \
        --agent="${AGENT_FILE}" \
        --agent-config="${checkpoint}" \
        --traffic-manager-seed="${seed}" \
        --port="${port}" \
        --traffic-manager-port="${tm_port}" \
        > "/tmp/eval_${route_id}.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[Route ${route_id}] ✅ Evaluation completed successfully"
    else
        echo "[Route ${route_id}] ❌ Evaluation failed with exit code ${exit_code}"
        echo "[Route ${route_id}] Check log: /tmp/eval_${route_id}.log"
    fi
    
    return $exit_code
}

# Start all CARLA servers
echo ""
echo "=========================================="
echo "Starting CARLA servers..."
echo "=========================================="
CARLA_PIDS=()
for i in "${!ROUTE_FILES[@]}"; do
    carla_pid=$(start_carla_server "${PORTS[$i]}" "${ROUTE_IDS[$i]}")
    CARLA_PIDS+=("$carla_pid")
    sleep 2  # Stagger server starts
done

# Wait a bit for all servers to stabilize
sleep 5

# Run evaluations in parallel using background processes
echo ""
echo "=========================================="
echo "Starting parallel evaluations..."
echo "=========================================="
EVAL_PIDS=()
for i in "${!ROUTE_FILES[@]}"; do
    (
        run_evaluation \
            "${ROUTE_FILES[$i]}" \
            "${ROUTE_IDS[$i]}" \
            "${PORTS[$i]}" \
            "${TM_PORTS[$i]}" \
            "${SEED}" \
            "${CHECKPOINT}"
    ) &
    eval_pid=$!
    EVAL_PIDS+=("$eval_pid")
    echo "[Route ${ROUTE_IDS[$i]}] Evaluation started with PID ${eval_pid}"
done

# Wait for all evaluations to complete
echo ""
echo "=========================================="
echo "Waiting for all evaluations to complete..."
echo "=========================================="
EXIT_CODES=()
for i in "${!EVAL_PIDS[@]}"; do
    pid="${EVAL_PIDS[$i]}"
    route_id="${ROUTE_IDS[$i]}"
    wait $pid
    exit_code=$?
    EXIT_CODES+=("$exit_code")
    if [ $exit_code -eq 0 ]; then
        echo "[Route ${route_id}] ✅ Completed successfully"
    else
        echo "[Route ${route_id}] ❌ Failed with exit code ${exit_code}"
    fi
done

# Cleanup: Kill all CARLA servers
echo ""
echo "=========================================="
echo "Cleaning up CARLA servers..."
echo "=========================================="
for i in "${!CARLA_PIDS[@]}"; do
    pid="${CARLA_PIDS[$i]}"
    route_id="${ROUTE_IDS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
        echo "[Route ${route_id}] Stopping CARLA server (PID: ${pid})..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
done

# Summary
echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
TOTAL_SUCCESS=0
TOTAL_FAILED=0
for i in "${!ROUTE_IDS[@]}"; do
    route_id="${ROUTE_IDS[$i]}"
    exit_code="${EXIT_CODES[$i]}"
    if [ $exit_code -eq 0 ]; then
        echo "✅ Route ${route_id}: SUCCESS"
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
        echo "❌ Route ${route_id}: FAILED (exit code: ${exit_code})"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
    echo "   Results: ${REPO_ROOT}/eval_results_rc/Bench2Drive/simlingo/bench2drive/${SEED}/res/${route_id}_res.json"
    echo "   Logs: /tmp/eval_${route_id}.log"
done

echo ""
echo "Total: ${TOTAL_SUCCESS} succeeded, ${TOTAL_FAILED} failed"
if [ $TOTAL_FAILED -eq 0 ]; then
    echo "✅ All evaluations completed successfully!"
    exit 0
else
    echo "⚠️  Some evaluations failed. Check logs for details."
    exit 1
fi

