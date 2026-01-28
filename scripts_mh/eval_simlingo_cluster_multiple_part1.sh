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
#SBATCH --time=03:30:00  # 30 minutes for testing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=tier3
#SBATCH --mem=64G

# Configuration - modify these variables as needed
ROUTE_FILE="${1:-/home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/bench2drive220_part1.xml}"
ROUTE_ID="${2:-bench2drive220_part1}"  # Route ID from filename (e.g., bench2drive_1.xml -> "1")
SEED="${3:-1}"
# /shared/rc/llm-gen-agent/mhu/pretrained_models/simlingo
CHECKPOINT="${4:-/shared/rc/llm-gen-agent/mhu/pretrained_models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
AGENT_FILE="/home/mh2803/projects/simlingo/team_code/agent_simlingo.py"
REPO_ROOT="/home/mh2803/projects/simlingo"
CARLA_ROOT="${CARLA_ROOT:-/home/mh2803/software/carla0915}"  # Adjust if CARLA is installed elsewhere
VIZ_PATH="${REPO_ROOT}/eval_results_rc/Bench2Drive/simlingo/bench2drive/${SEED}/viz/${ROUTE_ID}"
RESULT_FILE="${REPO_ROOT}/eval_results_rc/Bench2Drive/simlingo/bench2drive/${SEED}/res/${ROUTE_ID}_res.json"

# Ports for CARLA (adjust if needed, should be unique per job)
PORT="${5:-20000}"
TM_PORT="${6:-30000}"

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

# Auto-restart configuration
MAX_RESTARTS="${7:-3}"  # Maximum number of restarts (default: 10)
RESTART_DELAY="${8:-30}"  # Delay in seconds before restarting (default: 30)
RESTART_COUNT=0
USE_RESUME=false

# Function to check if CARLA server is running
check_carla_server() {
    # Check if there's a process listening on the CARLA port
    if command -v netstat &> /dev/null; then
        netstat -tuln 2>/dev/null | grep -q ":${PORT} " && return 0
    elif command -v ss &> /dev/null; then
        ss -tuln 2>/dev/null | grep -q ":${PORT} " && return 0
    fi
    # Fallback: check for CARLA processes
    ps aux | grep -i "CarlaUE4" | grep -v grep &>/dev/null && return 0
    return 1
}

# Function to kill any existing CARLA processes on this port
cleanup_carla() {
    echo "Cleaning up CARLA processes..."
    # Kill processes using the port
    if command -v lsof &> /dev/null; then
        lsof -ti:${PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    fi
    # Kill CARLA processes (more aggressive cleanup)
    pkill -9 -f "CarlaUE4.*-graphicsadapter" 2>/dev/null || true
    pkill -9 -f "CarlaUE4-Linux" 2>/dev/null || true
    sleep 2
}

# Function to monitor evaluation and detect hangs
# This runs in background and kills the evaluation if it hangs
monitor_evaluation() {
    local eval_pid=$1
    local log_file=$2
    local max_silence_time=${3:-600}  # 15 minutes of silence = hung (default)
    local last_update_time=$(date +%s)
    local last_size=0
    
    echo "[Monitor] Starting evaluation monitor (PID: ${eval_pid}, max silence: ${max_silence_time}s)"
    
    while kill -0 ${eval_pid} 2>/dev/null; do
        sleep 30  # Check every 30 seconds
        
        # Check if log file exists and has grown
        if [ -f "${log_file}" ]; then
            current_size=$(stat -c%s "${log_file}" 2>/dev/null || echo 0)
            if [ ${current_size} -gt ${last_size} ]; then
                last_update_time=$(date +%s)
                last_size=${current_size}
            fi
        fi
        
        # Check for timeout errors in log
        if [ -f "${log_file}" ]; then
            if grep -q "time-out of.*while waiting for the simulator" "${log_file}" 2>/dev/null || \
               grep -q "Watchdog exception.*Timeout" "${log_file}" 2>/dev/null || \
               grep -q "RuntimeError.*time-out" "${log_file}" 2>/dev/null || \
               grep -q "The simulation took longer than.*to update" "${log_file}" 2>/dev/null; then
                echo "[Monitor] ⚠️  Detected timeout/hang error in log, killing evaluation process..."
                kill -9 ${eval_pid} 2>/dev/null || true
                # Also kill CARLA
                cleanup_carla
                return 1
            fi
        fi
        
        # Check if process has been silent too long
        current_time=$(date +%s)
        silence_duration=$((current_time - last_update_time))
        if [ ${silence_duration} -gt ${max_silence_time} ]; then
            echo "[Monitor] ⚠️  Evaluation has been silent for ${silence_duration}s (>${max_silence_time}s), killing process..."
            kill -9 ${eval_pid} 2>/dev/null || true
            # Also kill CARLA
            cleanup_carla
            return 1
        fi
    done
    
    # Process exited normally
    return 0
}

# Function to run evaluation with timeout and monitoring
run_evaluation() {
    # Create log file for monitoring
    EVAL_LOG="${REPO_ROOT}/scripts/cluster_logs/eval_${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "${EVAL_LOG}")"
    
    # Maximum runtime for evaluation (2 hours = 7200 seconds)
    # Adjust based on your route count and expected time per route
    MAX_RUNTIME="${MAX_RUNTIME:-43200}"
    
    echo "📝 Evaluation log: ${EVAL_LOG}"
    echo "⏱️  Maximum runtime: ${MAX_RUNTIME}s ($((${MAX_RUNTIME}/60)) minutes)"
    
    # Run evaluation with timeout and log output
    # Use timeout command if available, otherwise rely on monitor
    if command -v timeout &> /dev/null; then
        TIMEOUT_CMD="timeout ${MAX_RUNTIME}"
    else
        TIMEOUT_CMD=""
        echo "⚠️  Warning: timeout command not available, relying on monitor only"
    fi
    
    if [ "$USE_RESUME" = true ]; then
        echo "🔄 Resuming from checkpoint: ${RESULT_FILE}"
        ${TIMEOUT_CMD} python -u "${REPO_ROOT}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py" \
            --routes="${ROUTE_FILE}" \
            --repetitions=1 \
            --track=SENSORS \
            --checkpoint="${RESULT_FILE}" \
            --timeout=600 \
            --agent="${AGENT_FILE}" \
            --agent-config="${CHECKPOINT}" \
            --traffic-manager-seed="${SEED}" \
            --port="${PORT}" \
            --traffic-manager-port="${TM_PORT}" \
            --resume=True \
            2>&1 | tee "${EVAL_LOG}" &
    else
        echo "🚀 Starting fresh evaluation"
        ${TIMEOUT_CMD} python -u "${REPO_ROOT}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py" \
            --routes="${ROUTE_FILE}" \
            --repetitions=1 \
            --track=SENSORS \
            --checkpoint="${RESULT_FILE}" \
            --timeout=600 \
            --agent="${AGENT_FILE}" \
            --agent-config="${CHECKPOINT}" \
            --traffic-manager-seed="${SEED}" \
            --port="${PORT}" \
            --traffic-manager-port="${TM_PORT}" \
            2>&1 | tee "${EVAL_LOG}" &
    fi
    
    local eval_pid=$!
    echo "[Eval] Evaluation process started with PID: ${eval_pid}"
    
    # Start monitor in background
    monitor_evaluation ${eval_pid} "${EVAL_LOG}" 900 &
    local monitor_pid=$!
    
    # Wait for evaluation to complete
    wait ${eval_pid} 2>/dev/null
    local exit_code=$?
    
    # Stop monitor (it will exit when eval_pid dies)
    kill ${monitor_pid} 2>/dev/null || true
    wait ${monitor_pid} 2>/dev/null || true
    
    # Check exit code
    if [ ${exit_code} -eq 124 ]; then
        echo "⚠️  Evaluation timed out after ${MAX_RUNTIME}s"
        cleanup_carla
        return 124
    elif [ ${exit_code} -ne 0 ]; then
        echo "⚠️  Evaluation exited with code: ${exit_code}"
        # Check log for timeout errors
        if [ -f "${EVAL_LOG}" ]; then
            if grep -q "time-out of.*while waiting for the simulator" "${EVAL_LOG}" 2>/dev/null || \
               grep -q "Watchdog exception.*Timeout" "${EVAL_LOG}" 2>/dev/null || \
               grep -q "RuntimeError.*time-out" "${EVAL_LOG}" 2>/dev/null || \
               grep -q "The simulation took longer than.*to update" "${EVAL_LOG}" 2>/dev/null; then
                echo "⚠️  Detected timeout/hang error in evaluation log"
                cleanup_carla
                return 124  # Return timeout code
            fi
        fi
    fi
    
    return ${exit_code}
}

# Main evaluation loop with auto-restart
echo "Starting evaluation with auto-restart support..."
echo "Max restarts: ${MAX_RESTARTS}"
echo "Restart delay: ${RESTART_DELAY} seconds"

while true; do
    # Check if checkpoint exists and is valid for resume
    if [ -f "${RESULT_FILE}" ] && [ "$USE_RESUME" = false ]; then
        # Check if checkpoint indicates we can resume
        CHECKPOINT_STATUS=$(python3 -c "
import json
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    entry_status = data.get('entry_status', '')
    checkpoint = data.get('_checkpoint', {})
    progress = checkpoint.get('progress', [0, 0])
    
    if entry_status == 'Crashed':
        print('crashed')  # Can resume from crash
    elif entry_status == 'Started':
        # Started but interrupted (e.g., time limit reached) - can resume
        if progress and len(progress) >= 2 and progress[0] < progress[1]:
            print('started_incomplete')  # Started but not all routes done
        else:
            print('started')  # Started but progress unclear, try to resume
    elif entry_status in ['Completed', '']:
        if progress and len(progress) >= 2 and progress[0] < progress[1]:
            print('incomplete')  # Can resume, not all routes done
        else:
            print('completed')  # Already completed
    else:
        print('unknown')  # Unknown status, try to resume
except Exception as e:
    print('invalid')  # Invalid checkpoint
" 2>/dev/null || echo "invalid")
        
        if [ "$CHECKPOINT_STATUS" = "crashed" ] || [ "$CHECKPOINT_STATUS" = "incomplete" ] || [ "$CHECKPOINT_STATUS" = "started" ] || [ "$CHECKPOINT_STATUS" = "started_incomplete" ] || [ "$CHECKPOINT_STATUS" = "unknown" ]; then
            USE_RESUME=true
            echo "📋 Found existing checkpoint (status: ${CHECKPOINT_STATUS}), will resume from it"
        elif [ "$CHECKPOINT_STATUS" = "completed" ]; then
            echo "ℹ️  Checkpoint exists and evaluation appears complete, starting fresh"
            USE_RESUME=false
        else
            echo "ℹ️  Checkpoint exists but appears invalid, starting fresh"
            USE_RESUME=false
        fi
    fi
    
    # Clean up any existing CARLA processes before starting
    cleanup_carla
    
    # Run evaluation
    run_evaluation
    EXIT_CODE=$?
    
    # Check if evaluation timed out or hung
    if [ ${EXIT_CODE} -eq 124 ]; then
        echo ""
        echo "⚠️  Evaluation timed out or hung, will restart..."
        EVAL_STATUS="timeout"
        # Force cleanup
        cleanup_carla
        sleep 5
    else
        # Check if evaluation completed successfully (no crash)
        # Check the checkpoint file to determine actual status
        EVAL_STATUS="unknown"
        if [ -f "${RESULT_FILE}" ]; then
            EVAL_STATUS=$(python3 -c "
import json
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    entry_status = data.get('entry_status', 'Unknown')
    checkpoint = data.get('_checkpoint', {})
    progress = checkpoint.get('progress', [0, 0])
    
    if entry_status == 'Crashed':
        print('crashed')
    elif entry_status in ['Completed', '']:
        if progress and len(progress) >= 2 and progress[0] >= progress[1]:
            print('completed')
        else:
            print('incomplete')
    elif entry_status == 'Started':
        # Started but interrupted - treat as incomplete
        print('incomplete')
    else:
        print('incomplete')  # Unknown status, treat as incomplete
except Exception as e:
    print('error')
" 2>/dev/null || echo "error")
        fi
        
        # If completed, exit successfully
        if [ "$EVAL_STATUS" = "completed" ]; then
            echo ""
            echo "✅ Evaluation completed successfully!"
            echo "Results saved to: ${RESULT_FILE}"
            echo "Visualizations saved to: ${VIZ_PATH}"
            exit 0
        fi
    fi
    
    # If crashed or incomplete, we need to restart
    # (If completed, we already exited above)
    RESTART_COUNT=$((RESTART_COUNT + 1))
    
    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        echo ""
        echo "❌ Maximum restart limit (${MAX_RESTARTS}) reached. Stopping."
        echo "Current results saved to: ${RESULT_FILE}"
        exit 1
    fi
    
    echo ""
    if [ "${EVAL_STATUS}" = "timeout" ]; then
        echo "⚠️  Evaluation timed out or hung"
        echo "   This usually indicates CARLA simulator froze or became unresponsive"
        echo "   CARLA processes have been killed, will restart evaluation"
    else
        echo "⚠️  Evaluation crashed or incomplete"
        echo "   Status: ${EVAL_STATUS}"
    fi
    echo "   Exit code: ${EXIT_CODE}"
    echo "   Restart count: ${RESTART_COUNT}/${MAX_RESTARTS}"
    echo "⏳ Waiting ${RESTART_DELAY} seconds before restart..."
    
    # Clean up CARLA processes
    cleanup_carla
    
    # Wait before restarting
    sleep ${RESTART_DELAY}
    
    # Set resume flag for next attempt
    USE_RESUME=true
    
    echo "🔄 Restarting evaluation..."
done

