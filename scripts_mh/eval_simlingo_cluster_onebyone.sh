#!/bin/bash -l
# NOTE the -l flag!
#
# SimLingo Evaluation on Cluster (SLURM)
# Evaluates SimLingo model on Bench2Drive benchmark
# Processes all routes in onebyone directory sequentially in a single SLURM job
#
# Usage:
#   sbatch eval_simlingo_cluster_onebyone.sh [ROUTE_DIR] [SEED] [CHECKPOINT] [MAX_RETRIES] [ROUTE_MOD_OFFSET]
#
# Parameters:
#   ROUTE_DIR: Directory containing route XML files (default: /home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/onebyone)
#   SEED: Traffic manager seed (default: 1)
#   CHECKPOINT: Path to model checkpoint (default: pretrained model path)
#   MAX_RETRIES: Maximum retry attempts for failed routes (default: 1)
#   ROUTE_MOD_OFFSET: Optional filter to process only routes where route_id % 4 == OFFSET (0, 1, 2, or 3)
#                     If not specified, processes all routes
#
# Example: Process 1/4 of routes (route_id % 4 == 0)
#   sbatch eval_simlingo_cluster_onebyone.sh "" "" "" "" 0

#SBATCH --job-name=simlingo_eval_onebyone
#SBATCH --error=/home/mh2803/projects/simlingo/scripts/cluster_logs/eval_onebyone_err_%j.txt
#SBATCH --output=/home/mh2803/projects/simlingo/scripts/cluster_logs/eval_onebyone_out_%j.txt
#SBATCH --account=llm-gen-agent
#SBATCH --nodes=1
#SBATCH --time=09:20:00  # 3 days for processing all routes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=tier3
#SBATCH --mem=64G

# Configuration - modify these variables as needed
ROUTE_DIR="${1:-/home/mh2803/projects/simlingo/leaderboard/data/bench2drive_split}"
SEED="${2:-1}"
# CHECKPOINT="${3:-/shared/rc/llm-gen-agent/mhu/pretrained_models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
CHECKPOINT="${3:-/shared/rc/llm-gen-agent/mhu/simlingo_checkpoints/2026_01_30_14_47_27_sim_training_full_command_0130/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
MAX_RETRIES="${4:-1}"  # Maximum number of retries for failed routes (default: 1)
ROUTE_MOD_OFFSET="${5:-3}"  # Optional: filter routes by route_id % 4 (0, 1, 2, or 3). If not set, processes all routes.

AGENT_FILE="/home/mh2803/projects/simlingo/team_code/agent_simlingo.py"
REPO_ROOT="/home/mh2803/projects/simlingo"
CARLA_ROOT="${CARLA_ROOT:-/home/mh2803/software/carla0915}"
OUT_ROOT="${REPO_ROOT}/eval_results_rc/Bench2Drive"
AGENT_NAME="simlingo"
BENCHMARK="bench2drive"

# Base ports - will be incremented for each route
BASE_PORT=20002
BASE_TM_PORT=30002
PORT_INCREMENT=5

echo "=========================================="
echo "JOB ID: $SLURM_JOB_ID"
echo "Route Directory: ${ROUTE_DIR}"
echo "Seed: ${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
if [ -n "${ROUTE_MOD_OFFSET}" ]; then
    echo "ROUTE_MOD_OFFSET: ${ROUTE_MOD_OFFSET} (processing routes where route_id % 4 == ${ROUTE_MOD_OFFSET})"
else
    echo "Processing all routes sequentially..."
fi
echo "Max retries per route: ${MAX_RETRIES}"
echo "=========================================="

# Load spack modules (adjust based on your cluster setup)
spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Try to load libjpeg-turbo which may provide compatible libjpeg.so.8
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
elif command -v conda &> /dev/null && [ -n "${CONDA_PREFIX}" ]; then
    echo "Installing libjpeg-turbo via conda (this may take a moment)..."
    source /home/mh2803/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate simlingo 2>/dev/null || true
    conda install -y -c conda-forge libjpeg-turbo 2>&1 | tail -5
    if [ -f "${CONDA_PREFIX}/lib/libjpeg.so.8" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
        echo "Successfully installed and using conda libjpeg-turbo from ${CONDA_PREFIX}/lib"
    else
        echo "Warning: libjpeg-turbo installation may have failed"
    fi
fi

# Verify libjpeg.so.8 is available
if [ -z "$(echo ${LD_LIBRARY_PATH} | tr ':' '\n' | xargs -I {} find {} -name 'libjpeg.so.8' 2>/dev/null | head -1)" ]; then
    echo "Warning: libjpeg.so.8 not found in LD_LIBRARY_PATH"
    echo "CARLA may fail to load."
fi

# Change to project directory
cd "${REPO_ROOT}"

# Create base output directories
BASE_DIR="${OUT_ROOT}/${AGENT_NAME}/${BENCHMARK}/${SEED}"
mkdir -p "${BASE_DIR}/res"
mkdir -p "${BASE_DIR}/viz"
mkdir -p "${BASE_DIR}/logs"

# File to track failed routes
FAILED_ROUTES_FILE="${BASE_DIR}/failed_routes.txt"
RETRY_ROUTES_FILE="${BASE_DIR}/retry_routes.txt"

# Get all route files
# If RETRY_ROUTES_FILE exists, only process routes listed there (for retry)
if [ -f "${RETRY_ROUTES_FILE}" ] && [ -s "${RETRY_ROUTES_FILE}" ]; then
    echo "Found retry file: ${RETRY_ROUTES_FILE}"
    echo "Will only process routes listed in the retry file"
    ROUTE_FILES=()
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        # Find the route file matching this route ID or filename
        ROUTE_ID=$(echo "$line" | awk '{print $1}')
        FOUND_FILE=$(ls "${ROUTE_DIR}"/*${ROUTE_ID}*.xml 2>/dev/null | head -1)
        if [ -n "$FOUND_FILE" ] && [ -f "$FOUND_FILE" ]; then
            ROUTE_FILES+=("$FOUND_FILE")
        fi
    done < "${RETRY_ROUTES_FILE}"
    echo "Found ${#ROUTE_FILES[@]} route(s) to retry"
else
    ROUTE_FILES=($(ls "${ROUTE_DIR}"/*.xml 2>/dev/null | sort))
fi

# Filter routes by ROUTE_MOD_OFFSET if specified
if [ -n "${ROUTE_MOD_OFFSET}" ]; then
    if ! [[ "${ROUTE_MOD_OFFSET}" =~ ^[0-3]$ ]]; then
        echo "ERROR: ROUTE_MOD_OFFSET must be 0, 1, 2, or 3"
        exit 1
    fi
    FILTERED_ROUTE_FILES=()
    for ROUTE_FILE in "${ROUTE_FILES[@]}"; do
        ROUTE_BASENAME=$(basename "${ROUTE_FILE}" .xml)
        ROUTE_ID=$(echo "${ROUTE_BASENAME}" | sed 's/.*_//' || echo "${ROUTE_BASENAME}")
        # Remove leading zeros to get numeric route ID
        ROUTE_ID_NUM=$(echo "${ROUTE_ID}" | sed 's/^0*//')
        [ -z "${ROUTE_ID_NUM}" ] && ROUTE_ID_NUM=0
        MOD_RESULT=$((ROUTE_ID_NUM % 4))
        if [ "${MOD_RESULT}" -eq "${ROUTE_MOD_OFFSET}" ]; then
            FILTERED_ROUTE_FILES+=("${ROUTE_FILE}")
        fi
    done
    ROUTE_FILES=("${FILTERED_ROUTE_FILES[@]}")
fi

if [ ${#ROUTE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No route files found in ${ROUTE_DIR}"
    if [ -n "${ROUTE_MOD_OFFSET}" ]; then
        echo "       (after filtering by ROUTE_MOD_OFFSET=${ROUTE_MOD_OFFSET})"
    fi
    exit 1
fi

echo "Found ${#ROUTE_FILES[@]} route file(s) to process"
if [ -n "${ROUTE_MOD_OFFSET}" ]; then
    echo "Filtered by ROUTE_MOD_OFFSET=${ROUTE_MOD_OFFSET} (route_id % 4 == ${ROUTE_MOD_OFFSET})"
fi
echo "Max retries per route: ${MAX_RETRIES}"
echo ""

# Process each route sequentially
TOTAL_ROUTES=${#ROUTE_FILES[@]}
CURRENT_ROUTE=0
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# Initialize failed routes tracking
> "${FAILED_ROUTES_FILE}"  # Clear previous failed routes file

for ROUTE_FILE in "${ROUTE_FILES[@]}"; do
    CURRENT_ROUTE=$((CURRENT_ROUTE + 1))
    
    # Extract route ID from filename (e.g., bench2drive_4468.xml -> 4468)
    # This matches the Python logic: route.split("_")[-1][:-4].zfill(fill_zeros)
    ROUTE_BASENAME=$(basename "${ROUTE_FILE}" .xml)
    # Get the part after the last underscore (e.g., "4468" from "bench2drive_4468")
    ROUTE_ID=$(echo "${ROUTE_BASENAME}" | sed 's/.*_//' || echo "${ROUTE_BASENAME}")
    
    # Pad route ID with zeros to at least 3 digits (but don't truncate if longer)
    # This matches Python's zfill(3) behavior
    if [ ${#ROUTE_ID} -lt 3 ]; then
        ROUTE_ID=$(printf "%03d" "${ROUTE_ID}" 2>/dev/null || echo "${ROUTE_ID}")
    fi
    
    # Calculate ports for this route
    PORT_OFFSET=$((PORT_INCREMENT * (CURRENT_ROUTE - 1)))
    PORT=$((BASE_PORT + PORT_OFFSET))
    TM_PORT=$((BASE_TM_PORT + PORT_OFFSET))
    
    # Set up paths for this route
    VIZ_PATH="${BASE_DIR}/viz/${ROUTE_ID}"
    RESULT_FILE="${BASE_DIR}/res/${ROUTE_ID}_res.json"
    # Log files will be numbered by attempt (e.g., 11715_1_out.log, 11715_2_out.log)
    LOG_FILE_BASE="${BASE_DIR}/logs/${ROUTE_ID}"
    ERR_FILE_BASE="${BASE_DIR}/logs/${ROUTE_ID}"
    
    echo "=========================================="
    echo "[${CURRENT_ROUTE}/${TOTAL_ROUTES}] Processing Route: ${ROUTE_BASENAME}"
    echo "Route ID: ${ROUTE_ID}"
    echo "Route File: ${ROUTE_FILE}"
    echo "CARLA Port: ${PORT}"
    echo "Traffic Manager Port: ${TM_PORT}"
    echo "Result File: ${RESULT_FILE}"
    echo "=========================================="
    
    # Create output directories for this route
    mkdir -p "${VIZ_PATH}"
    mkdir -p "$(dirname ${RESULT_FILE})"
    mkdir -p "${BASE_DIR}/logs"  # Create logs directory (LOG_FILE will be defined in retry loop)
    
    export SAVE_PATH="${VIZ_PATH}"
    
    # Check if route is already completed successfully
    if [ -f "${RESULT_FILE}" ]; then
        # Check if evaluation completed successfully (more thorough check)
        CHECK_RESULT=$(python3 -c "
import json
import sys
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    checkpoint = data.get('_checkpoint', {})
    progress = checkpoint.get('progress', [])
    
    # Check if progress is complete
    if len(progress) < 2 or progress[0] < progress[1]:
        sys.exit(1)
    
    # Check for failed records
    records = checkpoint.get('records', [])
    for record in records:
        status = record.get('status', '')
        if 'Failed' in status:
            sys.exit(1)
    
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            echo "Route ${ROUTE_ID} already completed successfully. Skipping..."
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            SKIP_COUNT=$((SKIP_COUNT + 1))
            echo ""
            continue
        else
            echo "Route ${ROUTE_ID} result file exists but may be incomplete or failed. Reprocessing..."
        fi
    fi
    
    # Track retry attempts for this route
    ROUTE_RETRY_COUNT=0
    ROUTE_SUCCESS=false
    
    # Retry loop for this route
    while [ $ROUTE_RETRY_COUNT -le $MAX_RETRIES ] && [ "$ROUTE_SUCCESS" = false ]; do
        # Calculate attempt number (1-indexed)
        ATTEMPT_NUM=$((ROUTE_RETRY_COUNT + 1))
        
        # Set log file names with attempt number (e.g., 11715_1_out.log, 11715_2_out.log)
        LOG_FILE="${LOG_FILE_BASE}_${ATTEMPT_NUM}_out.log"
        ERR_FILE="${ERR_FILE_BASE}_${ATTEMPT_NUM}_err.log"
        
        if [ $ROUTE_RETRY_COUNT -gt 0 ]; then
            # Retry: clean up previous attempt (only visualization and result file, NOT logs)
            echo "Retrying route ${ROUTE_ID} (attempt ${ATTEMPT_NUM}/${MAX_RETRIES})..."
            rm -rf "${VIZ_PATH}"
            rm -f "${RESULT_FILE}"
        fi
        
        # Clean up old visualization directory
        mkdir -p "${VIZ_PATH}"
        
        # Run evaluation for this route
        if [ $ROUTE_RETRY_COUNT -eq 0 ]; then
            echo "Starting evaluation for route ${ROUTE_ID}..."
        fi
        echo "Logging to: ${LOG_FILE}"
        echo "Errors to: ${ERR_FILE}"
        
        START_TIME=$(date +%s)
        
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
            --traffic-manager-port="${TM_PORT}" \
            > "${LOG_FILE}" 2> "${ERR_FILE}"
        
        EVAL_EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        # Check if evaluation was successful (more thorough check)
        if [ $EVAL_EXIT_CODE -eq 0 ] && [ -f "${RESULT_FILE}" ]; then
            # Verify the result file is valid and complete
            CHECK_RESULT=$(python3 -c "
import json
import sys
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    checkpoint = data.get('_checkpoint', {})
    progress = checkpoint.get('progress', [])
    
    # Check if progress is complete
    if len(progress) < 2 or progress[0] < progress[1]:
        sys.exit(1)
    
    # Check for failed records
    records = checkpoint.get('records', [])
    for record in records:
        status = record.get('status', '')
        if 'Failed' in status:
            sys.exit(1)
    
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null)
            
            if [ $? -eq 0 ]; then
                echo "✅ Route ${ROUTE_ID} completed successfully in ${DURATION} seconds"
                if [ $ROUTE_RETRY_COUNT -gt 0 ]; then
                    echo "   (succeeded on retry attempt $((ROUTE_RETRY_COUNT + 1)))"
                fi
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                ROUTE_SUCCESS=true
            else
                echo "⚠️  Route ${ROUTE_ID} evaluation finished but result may be incomplete or contains failures"
                ROUTE_RETRY_COUNT=$((ROUTE_RETRY_COUNT + 1))
            fi
        else
            echo "❌ Route ${ROUTE_ID} evaluation failed (exit code: ${EVAL_EXIT_CODE})"
            ROUTE_RETRY_COUNT=$((ROUTE_RETRY_COUNT + 1))
        fi
        
        # If failed and still have retries, wait before retrying
        if [ "$ROUTE_SUCCESS" = false ] && [ $ROUTE_RETRY_COUNT -le $MAX_RETRIES ]; then
            echo "   Will retry route ${ROUTE_ID}..."
            sleep 5  # Wait a bit before retrying
        fi
    done
    
    # If route still failed after all retries, record it
    if [ "$ROUTE_SUCCESS" = false ]; then
        echo "❌ Route ${ROUTE_ID} failed after ${MAX_RETRIES} retry attempts"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        # Record failed route to file
        echo "${ROUTE_ID} ${ROUTE_FILE}" >> "${FAILED_ROUTES_FILE}"
    fi
    
    echo "Results saved to: ${RESULT_FILE}"
    echo "Visualizations saved to: ${VIZ_PATH}"
    echo ""
    
    # Small delay between routes to ensure clean shutdown
    sleep 2
done

echo "=========================================="
echo "All routes processed!"
echo "Total routes: ${TOTAL_ROUTES}"
echo "Successful: ${SUCCESS_COUNT}"
echo "Skipped (already completed): ${SKIP_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "=========================================="

# Create retry file for failed routes
if [ ${FAIL_COUNT} -gt 0 ] && [ -f "${FAILED_ROUTES_FILE}" ] && [ -s "${FAILED_ROUTES_FILE}" ]; then
    echo ""
    echo "Failed routes saved to: ${FAILED_ROUTES_FILE}"
    echo "To retry only failed routes, run:"
    echo "  sbatch ${0} ${ROUTE_DIR} ${SEED} ${CHECKPOINT} ${MAX_RETRIES}"
    echo "  (after copying ${FAILED_ROUTES_FILE} to ${RETRY_ROUTES_FILE})"
    echo ""
    echo "Or create ${RETRY_ROUTES_FILE} with route IDs (one per line):"
    cat "${FAILED_ROUTES_FILE}" | awk '{print $1}'
    echo ""
    
    # Copy failed routes to retry file for convenience
    cp "${FAILED_ROUTES_FILE}" "${RETRY_ROUTES_FILE}"
    echo "Created ${RETRY_ROUTES_FILE} for easy retry"
fi

# Exit with error if any routes failed
if [ ${FAIL_COUNT} -gt 0 ]; then
    exit 1
else
    exit 0
fi
