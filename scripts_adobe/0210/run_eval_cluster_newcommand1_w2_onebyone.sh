#!/bin/bash
# 本地运行 Bench2Drive 评估脚本（不使用 SLURM）- One-by-One 格式
# 此脚本处理 routes 的第 1/4 部分 (ROUTE_MOD_OFFSET=0)


# # 找到 script 7 的进程
# ps aux | grep "run_eval_cluster_oldcommand3_onebyone.sh" | grep -v grep
# ps aux | grep "run_eval_cluster_newcommand5_w2_onebyone.sh.sh" | grep -v grep


# # 然后用它的 PID 终止（假设 PID 是 162081）
# kill -9 162081

#### /code/doc_drive_search/Bench2Drive/data/bench2drive_split
ROUTE_DIR="${1:-/code/doc_drive_search/Bench2Drive/data/bench2drive_split}"
SEED="${2:-1}"
CHECKPOINT="${3:-/code/doc_drive_search/pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
GPU_RANK="${4:-5}"  # 脚本1默认使用 GPU 0
MAX_RETRIES="${5:-0}"
ROUTE_MOD_OFFSET="${6:-0}"  # 默认处理第 1/8 部分 (route_id % 8 == 0)

export CARLA_ROOT=/home/colligo/software/carla0915
export WORK_DIR=/code/doc_drive_search/Bench2Drive
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PROJECT_ROOT=/code/doc_drive_search

BASE_PORT=20000  # 脚本1使用端口范围 20000-20000+(routes*5)
BASE_TM_PORT=30000  # 脚本1使用 TM 端口范围 30000-30000+(routes*5)
PORT_INCREMENT=5

TRAFFIC_MANAGER_SEED=${SEED}
TEAM_AGENT=${PROJECT_ROOT}/team_code/agent_simlingo_cfg.py
TEAM_CONFIG=${CHECKPOINT}
RESUME=True

OUT_ROOT=${PROJECT_ROOT}/eval_results/agent_simlingo_cfg
AGENT_NAME="simlingo"
BENCHMARK="bench2drive"

export PYTHONPATH=${WORK_DIR}/scenario_runner:${PYTHONPATH}
export PYTHONPATH=${WORK_DIR}/leaderboard:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}

if [ ! -f "${TEAM_CONFIG}" ]; then
    echo "ERROR: Model file not found: ${TEAM_CONFIG}"
    exit 1
fi

if [ -d "${PROJECT_ROOT}/leaderboard" ] && [ ! -L "${PROJECT_ROOT}/leaderboard" ]; then
    mv "${PROJECT_ROOT}/leaderboard" "${PROJECT_ROOT}/leaderboard_backup"
    LEADERBOARD_BACKUP=true
else
    LEADERBOARD_BACKUP=false
fi

cleanup_carla() {
    local CARLA_PORT="${1:-}"
    [ -z "${CARLA_PORT}" ] && return 0
    
    local PIDS=$(ps aux | grep -E "[C]arlaUE4.*-carla-rpc-port=${CARLA_PORT}" | awk '{print $2}')
    if [ -n "${PIDS}" ]; then
        for PID in ${PIDS}; do
            kill -TERM "${PID}" 2>/dev/null || true
        done
        sleep 2
        for PID in ${PIDS}; do
            kill -0 "${PID}" 2>/dev/null && kill -9 "${PID}" 2>/dev/null || true
        done
    fi
    
    if command -v lsof >/dev/null 2>&1; then
        local LSOF_PIDS=$(lsof -ti:${CARLA_PORT} 2>/dev/null || true)
        if [ -n "${LSOF_PIDS}" ]; then
            for PID in ${LSOF_PIDS}; do
                ps -p "${PID}" -o comm= 2>/dev/null | grep -q -i carla && kill -TERM "${PID}" 2>/dev/null || true
            done
            sleep 1
            for PID in ${LSOF_PIDS}; do
                if kill -0 "${PID}" 2>/dev/null; then
                    ps -p "${PID}" -o comm= 2>/dev/null | grep -q -i carla && kill -9 "${PID}" 2>/dev/null || true
                fi
            done
        fi
    fi
    sleep 1
}

cd ${PROJECT_ROOT}
export PYTHONUNBUFFERED=1

BASE_DIR="${OUT_ROOT}/${AGENT_NAME}/${BENCHMARK}/0"
mkdir -p "${BASE_DIR}/res" "${BASE_DIR}/viz" "${BASE_DIR}/logs"

SCRIPT_LOG="${BASE_DIR}/script_run.log"
echo "===========================================" | tee "${SCRIPT_LOG}"
echo "Script started at $(date)" | tee -a "${SCRIPT_LOG}"
echo "===========================================" | tee -a "${SCRIPT_LOG}"


FAILED_ROUTES_FILE="${BASE_DIR}/failed_routes.txt"
RETRY_ROUTES_FILE="${BASE_DIR}/retry_routes.txt"

if [ -f "${RETRY_ROUTES_FILE}" ] && [ -s "${RETRY_ROUTES_FILE}" ]; then
    ROUTE_FILES=()
    while IFS= read -r line || [ -n "$line" ]; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        ROUTE_ID=$(echo "$line" | awk '{print $1}')
        FOUND_FILE=$(ls "${ROUTE_DIR}"/*${ROUTE_ID}*.xml 2>/dev/null | head -1)
        [ -n "$FOUND_FILE" ] && [ -f "$FOUND_FILE" ] && ROUTE_FILES+=("$FOUND_FILE")
    done < "${RETRY_ROUTES_FILE}"
else
    ROUTE_FILES=($(ls "${ROUTE_DIR}"/*.xml 2>/dev/null | sort))
fi

if [ -n "${ROUTE_MOD_OFFSET}" ]; then
    if ! [[ "${ROUTE_MOD_OFFSET}" =~ ^[0-7]$ ]]; then
        echo "ERROR: ROUTE_MOD_OFFSET must be 0, 1, 2, 3, 4, 5, 6, or 7"
        exit 1
    fi
    FILTERED_ROUTE_FILES=()
    for ROUTE_FILE in "${ROUTE_FILES[@]}"; do
        ROUTE_BASENAME=$(basename "${ROUTE_FILE}" .xml)
        ROUTE_ID=$(echo "${ROUTE_BASENAME}" | sed 's/.*_//' || echo "${ROUTE_BASENAME}")
        ROUTE_ID_NUM=$(echo "${ROUTE_ID}" | sed 's/^0*//')
        [ -z "${ROUTE_ID_NUM}" ] && ROUTE_ID_NUM=0
        MOD_RESULT=$((ROUTE_ID_NUM % 8))
        [ "${MOD_RESULT}" -eq "${ROUTE_MOD_OFFSET}" ] && FILTERED_ROUTE_FILES+=("${ROUTE_FILE}")
    done
    ROUTE_FILES=("${FILTERED_ROUTE_FILES[@]}")
fi

if [ ${#ROUTE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No route files found"
    [ "$LEADERBOARD_BACKUP" = true ] && mv "${PROJECT_ROOT}/leaderboard_backup" "${PROJECT_ROOT}/leaderboard"
    exit 1
fi

echo "=========================================="
echo "Processing routes with ROUTE_MOD_OFFSET=${ROUTE_MOD_OFFSET}"
echo "This script handles routes where route_id % 8 == ${ROUTE_MOD_OFFSET}"
echo "Using GPU: ${GPU_RANK}"
echo "Port range: ${BASE_PORT}-$((BASE_PORT + ${#ROUTE_FILES[@]} * PORT_INCREMENT))"
echo "TM Port range: ${BASE_TM_PORT}-$((BASE_TM_PORT + ${#ROUTE_FILES[@]} * PORT_INCREMENT))"
echo "=========================================="

TOTAL_ROUTES=${#ROUTE_FILES[@]}
CURRENT_ROUTE=0
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

> "${FAILED_ROUTES_FILE}"

for ROUTE_FILE in "${ROUTE_FILES[@]}"; do
    CURRENT_ROUTE=$((CURRENT_ROUTE + 1))
    ROUTE_BASENAME=$(basename "${ROUTE_FILE}" .xml)
    ROUTE_ID=$(echo "${ROUTE_BASENAME}" | sed 's/.*_//' || echo "${ROUTE_BASENAME}")
    [ ${#ROUTE_ID} -lt 3 ] && ROUTE_ID=$(printf "%03d" "${ROUTE_ID}" 2>/dev/null || echo "${ROUTE_ID}")
    
    PORT_OFFSET=$((PORT_INCREMENT * (CURRENT_ROUTE - 1)))
    PORT=$((BASE_PORT + PORT_OFFSET))
    TM_PORT=$((BASE_TM_PORT + PORT_OFFSET))
    
    VIZ_PATH="${BASE_DIR}/viz/${ROUTE_ID}"
    RESULT_FILE="${BASE_DIR}/res/${ROUTE_ID}_res.json"
    LOG_FILE="${BASE_DIR}/logs/${ROUTE_ID}_out.log"
    ERR_FILE="${BASE_DIR}/logs/${ROUTE_ID}_err.log"
    
    mkdir -p "${VIZ_PATH}" "$(dirname ${RESULT_FILE})" "$(dirname ${LOG_FILE})"
    export SAVE_PATH="${VIZ_PATH}"
    
    if [ -f "${RESULT_FILE}" ]; then
        CHECK_RESULT=$(python3 -c "
import json
import sys
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    checkpoint = data.get('_checkpoint', {})
    progress = checkpoint.get('progress', [])
    # Skip if route has been processed
    if len(progress) >= 2:
        records = checkpoint.get('records', [])
        if records:
            status = records[0].get('status', '')
            # Skip both Completed and Failed routes
            if 'Completed' in status or 'Failed' in status:
                sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null)
        if [ $? -eq 0 ]; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            echo "[${CURRENT_ROUTE}/${TOTAL_ROUTES}] Route ${ROUTE_ID}: SKIPPED (already processed)" | tee -a "${SCRIPT_LOG}"
            continue
        fi
    fi
    
    ROUTE_RETRY_COUNT=0
    ROUTE_SUCCESS=false
    
    while [ $ROUTE_RETRY_COUNT -le $MAX_RETRIES ] && [ "$ROUTE_SUCCESS" = false ]; do
        [ $ROUTE_RETRY_COUNT -gt 0 ] && cleanup_carla "${PORT}" && rm -rf "${VIZ_PATH}" && rm -f "${RESULT_FILE}"
        cleanup_carla "${PORT}"
        mkdir -p "${VIZ_PATH}"
        
        START_TIME=$(date +%s)
        CUDA_VISIBLE_DEVICES=${GPU_RANK} python -u ${WORK_DIR}/leaderboard/leaderboard/leaderboard_evaluator.py \
            --routes="${ROUTE_FILE}" \
            --repetitions=1 \
            --track=SENSORS \
            --checkpoint="${RESULT_FILE}" \
            --agent="${TEAM_AGENT}" \
            --agent-config="${TEAM_CONFIG}" \
            --debug=0 \
            --resume=${RESUME} \
            --port="${PORT}" \
            --traffic-manager-port="${TM_PORT}" \
            --traffic-manager-seed="${TRAFFIC_MANAGER_SEED}" \
            --gpu-rank=${GPU_RANK} \
            --timeout=600 \
            > "${LOG_FILE}" 2> "${ERR_FILE}"
        
        EVAL_EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        cleanup_carla "${PORT}"
        
        if [ $EVAL_EXIT_CODE -eq 0 ] && [ -f "${RESULT_FILE}" ]; then
            CHECK_RESULT=$(python3 -c "
import json
import sys
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    checkpoint = data.get('_checkpoint', {})
    progress = checkpoint.get('progress', [])
    if len(progress) < 2 or progress[0] < progress[1]:
        sys.exit(1)
    records = checkpoint.get('records', [])
    for record in records:
        if 'Failed' in record.get('status', ''):
            sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null)
            
            if [ $? -eq 0 ]; then
                echo "[${CURRENT_ROUTE}/${TOTAL_ROUTES}] Route ${ROUTE_ID}: OK (${DURATION}s)" | tee -a "${SCRIPT_LOG}"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                ROUTE_SUCCESS=true
            else
                ROUTE_RETRY_COUNT=$((ROUTE_RETRY_COUNT + 1))
            fi
        else
            echo "[${CURRENT_ROUTE}/${TOTAL_ROUTES}] Route ${ROUTE_ID}: FAILED (exit ${EVAL_EXIT_CODE})" | tee -a "${SCRIPT_LOG}"
            ROUTE_RETRY_COUNT=$((ROUTE_RETRY_COUNT + 1))
        fi
        
        [ "$ROUTE_SUCCESS" = false ] && [ $ROUTE_RETRY_COUNT -le $MAX_RETRIES ] && sleep 5
    done
    
    if [ "$ROUTE_SUCCESS" = false ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "${ROUTE_ID} ${ROUTE_FILE}" >> "${FAILED_ROUTES_FILE}"
    fi
    
    sleep 2
done
echo "Summary: ${SUCCESS_COUNT} success, ${SKIP_COUNT} skipped, ${FAIL_COUNT} failed" | tee -a "${SCRIPT_LOG}"
[ ${FAIL_COUNT} -gt 0 ] && [ -f "${FAILED_ROUTES_FILE}" ] && [ -s "${FAILED_ROUTES_FILE}" ] && \
    cp "${FAILED_ROUTES_FILE}" "${RETRY_ROUTES_FILE}"
[ "$LEADERBOARD_BACKUP" = true ] && mv "${PROJECT_ROOT}/leaderboard_backup" "${PROJECT_ROOT}/leaderboard"
echo "===========================================" | tee -a "${SCRIPT_LOG}"
echo "Script finished at $(date)" | tee -a "${SCRIPT_LOG}"
echo "===========================================" | tee -a "${SCRIPT_LOG}"
[ ${FAIL_COUNT} -gt 0 ] && exit 1 || exit 0
