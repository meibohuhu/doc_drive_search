#!/bin/bash
# 本地运行 Bench2Drive 评估脚本（不使用 SLURM）
###

### conda activate simlingo
### source carla_exports.sh
### bash run_eval_local.sh

### bash Bench2Drive/tools/clean_carla.sh

###
# 设置路径
export CARLA_ROOT=/code/software/carla0915
export WORK_DIR=/code/doc_drive_search/Bench2Drive  # Bench2Drive 需要 WORK_DIR 指向 Bench2Drive 目录
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
# 项目根目录（用于 team_code 等模块）
export PROJECT_ROOT=/code/doc_drive_search
# 设置 PYTHONPATH，确保 Bench2Drive 的 leaderboard 优先
# 注意：顺序很重要，Bench2Drive 的模块必须在项目根目录的 leaderboard 之前
export PYTHONPATH=${WORK_DIR}/scenario_runner:${PYTHONPATH}
export PYTHONPATH=${WORK_DIR}/leaderboard:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}  # 添加 PROJECT_ROOT 以便导入 team_code
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}

# 评估配置
BASE_PORT=20001
BASE_TM_PORT=30001
TRAFFIC_MANAGER_SEED=1  # 添加traffic manager seed，与官方脚本一致
IS_BENCH2DRIVE=True
# ROUTES='/local1/mhu/doc_drive_search/leaderboard_backup/data/bench2drive_split/bench2drive_199.xml'  #  bench2drive_mini_10
ROUTES='/code/doc_drive_search/Bench2Drive/data/bench2drive220_part2_1.xml'  #  bench2drive_mini_10
TEAM_AGENT=${PROJECT_ROOT}/team_code/agent_simlingo.py
TEAM_CONFIG=${PROJECT_ROOT}/pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
CHECKPOINT_ENDPOINT=${PROJECT_ROOT}/eval_results/Bench2Drive/simlingo_bench2drive220_newcommand_part2_1.json
SAVE_PATH=${PROJECT_ROOT}/eval_results/Bench2Drive/simlingo_bench2drive220_newcommand_part2_1
PLANNER_TYPE=only_traj
GPU_RANK=6  # 如果 GPU 0 有问题，可以尝试 GPU_RANK=1 或 2
RESUME=True  # 改为False，除非需要从checkpoint恢复（官方脚本没有这个参数）

# 设置环境变量供 agent 使用
export SAVE_PATH=${SAVE_PATH}

# 创建输出目录
mkdir -p ${SAVE_PATH}


# 检查路由文件是否存在
if [ ! -f "${ROUTES}" ]; then
    echo "❌ 错误: 路由文件不存在: ${ROUTES}"
    echo "请先准备路由文件（见 BENCH2DRIVE_EVAL_GUIDE.md）"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "${TEAM_CONFIG}" ]; then
    echo "❌ 错误: 模型文件不存在: ${TEAM_CONFIG}"
    echo "请先下载模型: python download_model.py"
    exit 1
fi

# 临时重命名项目根目录的 leaderboard，避免导入冲突
if [ -d "${PROJECT_ROOT}/leaderboard" ] && [ ! -L "${PROJECT_ROOT}/leaderboard" ]; then
    echo "⚠️  临时重命名项目根目录的 leaderboard 以避免导入冲突..."
    mv "${PROJECT_ROOT}/leaderboard" "${PROJECT_ROOT}/leaderboard_backup"
    LEADERBOARD_BACKUP=true
else
    LEADERBOARD_BACKUP=false
fi

# 注意：Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py 会自动启动 CARLA
# 因此不需要手动启动 CARLA 服务器
echo "注意: leaderboard_evaluator.py 会自动启动 CARLA 服务器"
CARLA_PID=""  # 不再手动管理 CARLA PID

# 定义清理函数
cleanup_carla() {
    echo "执行清理操作..."
    # 杀死 CARLA 进程（如果有手动启动的）
    if [ ! -z "${CARLA_PID}" ]; then
        kill ${CARLA_PID} 2>/dev/null || true
        wait ${CARLA_PID} 2>/dev/null || true
    fi
    # 清理所有 CARLA 进程（包括 evaluator 自动启动的）
    pkill -f CarlaUE4 2>/dev/null || true
    # 恢复 leaderboard 目录
    if [ "$LEADERBOARD_BACKUP" = true ]; then
        echo "恢复 leaderboard 目录..."
        mv "${PROJECT_ROOT}/leaderboard_backup" "${PROJECT_ROOT}/leaderboard"
    fi
    # 清理所有 CARLA 进程（备用方法）
    bash "${PROJECT_ROOT}/clean_carla.sh" 2>/dev/null || true
    echo "清理完成"
}

# 注册清理函数
# trap cleanup_carla EXIT INT TERM

# 运行评估（使用 Bench2Drive 的 leaderboard_evaluator.py，它会自动启动 CARLA）
echo "开始评估..."
echo "提示: 评估过程中请保持终端打开，不要中断"
echo "如果评估停止，请检查："
echo "  1. CARLA服务器是否还在运行: ps aux | grep CarlaUE4"
echo "  2. GPU内存是否充足: nvidia-smi"
echo "  3. 系统内存是否充足: free -h"
echo ""

# 注意：切换到项目根目录，以便导入 team_code 模块
cd ${PROJECT_ROOT}

# 设置Python的unbuffered输出，以便实时看到日志
export PYTHONUNBUFFERED=1

# 运行评估并捕获输出
EVAL_LOG="${SAVE_PATH}/eval_$(date +%Y%m%d_%H%M%S).log"
echo "评估日志将保存到: ${EVAL_LOG}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_RANK} python -u ${WORK_DIR}/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes="${ROUTES}" \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint="${CHECKPOINT_ENDPOINT}" \
    --agent="${TEAM_AGENT}" \
    --agent-config="${TEAM_CONFIG}" \
    --debug=0 \
    --resume=${RESUME} \
    --port=${BASE_PORT} \
    --traffic-manager-port=${BASE_TM_PORT} \
    --traffic-manager-seed=${TRAFFIC_MANAGER_SEED} \
    --gpu-rank=1 \
    --timeout=600 2>&1 | tee "${EVAL_LOG}" || {  # timeout=600秒（10分钟），如果agent慢可以增加
    EVAL_EXIT_CODE=${PIPESTATUS[0]}
    echo ""
    echo "=========================================="
    echo "❌ 评估失败，退出码: $EVAL_EXIT_CODE"
    echo "=========================================="
    echo "请检查日志文件: ${EVAL_LOG}"
    echo "最后50行日志:"
    tail -n 50 "${EVAL_LOG}" || true
    echo ""
    echo "检查CARLA进程:"
    ps aux | grep -i carla | grep -v grep || echo "  没有CARLA进程在运行"
    echo ""
    # cleanup_carla
    exit $EVAL_EXIT_CODE
}

echo "评估完成！结果保存在: ${CHECKPOINT_ENDPOINT}"

