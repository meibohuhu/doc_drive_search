#!/bin/bash
# 批量评估脚本的使用示例
    # verbose_prompt

# 设置路径
CSV_FILE="/code/doc_drive_search/data/pending_process/simlingo_bench2drive220_newcommand_part2_merged_eval_steps_extracted.csv"
IMAGE_BASE="/code/doc_drive_search/eval_results/Bench2Drive/simlingo_bench2drive220_newcommand_part1/RouteScenario_1792_rep0_Town12_HazardAtSideLane_1_18_02_03_23_39_48/debug_viz/simlingo/iter_013.ckpt"
OUTPUT_DIR="/code/doc_drive_search/data/pending_process"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"  # 或 "Qwen/Qwen2.5-VL-7B-Instruct" Qwen/Qwen3-VL-2B-Instruct google/gemma-3-1b-it
DEVICE="cuda"

# 运行评估
python3 /code/doc_drive_search/scripts/batch_command_alignment_evaluator.py \
    --csv "$CSV_FILE" \
    --image_base "$IMAGE_BASE" \
    --output "$OUTPUT_DIR/command_alignment_results.json" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --interval 5 \
    --max_tokens 200 \

echo "Evaluation completed! Results saved to: $OUTPUT_DIR/command_alignment_results.json"


