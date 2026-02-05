#!/bin/bash
# 批量评估脚本的使用示例
    # verbose_prompt

# 设置路径
CSV_FILE="/local1/mhu/doc_drive_search/data/bench2drive_multiple_test_goodwithtworoutes/eval_steps_extracted.csv"
IMAGE_BASE="/local1/mhu/doc_drive_search/data/bench2drive_multiple_test_goodwithtworoutes/RouteScenario_1711_rep0_Town12_ParkingCutIn_1_15_02_03_02_54_29"
OUTPUT_DIR="/local1/mhu/doc_drive_search/data/bench2drive_multiple_test_goodwithtworoutes"
MODEL="Qwen/Qwen3-VL-2B-Instruct"  # 或 "Qwen/Qwen2.5-VL-7B-Instruct" Qwen/Qwen3-VL-2B-Instruct google/gemma-3-1b-it
DEVICE="cuda"

# 运行评估
python3 /local1/mhu/doc_drive_search/scripts/batch_command_alignment_evaluator.py \
    --csv "$CSV_FILE" \
    --image_base "$IMAGE_BASE" \
    --output "$OUTPUT_DIR/command_alignment_results.json" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --interval 5 \
    --max_tokens 200 \

echo "Evaluation completed! Results saved to: $OUTPUT_DIR/command_alignment_results.json"


