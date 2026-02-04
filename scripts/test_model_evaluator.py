#!/usr/bin/env python3
"""
测试脚本：使用ModelBasedCommandAlignmentEvaluator评估单个图像
"""

import sys
import re
from pathlib import Path

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from model_based_command_alignment_evaluator import ModelBasedCommandAlignmentEvaluator
except ImportError as e:
    print(f"[ERROR] Failed to import evaluator: {e}")
    print("\n[INFO] This might be due to missing dependencies or version conflicts.")
    print("[INFO] Please ensure you have the required packages installed:")
    print("  - transformers")
    print("  - torch")
    print("  - Pillow")
    print("  - urllib3 (compatible version)")
    print("\n[INFO] You may need to update urllib3:")
    print("  pip install --upgrade urllib3")
    sys.exit(1)

def main():
    # 配置
    image_path = "/local1/mhu/doc_drive_search/scripts/190.png"
    # command = "go straight at the next intersection in 6 meters"
    command = "go straight at the next intersection and then follow the road"
    step = 610
    
    print("=" * 80)
    print("Model-based Command Alignment Evaluator Test")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Command: {command}")
    print(f"Step: {step}")
    print("=" * 80)
    print()
    
    # 初始化评估器
    print("[INFO] Initializing evaluator...")
    try:
        evaluator = ModelBasedCommandAlignmentEvaluator(
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            device="cuda",  # 如果有GPU，使用cuda；否则会自动fallback到cpu
            max_new_tokens=100
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize evaluator: {e}")
        return
    
    # 执行评估
    print(f"\n[INFO] Evaluating image with command: '{command}'...")
    print("-" * 80)
    
    try:
        result = evaluator.evaluate_from_image_path(
            image_path=image_path,
            actual_command=command,  # 直接传入文本命令
            step=step
        )
        
        # 显示结果
        print("\n" + "=" * 80)
        print("EVALUATION RESULT")
        print("=" * 80)
        print(f"Step: {result.get('step', 'N/A')}")
        print(f"Command: {result.get('actual_command_str', 'N/A')}")
        print(f"Is Aligned: {result.get('is_aligned', False)}")
        print(f"Model Response: {result.get('model_response', 'N/A')}")
        if 'reason' in result and result.get('reason'):
            reason = result.get('reason', 'N/A')
            # 移除可能重复的"Reason:"前缀
            reason = re.sub(r'^\s*reason:\s*', '', reason, flags=re.IGNORECASE).strip()
            print(f"Reason: {reason}")
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        
        print("=" * 80)
        
        # 显示统计信息（如果有多个结果）
        stats = evaluator.get_statistics()
        if stats:
            print("\nStatistics:")
            print(f"  Total steps: {stats.get('total_steps', 0)}")
            print(f"  Aligned steps: {stats.get('aligned_steps', 0)}")
            print(f"  Overall alignment rate: {stats.get('overall_alignment_rate', 0.0):.2%}")
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

