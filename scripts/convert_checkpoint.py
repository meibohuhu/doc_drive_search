#!/usr/bin/env python3
"""
Convert DeepSpeed checkpoint to pytorch_model.pt format
Usage:
    python convert_checkpoint.py <checkpoint_dir> <output_path>
    
Note: checkpoint_dir should be the parent directory containing 'latest' file and 'checkpoint' subdirectory
    (e.g., epoch=005.ckpt, not epoch=005.ckpt/checkpoint)
    
Example:
    python convert_checkpoint.py /path/to/epoch=005.ckpt /path/to/epoch=005.ckpt/pytorch_model.pt

    conda run -n simlingo python /home/mh2803/projects/simlingo/convert_checkpoint.py \
    /shared/rc/llm-gen-agent/mhu/simlingo_checkpoints/2026_01_30_14_47_27_sim_training_full_command_0130/checkpoints/epoch=last.ckpt \
    /shared/rc/llm-gen-agent/mhu/simlingo_checkpoints/2026_01_30_14_47_27_sim_training_full_command_0130/checkpoints/epoch=last.ckpt/pytorch_model.pt

    conda run -n simlingo python /code/doc_drive_search/scripts/convert_checkpoint.py \
    /code/doc_drive_search/pretrained/nolmdrive/checkpoints/epoch=015.ckpt \
    /code/doc_drive_search/pretrained/nolmdrive/checkpoints/epoch=015.ckpt/pytorch_model.pt

    conda run -n simlingo python /code/doc_drive_search/scripts/convert_checkpoint.py \
    /code/doc_drive_search/pretrained/withlmdrive/checkpoints/epoch=013.ckpt \
    /code/doc_drive_search/pretrained/withlmdrive/checkpoints/epoch=013.ckpt/pytorch_model.pt

    

"""

import os
import sys
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

def convert_checkpoint(checkpoint_dir, output_path):
    """
    Convert DeepSpeed checkpoint directory to a single pytorch_model.pt file
    
    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g., epoch=005.ckpt/checkpoint)
        output_path: Path where to save the converted pytorch_model.pt file
    """
    print(f"Converting checkpoint from: {checkpoint_dir}")
    print(f"Output will be saved to: {output_path}")
    
    # Check if checkpoint directory exists
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Convert DeepSpeed checkpoint to state dict
    print("Loading DeepSpeed checkpoint...")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Save as pytorch_model.pt
    print(f"Saving converted checkpoint to: {output_path}")
    torch.save(state_dict, output_path)
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024**3)  # Size in GB
    print(f"✓ Conversion complete!")
    print(f"  Output file: {output_path}")
    print(f"  File size: {file_size:.2f} GB")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <checkpoint_dir> <output_path>")
        print("\nExample:")
        print("  python convert_checkpoint.py \\")
        print("    /shared/rc/llm-gen-agent/mhu/simlingo_checkpoints/2026_01_30_14_47_27_sim_training_full_command_0130/checkpoints/epoch=005.ckpt \\")
        print("    /shared/rc/llm-gen-agent/mhu/simlingo_checkpoints/2026_01_30_14_47_27_sim_training_full_command_0130/checkpoints/epoch=005.ckpt/pytorch_model.pt")
        print("\nNote: Use the parent directory (epoch=005.ckpt), not the checkpoint subdirectory")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_checkpoint(checkpoint_dir, output_path)

