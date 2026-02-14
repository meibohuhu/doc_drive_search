#!/usr/bin/env python3
"""
Script to download models from Hugging Face
pip install huggingface-hub
python download_simlingo_model.py --model-id PhoenixHu/2026_02_01_00_01_39_adobe_training_full_command_0130_nolmdrive --local-dir /code/doc_drive_search/pretrained/nolmdrive
python download_simlingo_model.py --model-id RenzKa/simlingo --local-dir /code/doc_drive_search/pretrained/simlingo
python download_simlingo_model.py --model-id PhoenixHu/2026_01_30_14_47_27_sim_training_full_command_0130 --local-dir /code/doc_drive_search/pretrained/withlmdrive

/code/doc_drive_search/pretrained

"""

from huggingface_hub import snapshot_download
import os

def download_simlingo_model(local_dir="./models/simlingo", model_id="RenzKa/simlingo"):
    """
    Download a model from Hugging Face

    Args:
        local_dir: Local directory path where the model will be saved
        model_id: HuggingFace model repository ID
    """

    print(f"Downloading {model_id} to {local_dir}...")

    try:
        # Create the directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)

        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"\nModel successfully downloaded to: {os.path.abspath(local_dir)}")

    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nMake sure you have huggingface_hub installed:")
        print("  pip install huggingface-hub")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument(
        "--model-id",
        type=str,
        default="RenzKa/simlingo",
        help="HuggingFace model repository ID (default: RenzKa/simlingo)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./models/simlingo",
        help="Local directory to save the model (default: ./models/simlingo)"
    )

    args = parser.parse_args()
    download_simlingo_model(args.local_dir, args.model_id)
