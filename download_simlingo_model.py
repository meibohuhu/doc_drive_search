#!/usr/bin/env python3
"""
Script to download the RenzKa/simlingo model from Hugging Face
pip install huggingface-hub
python download_simlingo.py --local-dir /code/doc_drive_search/pretrained

/code/doc_drive_search/pretrained

"""

from huggingface_hub import snapshot_download
import os

def download_simlingo_model(local_dir="./models/simlingo"):
    """
    Download the RenzKa/simlingo model from Hugging Face

    Args:
        local_dir: Local directory path where the model will be saved
    """
    model_id = "RenzKa/simlingo"

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

    parser = argparse.ArgumentParser(description="Download RenzKa/simlingo model from Hugging Face")
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./models/simlingo",
        help="Local directory to save the model (default: ./models/simlingo)"
    )

    args = parser.parse_args()
    download_simlingo_model(args.local_dir)
