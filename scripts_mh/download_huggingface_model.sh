#!/bin/bash
# Download SimLingo model from Hugging Face: RenzKa/simlingo

MODEL_NAME="RenzKa/simlingo"
SAVE_DIR="${1:-/code/doc_drive_search/data/pretrained_models/simlingo}"

echo "Downloading ${MODEL_NAME} to ${SAVE_DIR}..."

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL_NAME}',
    local_dir='${SAVE_DIR}',
    local_dir_use_symlinks=False,
    resume_download=True
)
"

echo "✅ Download completed: ${SAVE_DIR}"