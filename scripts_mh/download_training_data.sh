#!/bin/bash
# Download SimLingo training data files only (data_simlingo_*training*)
# Excludes validation files

### cd /home/mh2803/projects/simlingo && bash download_training_data.sh

set -e

# Activate conda environment
# Try different conda paths
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

conda activate simlingo || {
    echo "Warning: Could not activate simlingo environment, trying to use pip installed packages"
}

# Set dataset directory
DATASET_DIR="${DATASET_DIR:-/shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo}"

echo "=========================================="
echo "SimLingo Training Data Download Script"
echo "=========================================="
echo "Dataset directory: ${DATASET_DIR}"
echo "Downloading only training files (data_simlingo_*training*)"
echo ""

# Create directory
mkdir -p "${DATASET_DIR}"
cd "${DATASET_DIR}"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    pip install -U "huggingface_hub[cli]"
fi

# Get list of training files
echo "Getting list of training files..."
python3 << 'PYTHON_SCRIPT'
from huggingface_hub import list_repo_files
import sys

files = list_repo_files('RenzKa/simlingo', repo_type='dataset')
training_files = [f for f in files if f.startswith('data_simlingo_') and 'training' in f and 'validation' not in f]

print(f"Found {len(training_files)} training files:")
for f in sorted(training_files):
    print(f"  - {f}")

# Write to file for download script
with open('training_files.txt', 'w') as f:
    for file in training_files:
        f.write(f"{file}\n")

print(f"\nFile list saved to training_files.txt")
PYTHON_SCRIPT

# Download training files using huggingface-cli
echo ""
echo "=========================================="
echo "Starting download of training files..."
echo "This may take a long time (~545 GB compressed)..."
echo "=========================================="
echo ""

# Download files matching the pattern
huggingface-cli download RenzKa/simlingo \
    --repo-type dataset \
    --include "data_simlingo_*training*.tar.gz" \
    --local-dir . \
    --local-dir-use-symlinks False \
    --resume-download

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training data download completed!"
    echo "=========================================="
    echo "Files downloaded to: ${DATASET_DIR}"
    echo ""
    echo "Downloaded files:"
    ls -lh data_simlingo_*training*.tar.gz 2>/dev/null | wc -l | xargs echo "  Total:"
    du -sh . | awk '{print "  Total size: " $1}'
    echo ""
    echo "Next steps:"
    echo "1. Extract files: for file in data_simlingo_*training*.tar.gz; do tar -xzf \"\$file\" -C extracted/; done"
    echo "2. Or use the extraction script if available"
else
    echo ""
    echo "=========================================="
    echo "✗ Download failed!"
    echo "=========================================="
    echo "Please check your internet connection and try again"
    exit 1
fi

