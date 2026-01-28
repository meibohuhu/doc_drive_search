#!/bin/bash
# Download SimLingo data files
# 
# Usage:
#   ./download_training_data.sh                    # Download training files (default)
#   ./download_training_data.sh training          # Download training files
#   ./download_training_data.sh validation         # Download validation files
#   ./download_training_data.sh all               # Download both training and validation files

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

# Determine what to download based on argument
DOWNLOAD_TYPE="${1:-training}"

case "$DOWNLOAD_TYPE" in
    training)
        PATTERN="data_simlingo_*training*.tar.gz"
        INCLUDE_PATTERN="data_simlingo_*training*.tar.gz"
        FILTER_PYTHON="'training' in f and 'validation' not in f"
        TYPE_NAME="training"
        ;;
    validation)
        PATTERN="data_simlingo_validation_*.tar.gz"
        INCLUDE_PATTERN="data_simlingo_validation_*.tar.gz"
        FILTER_PYTHON="'validation' in f"
        TYPE_NAME="validation"
        ;;
    all)
        PATTERN="data_simlingo_*.tar.gz"
        INCLUDE_PATTERN="data_simlingo_*.tar.gz"
        FILTER_PYTHON="True"
        TYPE_NAME="all (training + validation)"
        ;;
    *)
        echo "Error: Unknown download type: $DOWNLOAD_TYPE"
        echo "Usage: $0 [training|validation|all]"
        exit 1
        ;;
esac

echo "=========================================="
echo "SimLingo Data Download Script"
echo "=========================================="
echo "Dataset directory: ${DATASET_DIR}"
echo "Download type: ${TYPE_NAME}"
echo "Pattern: ${PATTERN}"
echo ""

# Create directory
mkdir -p "${DATASET_DIR}"
cd "${DATASET_DIR}"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    pip install -U "huggingface_hub[cli]"
fi

# Get list of files to download
echo "Getting list of ${TYPE_NAME} files..."
python3 << PYTHON_SCRIPT
from huggingface_hub import list_repo_files
import sys

files = list_repo_files('RenzKa/simlingo', repo_type='dataset')
PYTHON_SCRIPT

# Use different Python scripts based on download type
if [ "$DOWNLOAD_TYPE" = "training" ]; then
    python3 << 'PYTHON_SCRIPT'
from huggingface_hub import list_repo_files

files = list_repo_files('RenzKa/simlingo', repo_type='dataset')
target_files = [f for f in files if f.startswith('data_simlingo_') and 'training' in f and 'validation' not in f]

print(f"Found {len(target_files)} training files:")
for f in sorted(target_files):
    print(f"  - {f}")

with open('training_files.txt', 'w') as f:
    for file in target_files:
        f.write(f"{file}\n")

print(f"\nFile list saved to training_files.txt")
PYTHON_SCRIPT
elif [ "$DOWNLOAD_TYPE" = "validation" ]; then
    python3 << 'PYTHON_SCRIPT'
from huggingface_hub import list_repo_files

files = list_repo_files('RenzKa/simlingo', repo_type='dataset')
target_files = [f for f in files if f.startswith('data_simlingo_validation_')]

print(f"Found {len(target_files)} validation files:")
for f in sorted(target_files):
    print(f"  - {f}")

with open('validation_files.txt', 'w') as f:
    for file in target_files:
        f.write(f"{file}\n")

print(f"\nFile list saved to validation_files.txt")
PYTHON_SCRIPT
else
    python3 << 'PYTHON_SCRIPT'
from huggingface_hub import list_repo_files

files = list_repo_files('RenzKa/simlingo', repo_type='dataset')
target_files = [f for f in files if f.startswith('data_simlingo_')]

print(f"Found {len(target_files)} files (training + validation):")
for f in sorted(target_files):
    print(f"  - {f}")

with open('all_files.txt', 'w') as f:
    for file in target_files:
        f.write(f"{file}\n")

print(f"\nFile list saved to all_files.txt")
PYTHON_SCRIPT
fi

# Download files using huggingface-cli
echo ""
echo "=========================================="
echo "Starting download of ${TYPE_NAME} files..."
if [ "$DOWNLOAD_TYPE" = "training" ]; then
    echo "This may take a long time (~545 GB compressed)..."
elif [ "$DOWNLOAD_TYPE" = "validation" ]; then
    echo "This may take some time..."
else
    echo "This may take a very long time..."
fi
echo "=========================================="
echo ""

# Download files matching the pattern
huggingface-cli download RenzKa/simlingo \
    --repo-type dataset \
    --include "${INCLUDE_PATTERN}" \
    --local-dir . \
    --local-dir-use-symlinks False \
    --resume-download

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ ${TYPE_NAME^} data download completed!"
    echo "=========================================="
    echo "Files downloaded to: ${DATASET_DIR}"
    echo ""
    echo "Downloaded files:"
    ls -lh ${PATTERN} 2>/dev/null | wc -l | xargs echo "  Total:"
    du -sh . | awk '{print "  Total size: " $1}'
    echo ""
    echo "Next steps:"
    echo "1. Extract files using: ./scripts_mh/extract_training_data.sh"
    echo "2. Or extract manually: for file in ${PATTERN}; do tar -xzf \"\$file\" -C extracted/; done"
else
    echo ""
    echo "=========================================="
    echo "✗ Download failed!"
    echo "=========================================="
    echo "Please check your internet connection and try again"
    exit 1
fi

