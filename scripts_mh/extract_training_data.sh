#!/bin/bash
# Extract SimLingo training data files
# Continues extraction even if target directory already has content (may overwrite existing files)
# 
# Usage:
#   ./extract_training_data.sh                                    # Extract all training files
#   ./extract_training_data.sh training                          # Extract all training files
#   ./extract_training_data.sh validation                        # Extract all validation files
#   ./extract_training_data.sh <filename>                        # Extract specific file
#   ./extract_training_data.sh <full_path_to_file>               # Extract specific file with full path
## ./scripts_mh/extract_training_data.sh /shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo/data_simlingo_training_parking_lane_Town12_short_random_weather_seed_10_balanced_150_chunk_001.tar.gz

set -e

# # Activate conda environment
# # Try different conda paths
# if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
#     source ~/miniconda3/etc/profile.d/conda.sh
# elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
#     source ~/anaconda3/etc/profile.d/conda.sh
# elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
#     source /opt/conda/etc/profile.d/conda.sh
# fi

# conda activate simlingo || {
#     echo "Warning: Could not activate simlingo environment"
# }

# Set directories
COMPRESSED_DIR="${COMPRESSED_DIR:-/mnt/localssd/simlingo}"
EXTRACTED_DIR="${EXTRACTED_DIR:-/mnt/localssd/simlingo_extracted}"

# Determine extraction mode based on arguments
EXTRACT_MODE="training"  # default: training
SPECIFIC_FILE=""

if [ $# -gt 0 ]; then
    # Check if first argument is a mode keyword or a file path
    if [ "$1" = "training" ] || [ "$1" = "validation" ]; then
        EXTRACT_MODE="$1"
    elif [ -f "$1" ] || [ -f "${COMPRESSED_DIR}/$1" ]; then
        # It's a file path
        SPECIFIC_FILE="$1"
        EXTRACT_MODE="specific"
    else
        # Assume it's a filename to search for
        SPECIFIC_FILE="$1"
        EXTRACT_MODE="specific"
    fi
fi

# Set file pattern based on mode
case "$EXTRACT_MODE" in
    training)
        FILE_PATTERN="data_simlingo_*training*.tar.gz"
        TYPE_NAME="training"
        ;;
    validation)
        FILE_PATTERN="data_simlingo_validation_*.tar.gz"
        TYPE_NAME="validation"
        ;;
    specific)
        FILE_PATTERN=""  # Not used for specific file
        TYPE_NAME="specific file"
        ;;
esac

# Display header
if [ "$EXTRACT_MODE" = "specific" ]; then
    echo "=========================================="
    echo "SimLingo Data Extraction Script"
    echo "=========================================="
    echo "Extracting specific file: ${SPECIFIC_FILE}"
    echo "Extraction target directory: ${EXTRACTED_DIR}"
    echo ""
else
    echo "=========================================="
    echo "SimLingo Data Extraction Script"
    echo "=========================================="
    echo "Extraction mode: ${TYPE_NAME}"
    echo "File pattern: ${FILE_PATTERN}"
    echo "Compressed files directory: ${COMPRESSED_DIR}"
    echo "Extraction target directory: ${EXTRACTED_DIR}"
    echo ""
fi

# Check if compressed directory exists
if [ ! -d "${COMPRESSED_DIR}" ]; then
    echo "Error: Compressed directory does not exist: ${COMPRESSED_DIR}"
    echo "Please download the data first using download_training_data.sh"
    exit 1
fi

# Check if target directory already has content (informational only)
if [ -d "${EXTRACTED_DIR}" ] && [ "$(ls -A ${EXTRACTED_DIR} 2>/dev/null)" ]; then
    echo "=========================================="
    echo "⚠️  Target directory already has content!"
    echo "=========================================="
    echo "Extraction target: ${EXTRACTED_DIR}"
    echo "Directory size: $(du -sh ${EXTRACTED_DIR} | awk '{print $1}')"
    echo "File count: $(find ${EXTRACTED_DIR} -type f | wc -l) files"
    echo ""
    echo "Continuing extraction (existing files may be overwritten)..."
    echo "=========================================="
    echo ""
fi

# Create extraction directory
mkdir -p "${EXTRACTED_DIR}"

# Navigate to compressed directory
cd "${COMPRESSED_DIR}"

# Function to extract a single file with progress tracking
extract_file() {
    local file=$1
    local counter=$2
    local total=$3
    local filename=$(basename "$file")
    if [ -n "$counter" ] && [ -n "$total" ]; then
        echo "[${counter}/${total}] Starting: ${filename}"
    else
        echo "Starting: ${filename}"
    fi
    tar -xzf "$file" -C "${EXTRACTED_DIR}/" 2>&1 | grep -v "tar: Removing leading" > /dev/null 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        if [ -n "$counter" ] && [ -n "$total" ]; then
            echo "[${counter}/${total}] ✓ Completed: ${filename}"
        else
            echo "✓ Completed: ${filename}"
        fi
    else
        if [ -n "$counter" ] && [ -n "$total" ]; then
            echo "[${counter}/${total}] ✗ Failed: ${filename} (exit code: $exit_code)"
        else
            echo "✗ Failed: ${filename} (exit code: $exit_code)"
        fi
    fi
    return $exit_code
}

# Export function and variables for parallel execution
export -f extract_file
export EXTRACTED_DIR

# If a specific file was provided, extract only that file
if [ "$EXTRACT_MODE" = "specific" ] && [ -n "${SPECIFIC_FILE}" ]; then
    # Check if file is a full path or just filename
    if [ -f "${SPECIFIC_FILE}" ]; then
        # Full path provided
        FILE_PATH="${SPECIFIC_FILE}"
        FILENAME=$(basename "${SPECIFIC_FILE}")
    elif [ -f "${COMPRESSED_DIR}/${SPECIFIC_FILE}" ]; then
        # Filename provided, construct full path
        FILE_PATH="${COMPRESSED_DIR}/${SPECIFIC_FILE}"
        FILENAME="${SPECIFIC_FILE}"
    else
        echo "Error: File not found: ${SPECIFIC_FILE}"
        echo "Searched in: ${COMPRESSED_DIR}"
        exit 1
    fi
    
    echo "Extracting: ${FILENAME}"
    extract_file "${FILE_PATH}" "" ""
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Extraction completed!"
        echo "=========================================="
        echo "Extracted data location: ${EXTRACTED_DIR}"
        echo "Total size: $(du -sh ${EXTRACTED_DIR} | awk '{print $1}')"
        echo "Total files: $(find ${EXTRACTED_DIR} -type f | wc -l)"
        echo ""
    else
        echo ""
        echo "=========================================="
        echo "✗ Extraction failed!"
        echo "=========================================="
        exit $EXIT_CODE
    fi
else
    # Extract all files matching the pattern
    # Count tar.gz files matching the pattern
    TAR_COUNT=$(ls -1 ${FILE_PATTERN} 2>/dev/null | wc -l)
    
    if [ $TAR_COUNT -eq 0 ]; then
        echo "Warning: No ${TYPE_NAME} tar.gz files found in ${COMPRESSED_DIR}"
        echo "Pattern searched: ${FILE_PATTERN}"
        echo "Please check if files were downloaded correctly"
        exit 1
    fi
    
    echo "Found ${TAR_COUNT} ${TYPE_NAME} archive files"
    echo "Starting parallel extraction..."
    echo ""
    
    # Set number of parallel processes (default: 4, can be overridden via PARALLEL_JOBS env var)
    # Adjust based on your CPU cores and I/O capacity
    PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
    echo "Using ${PARALLEL_JOBS} parallel processes"
    echo ""
    
    # Extract files in parallel using xargs
    # Format: counter|total|filepath
    COUNTER=0
    for file in ${FILE_PATTERN}; do
        COUNTER=$((COUNTER + 1))
        echo "${COUNTER}|${TAR_COUNT}|${file}"
    done | xargs -n1 -P${PARALLEL_JOBS} -I{} bash -c 'IFS="|" read -r counter total file <<< "{}"; extract_file "$file" "$counter" "$total"'
    
    echo ""
    echo "=========================================="
    echo "✓ Extraction completed!"
    echo "=========================================="
    echo "Extracted data location: ${EXTRACTED_DIR}"
    echo "Total size: $(du -sh ${EXTRACTED_DIR} | awk '{print $1}')"
    echo "Total files: $(find ${EXTRACTED_DIR} -type f | wc -l)"
    echo ""
fi
