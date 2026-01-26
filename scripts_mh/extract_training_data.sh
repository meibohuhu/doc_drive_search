#!/bin/bash
# Extract SimLingo training data files
# Continues extraction even if target directory already has content (may overwrite existing files)

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
    echo "Warning: Could not activate simlingo environment"
}

# Set directories
COMPRESSED_DIR="${COMPRESSED_DIR:-/shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo}"
EXTRACTED_DIR="${EXTRACTED_DIR:-/shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo_extracted}"

echo "=========================================="
echo "SimLingo Training Data Extraction Script"
echo "=========================================="
echo "Compressed files directory: ${COMPRESSED_DIR}"
echo "Extraction target directory: ${EXTRACTED_DIR}"
echo ""

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

# Count tar.gz files
TAR_COUNT=$(ls -1 data_simlingo_*training*.tar.gz 2>/dev/null | wc -l)

if [ $TAR_COUNT -eq 0 ]; then
    echo "Warning: No training data tar.gz files found in ${COMPRESSED_DIR}"
    echo "Please check if files were downloaded correctly"
    exit 1
fi

echo "Found ${TAR_COUNT} training data archive files"
echo "Starting parallel extraction..."
echo ""

# Set number of parallel processes (default: 4, can be overridden via PARALLEL_JOBS env var)
# Adjust based on your CPU cores and I/O capacity
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
echo "Using ${PARALLEL_JOBS} parallel processes"
echo ""

# Function to extract a single file with progress tracking
extract_file() {
    local file=$1
    local counter=$2
    local total=$3
    local filename=$(basename "$file")
    echo "[${counter}/${total}] Starting: ${filename}"
    tar -xzf "$file" -C "${EXTRACTED_DIR}/" 2>&1 | grep -v "tar: Removing leading" > /dev/null 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[${counter}/${total}] ✓ Completed: ${filename}"
    else
        echo "[${counter}/${total}] ✗ Failed: ${filename} (exit code: $exit_code)"
    fi
    return $exit_code
}

# Export function and variables for parallel execution
export -f extract_file
export EXTRACTED_DIR

# Extract files in parallel using xargs
# Format: counter|total|filepath
COUNTER=0
for file in data_simlingo_*training*.tar.gz; do
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

