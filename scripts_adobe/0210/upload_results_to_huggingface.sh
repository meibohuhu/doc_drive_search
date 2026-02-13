#!/bin/bash
# Script to package and upload evaluation results to Hugging Face

# Configuration
HF_REPO="${1:-your-username/your-repo-name}"  # Pass as first argument or set here
HF_TOKEN="${2:-}"  # Pass as second argument or set here, or use: export HF_TOKEN=your_token
RESULTS_BASE_DIR="/code/doc_drive_search/eval_results/agent_simlingo_cfg/simlingo/bench2drive"
TEMP_DIR="/tmp/eval_results_upload"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================="
echo "Evaluation Results Upload to HuggingFace"
echo "========================================="
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}Error: huggingface-cli not found${NC}"
    echo "Please install it with: pip install huggingface_hub"
    exit 1
fi

# Check authentication
if [ -z "$HF_TOKEN" ]; then
    # No token provided, check if logged in
    if ! huggingface-cli whoami &> /dev/null; then
        echo -e "${RED}Error: Not authenticated with HuggingFace${NC}"
        echo ""
        echo "Please either:"
        echo "  1. Provide token as argument: $0 <repo-name> <token>"
        echo "  2. Set environment variable: export HF_TOKEN=your_token"
        echo "  3. Login with: huggingface-cli login"
        exit 1
    else
        echo -e "${GREEN}✓ Authenticated via huggingface-cli login${NC}"
    fi
else
    echo -e "${GREEN}✓ Using provided HuggingFace token${NC}"
fi

# Validate repo name
if [[ "$HF_REPO" == "your-username/your-repo-name" ]]; then
    echo -e "${RED}Error: Please provide a valid HuggingFace repository name${NC}"
    echo "Usage: $0 <repo-name> [token]"
    echo "Example: $0 username/repo-name hf_xxxxx"
    exit 1
fi

echo "Repository: $HF_REPO"
echo ""

# Create temporary directory
echo "Creating temporary directory..."
mkdir -p "$TEMP_DIR"
rm -f "$TEMP_DIR"/*.tar.gz 2>/dev/null

# Package each folder (0-7)
echo ""
echo "Packaging folders..."
TOTAL_SIZE=0

for i in {0..7}; do
    FOLDER="$RESULTS_BASE_DIR/$i"
    ARCHIVE_NAME="bench2drive_results_${i}_${TIMESTAMP}.tar.gz"

    echo -e "${YELLOW}Packaging folder $i...${NC}"

    if [ ! -d "$FOLDER" ]; then
        echo -e "${RED}  Error: Folder not found: $FOLDER${NC}"
        continue
    fi

    # Show folder size before packaging
    FOLDER_SIZE=$(du -sh "$FOLDER" | cut -f1)
    echo "  Folder size: $FOLDER_SIZE"

    # Create tar.gz archive
    tar -czf "$TEMP_DIR/$ARCHIVE_NAME" -C "$(dirname $FOLDER)" "$(basename $FOLDER)" 2>/dev/null

    if [ $? -eq 0 ]; then
        ARCHIVE_SIZE=$(du -h "$TEMP_DIR/$ARCHIVE_NAME" | cut -f1)
        echo -e "${GREEN}  ✓ Created: $ARCHIVE_NAME ($ARCHIVE_SIZE)${NC}"

        # Add to total size
        SIZE_BYTES=$(stat -c%s "$TEMP_DIR/$ARCHIVE_NAME")
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE_BYTES))
    else
        echo -e "${RED}  ✗ Failed to create archive${NC}"
    fi
    echo ""
done

# Convert total size to human readable
TOTAL_SIZE_HR=$(numfmt --to=iec-i --suffix=B "$TOTAL_SIZE" 2>/dev/null || echo "$TOTAL_SIZE bytes")

echo "========================================="
echo "Summary:"
echo "  Archives created: $(ls -1 "$TEMP_DIR"/*.tar.gz 2>/dev/null | wc -l)"
echo "  Total size: $TOTAL_SIZE_HR"
echo "  Location: $TEMP_DIR"
echo ""
ls -lh "$TEMP_DIR"/*.tar.gz
echo "========================================="
echo ""

# Upload to HuggingFace
read -p "Do you want to upload to HuggingFace now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}Uploading to HuggingFace...${NC}"
    echo "Repository: $HF_REPO"
    echo ""

    UPLOAD_COUNT=0
    FAIL_COUNT=0

    for ARCHIVE in "$TEMP_DIR"/*.tar.gz; do
        ARCHIVE_NAME=$(basename "$ARCHIVE")
        echo -e "${YELLOW}Uploading: $ARCHIVE_NAME${NC}"

        # Upload using huggingface-cli (with token if provided)
        if [ -n "$HF_TOKEN" ]; then
            huggingface-cli upload "$HF_REPO" "$ARCHIVE" "eval_results/$ARCHIVE_NAME" --token "$HF_TOKEN"
        else
            huggingface-cli upload "$HF_REPO" "$ARCHIVE" "eval_results/$ARCHIVE_NAME"
        fi

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✓ Uploaded successfully${NC}"
            UPLOAD_COUNT=$((UPLOAD_COUNT + 1))
        else
            echo -e "${RED}  ✗ Upload failed${NC}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        echo ""
    done

    echo "========================================="
    echo "Upload Summary:"
    echo "  Successful: $UPLOAD_COUNT"
    echo "  Failed: $FAIL_COUNT"
    echo "========================================="

    if [ $FAIL_COUNT -eq 0 ]; then
        echo -e "${GREEN}All uploads completed successfully!${NC}"

        # Ask if user wants to delete local archives
        echo ""
        read -p "Delete local archives? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$TEMP_DIR"
            echo "Local archives deleted."
        fi
    else
        echo -e "${RED}Some uploads failed. Local archives kept in: $TEMP_DIR${NC}"
    fi
else
    echo ""
    echo "Upload cancelled. Archives are saved in: $TEMP_DIR"
    echo ""
    echo "To upload manually later:"
    echo "  huggingface-cli upload $HF_REPO <archive_file> eval_results/<archive_name>"
fi

echo ""
echo "========================================="
echo "Script finished at $(date)"
echo "========================================="
