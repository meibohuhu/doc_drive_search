#!/bin/bash
# SimLingo数据集下载脚本（优化版）
# 使用Hugging Face CLI，比git lfs更快

set -e

# 激活conda环境
# 尝试不同的conda路径
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

conda activate simlingo || {
    echo "警告: 无法激活simlingo环境，尝试直接使用pip安装的包"
}

# 设置数据集目录
DATASET_DIR="/shared/rc/llm-gen-agent/mhu/simlingo_dataset/database"
EXTRACTED_DIR="${DATASET_DIR}/simlingo_extracted"

echo "=========================================="
echo "SimLingo 数据集下载脚本"
echo "=========================================="
echo "数据集目录: ${DATASET_DIR}"
echo "解压目录: ${EXTRACTED_DIR}"
echo ""

# 创建目录
mkdir -p "${DATASET_DIR}"
cd "${DATASET_DIR}"

# 检查是否已安装huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "安装 huggingface-cli..."
    pip install -U "huggingface_hub[cli]"
fi

# 方法1: 使用huggingface-cli下载（推荐，更快）
echo "使用 huggingface-cli 下载数据集..."
echo "这可能需要较长时间，请耐心等待..."
echo ""

huggingface-cli download RenzKa/simlingo \
    --repo-type dataset \
    --local-dir simlingo \
    --local-dir-use-symlinks False

# 如果上面的命令失败，尝试方法2
if [ $? -ne 0 ]; then
    echo ""
    echo "huggingface-cli下载失败，尝试使用git lfs..."
    echo ""
    
    # 方法2: 使用git lfs（备用方案）
    if [ ! -d "simlingo" ]; then
        git clone https://huggingface.co/datasets/RenzKa/simlingo
    fi
    
    cd simlingo
    git lfs pull
    cd ..
fi

# 解压数据
echo ""
echo "=========================================="
echo "开始解压数据集..."
echo "=========================================="

mkdir -p "${EXTRACTED_DIR}"
cd simlingo

# 统计tar.gz文件数量
TAR_COUNT=$(ls -1 *.tar.gz 2>/dev/null | wc -l)
if [ $TAR_COUNT -eq 0 ]; then
    echo "警告: 未找到tar.gz文件，可能下载未完成或文件格式不同"
    echo "请检查下载是否成功"
    exit 1
fi

echo "找到 ${TAR_COUNT} 个压缩文件"
echo ""

# 解压所有tar.gz文件
COUNTER=0
for file in *.tar.gz; do
    COUNTER=$((COUNTER + 1))
    echo "[${COUNTER}/${TAR_COUNT}] 正在解压: $file"
    tar -xzf "$file" -C "${EXTRACTED_DIR}/" 2>&1 | grep -v "tar: Removing leading" || true
    echo "  ✓ 完成"
done

echo ""
echo "=========================================="
echo "✓ 数据集下载和解压完成！"
echo "=========================================="
echo "数据集位置: ${EXTRACTED_DIR}"
echo ""
echo "下一步："
echo "1. 检查数据集结构: ls -lh ${EXTRACTED_DIR}"
echo "2. 在训练配置中设置 data_path: ${EXTRACTED_DIR}"

