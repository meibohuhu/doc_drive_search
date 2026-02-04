#!/bin/bash

# Setup QwenVL Conda Environment
# 此脚本用于创建和配置 qwenvl conda 环境

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda 未找到。请先安装 Anaconda 或 Miniconda。"
    echo "如果 conda 已安装但未在 PATH 中，请使用完整路径，例如："
    echo "  \$HOME/anaconda3/bin/conda"
    exit 1
fi

# 获取 conda 路径（如果使用完整路径）
CONDA_CMD="${CONDA_CMD:-conda}"

# setup qwenvl conda environment ###############################################################################
echo "接受 ToS..."
$CONDA_CMD tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$CONDA_CMD tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "正在创建qwenvl环境..."
# 检查环境是否已存在
if $CONDA_CMD env list | grep -q "^qwenvl "; then
    echo "环境 qwenvl 已存在，跳过创建"
    echo "如需重新创建，请先删除：conda env remove -n qwenvl"
else
    echo "正在创建新环境 qwenvl..."
    $CONDA_CMD create -n qwenvl python=3.10 -y
fi

echo "正在激活qwenvl环境并安装包..."
# 激活环境并安装包
# 注意：在脚本中激活 conda 环境需要使用 source activate 或 conda activate
# 如果从脚本运行，建议使用：source $(conda info --base)/etc/profile.d/conda.sh && conda activate qwenvl
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || eval "$($CONDA_CMD shell.bash hook)"
conda activate qwenvl

############ 安装包
echo "正在安装构建依赖（psutil, packaging, ninja）..."
pip install psutil packaging ninja

echo "正在安装PyTorch with CUDA 12.1..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "正在安装flash-attn..."
if pip install flash-attn==2.5.7 --no-build-isolation; then
    echo "✅ flash-attn 安装成功"
else
    echo "⚠️  第一次安装失败，尝试使用 --no-deps 选项..."
    pip install flash-attn==2.5.7 --no-build-isolation --no-deps || echo "⚠️  flash-attn 安装失败，但可以继续（某些功能可能较慢）"
fi

# 如果提供了 requirements 文件路径，则安装
if [ -n "$REQUIREMENTS_FILE" ] && [ -f "$REQUIREMENTS_FILE" ]; then
    echo "正在使用 $REQUIREMENTS_FILE 安装其他依赖..."
    pip install -r "$REQUIREMENTS_FILE"
elif [ -f "requirements_qwenvl.txt" ]; then
    echo "正在使用当前目录的 requirements_qwenvl.txt 安装其他依赖..."
    pip install -r requirements_qwenvl.txt
else
    echo "⚠️  未找到 requirements 文件，跳过额外依赖安装"
    echo "   如需安装，请设置 REQUIREMENTS_FILE 环境变量或确保 requirements_qwenvl.txt 在当前目录"
fi

echo "正在清理 pip 缓存..."
pip cache purge

echo ""
echo "✅ qwenvl 环境设置完成！"
echo ""
echo "使用方法："
echo "  conda activate qwenvl"
echo ""
echo "或者如果 conda activate 不可用："
echo "  source \$(conda info --base)/etc/profile.d/conda.sh"
echo "  conda activate qwenvl"

