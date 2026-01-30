
# 配置信息 - 请替换以下placeholder
GITHUB_TOKEN=""
HUGGINGFACE_TOKEN=""

# setup anaconda ###############################################################################
echo "正在下载Anaconda..."
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh

echo "正在安装Anaconda..."
# 使用静默安装，-b 表示batch mode，-p 指定安装路径
bash Anaconda3-2025.06-0-Linux-x86_64.sh -b -p $HOME/anaconda3

echo "正在初始化conda..."
# 重新加载bashrc以使conda可用
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"

echo "验证conda安装..."
conda --version


# setup github ###############################################################################
echo "正在安装GitHub CLI..."
# 根据系统安装GitHub CLI
if command -v apt-get &> /dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install gh -y
elif command -v yum &> /dev/null; then
    sudo dnf install 'dnf-command(config-manager)'
    sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
    sudo dnf install gh -y
fi

echo "正在设置SSH密钥..."
# 确保.ssh目录存在
mkdir -p ~/.ssh
cd ~/.ssh

# 生成SSH密钥，使用正确的语法避免echo参数问题
ssh-keygen -t rsa -C 'mh2803_simlingo@rit.edu' -f ~/.ssh/id_rsa -N ""

echo "正在启动ssh-agent并添加密钥..."
# 启动ssh-agent并添加密钥
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

echo "正在使用GitHub CLI登录..."
# 使用GitHub CLI登录
echo "$GITHUB_TOKEN" | gh auth login --with-token

echo "正在添加SSH密钥到GitHub..."
# 添加SSH密钥到GitHub
gh ssh-key add ~/.ssh/id_rsa.pub --title "setup-script-key"

echo "正在测试GitHub连接..."
# 测试GitHub连接，自动接受host key
ssh-keyscan -H github.com >> ~/.ssh/known_hosts
ssh -T git@github.com

# setup repo ###############################################################################
echo "正在克隆仓库..."
# 确保目录存在
mkdir -p /code
cd /code

# 检查仓库是否已存在
if [ -d "/code/doc_drive_search/.git" ]; then
    echo "仓库已存在于 /code/doc_drive_search，跳过克隆"
    cd /code/doc_drive_search
    echo "正在更新仓库..."
    git pull || echo "更新失败，请手动检查"
else
    echo "正在从 git@github.com:meibohuhu/doc_drive_search.git 克隆仓库..."
    git clone git@github.com:meibohuhu/doc_drive_search.git
    cd /code/doc_drive_search
fi

# setup simlingo conda environment ###############################################################################
echo "接受 ToS..."
$HOME/anaconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/anaconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "正在创建simlingo环境..."
# 检查环境是否已存在
if conda env list | grep -q "^simlingo "; then
    echo "环境 simlingo 已存在，跳过创建"
else
    echo "正在创建新环境 simlingo..."
    # 进入项目目录（假设simlingo项目在 //code/doc_drive_search）
    cd /code/doc_drive_search
    
    # 使用 environment_simplified.yaml 创建环境
    if [ -f "environment_simplified.yaml" ]; then
        echo "使用 environment_simplified.yaml 创建环境..."
        $HOME/anaconda3/bin/conda env create -f environment_simplified.yaml
    else
        echo "⚠️  未找到 environment_simplified.yaml，使用默认配置创建环境..."
        $HOME/anaconda3/bin/conda create -n simlingo python=3.8 -y
    fi
fi

echo "正在进入项目目录..."
cd /code/doc_drive_search

echo "正在激活simlingo环境..."
# 激活环境
conda activate simlingo

# 注意：PyTorch 2.2.0 已包含在 environment_simplified.yaml 中，无需单独安装


pip cache purge


# 设置环境变量
echo "正在设置环境变量..."
export PYTHONPATH="/home/mh2803/projects/simlingo:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# 添加到bashrc以便永久生效
if ! grep -q "export PYTHONPATH.*simlingo" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# SimLingo environment" >> ~/.bashrc
    echo "export PYTHONPATH=\"/home/mh2803/projects/simlingo:\${PYTHONPATH:-}\"" >> ~/.bashrc
    echo "export TOKENIZERS_PARALLELISM=false" >> ~/.bashrc
fi

echo "设置完成！"

# enable conda env  ############################################################################### 

# 永久初始化conda并激活base环境
echo "正在永久启用conda..."
$HOME/anaconda3/bin/conda init bash
exec bash