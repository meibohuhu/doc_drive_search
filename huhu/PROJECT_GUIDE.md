# SimLingo 项目运行指南与结构分析

> **⚠️ 硬件要求**：运行 `setup_carla.sh` 需要满足一定的硬件要求。详细说明请参考 [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md)
> 
> **快速参考**：
> - **最低配置**：16 GB RAM, 6 GB GPU显存, 50 GB 存储空间
> - **推荐配置**：32 GB RAM, 8 GB GPU显存, 100 GB SSD
> - **训练推荐**：64 GB RAM, 10+ GB GPU显存, 200 GB NVMe SSD

## 📋 项目运行步骤

### 阶段一：环境设置

#### 1. 克隆仓库并设置CARLA
```bash
git clone git@github.com:RenzKa/simlingo.git
cd simlingo
chmod +x setup_carla.sh
./setup_carla.sh
```

#### 2. 创建Conda环境
```bash
# 创建基础环境
conda env create -f environment.yaml
conda activate simlingo

# 单独安装PyTorch（确保CUDA版本正确）
pip install torch==2.2.0

# 单独安装flash-attn
pip install flash-attn==2.7.0.post2
```

#### 3. 配置环境变量
```bash
export CARLA_ROOT=/path/to/CARLA/root
export WORK_DIR=/path/to/simlingo
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```

### 阶段二：数据集准备（可选）

#### 选项A：下载预训练数据集（推荐）
```bash
# 使用Git LFS下载完整数据集
git clone https://huggingface.co/datasets/RenzKa/simlingo
cd simlingo
git lfs pull

# 解压到指定目录
mkdir -p database/simlingo
for file in *.tar.gz; do
    tar -xzf "$file" -C database/simlingo/
done
```

#### 选项B：自行生成数据集
1. **生成驾驶数据**
   - 修改 `collect_dataset_slurm.py` 第213-230行的路径配置
   - 配置 `partition.txt`（SLURM分区）
   - 配置 `max_num_jobs.txt`（并行任务数）
   - 运行：`sbatch 0_run_collect_dataset_slurm.sh`

2. **数据集清理**
   ```bash
   python dataset_generation/delete_failed_runs.py
   python dataset_generation/delete_infraction_routes.py
   ```

3. **生成语言标签**
   - VQA标签：`python dataset_generation/language_labels/drivelm/carla_vqa_generator_main.py`
   - Commentary标签：`python dataset_generation/language_labels/commentary/carla_commentary_generator_main.py`
   - Dreamer数据：`python dataset_generation/dreamer_data/dreamer_generator.py`

### 阶段三：模型训练

#### 训练SimLingo-Base（基础模型）
- 训练代码位于：`simlingo_base_training/`
- 使用Hydra配置管理
- 检查 `config.yaml` 中的数据集路径

#### 训练SimLingo（完整模型）
```bash
# 使用SLURM脚本（集群环境）
sbatch train_simlingo_seed1.sh

# 或转换为本地bash脚本运行
python simlingo_training/train.py
```
- 训练入口：`simlingo_training/train.py`
- 配置管理：Hydra（`config.py` + `config/` 目录下的yaml文件）
- 日志：默认使用Wandb（需要登录）

### 阶段四：模型评估

#### 1. 下载预训练模型（如果只做评估）
```bash
# 从Hugging Face下载模型
# https://huggingface.co/RenzKa/simlingo
```

#### 2. 闭环驾驶评估（Bench2Drive）
```bash
# 启动评估（SLURM集群）
python start_eval_simlingo.py
# 注意：需要修改脚本中的TODO标记的配置项

# 获取结果
python Bench2Drive/tools/merge_route_json.py
```

#### 3. 语言能力评估
```bash
# 修改 simlingo_training/eval.py 中的 eval_mode
# 可选值：'QA', 'commentary', 'Dreaming'
python simlingo_training/eval.py

# 计算评估指标（需要OpenAI API key）
# 先在 simlingo_training/utils/gpt_eval.py 中配置API key
python simlingo_training/eval_metrics.py
```

---

## 📁 项目目录结构分析

### 根目录文件
```
simlingo/
├── README.md                          # 项目说明文档
├── environment.yaml                   # Conda环境配置
├── setup_carla.sh                     # CARLA设置脚本
├── train_simlingo_seed1.sh           # 训练启动脚本（SLURM）
├── collect_dataset_slurm.py          # 数据集收集脚本（SLURM）
├── start_eval_simlingo.py            # 评估启动脚本
├── partition.txt                      # SLURM分区配置
├── max_num_jobs.txt                   # 最大并行任务数配置
└── 0_run_collect_dataset_slurm.sh    # SLURM批处理脚本
```

### 核心目录结构

#### 1. **CARLA相关目录**
```
├── leaderboard/                       # CARLA评估路由文件（主要用于评估）
├── leaderboard_autopilot/            # 数据收集用的leaderboard（修改版）
├── scenario_runner/                   # CARLA场景运行器（评估用）
├── scenario_runner_autopilot/        # 数据收集用的scenario_runner（修改版）
└── Bench2Drive/                       # Bench2Drive基准测试
    ├── leaderboard/                   # Bench2Drive专用leaderboard
    ├── scenario_runner/               # Bench2Drive专用scenario_runner
    ├── tools/                         # 评估工具（merge_route_json.py等）
    └── eval.json                      # 评估配置
```

**功能说明：**
- `leaderboard_autopilot` 和 `scenario_runner_autopilot`：用于数据收集，包含PDM-Lite专家所需的额外信息
- `leaderboard` 和 `scenario_runner`：用于评估，包含路由文件
- `Bench2Drive`：独立的基准测试框架

#### 2. **训练相关目录**
```
├── simlingo_base_training/            # SimLingo-Base训练代码（无语言能力）
│   ├── config.py                      # 配置定义
│   ├── config/                        # Hydra配置文件
│   ├── models/                        # 模型定义
│   ├── dataloader/                    # 数据加载器
│   ├── callbacks/                     # 训练回调（可视化等）
│   └── utils/                         # 工具函数
│
└── simlingo_training/                 # SimLingo完整模型训练代码
    ├── train.py                       # 训练入口文件 ⭐
    ├── eval.py                        # 评估入口文件 ⭐
    ├── eval_metrics.py                # 评估指标计算
    ├── config.py                      # Hydra配置定义
    ├── config/                        # Hydra YAML配置文件
    ├── models/                        # 模型架构定义
    ├── dataloader/                    # 数据加载器
    ├── callbacks/                     # 训练回调（可视化waypoints等）
    └── utils/                         # 工具函数
        └── gpt_eval.py                # GPT评估工具（需要API key）
```

**功能说明：**
- `simlingo_base_training`：训练基础模型（CarLLaVA，无语言能力）
- `simlingo_training`：训练完整SimLingo模型（包含语言能力）
- 使用Hydra进行配置管理
- 支持Wandb日志记录
- 包含训练过程可视化回调

#### 3. **数据集生成目录**
```
dataset_generation/
├── data_buckets/                      # 数据分桶工具
│   └── carla_get_buckets.py          # 生成数据桶
│
├── language_labels/                   # 语言标签生成
│   ├── drivelm/                       # VQA标签（基于DriveLM）
│   │   └── carla_vqa_generator_main.py
│   └── commentary/                    # Commentary标签
│       └── carla_commentary_generator_main.py
│
├── dreamer_data/                      # Dreamer数据生成
│   └── dreamer_generator.py          # 生成Action Dreaming数据
│
├── get_augmentations/                 # 数据增强工具
│   ├── gpt_augment_vqa.py            # VQA数据增强（ChatGPT）
│   └── commentary_merge_augmented.py # Commentary数据合并
│
├── delete_failed_runs.py             # 删除失败的路由
├── delete_infraction_routes.py       # 删除违规路由
├── split_route_files.py              # 分割路由文件
└── split_route_files.sh              # 路由文件分割脚本
```

**功能说明：**
- 数据收集后的清理工具
- 语言标签生成（VQA、Commentary、Dreamer）
- 数据增强（使用ChatGPT）
- 路由文件处理

#### 4. **团队代码目录（Agent实现）**
```
team_code/
├── agent_simlingo.py                 # SimLingo agent实现 ⭐
├── autopilot.py                      # 自动导航
├── data_agent.py                     # 数据收集agent（DriveLM）
├── config_simlingo.py                # SimLingo配置
├── config_simlingo_base.py           # SimLingo-Base配置
├── config.py                         # 基础配置
├── nav_planner.py                    # 导航规划器
├── privileged_route_planner.py       # 特权路由规划器
├── lateral_controller.py             # 横向控制器
├── longitudinal_controller.py        # 纵向控制器
├── kinematic_bicycle_model.py        # 运动学自行车模型
├── birds_eye_view/                   # 鸟瞰图相关
├── speed_limits/                     # 速度限制
├── transfuser_utils.py               # TransFuser工具函数
├── simlingo_utils.py                 # SimLingo工具函数
├── scenario_logger.py                # 场景日志记录
└── visualize_dataset.py              # 数据集可视化
```

**功能说明：**
- 包含所有在CARLA中运行的闭环agent
- `agent_simlingo.py`：SimLingo模型的主要agent
- `data_agent.py`：数据收集时使用的agent（保存辅助信息）
- 控制器和规划器模块

#### 5. **数据目录**
```
data/
├── simlingo.zip                      # 路由文件压缩包
├── augmented_templates/              # 增强模板
│   ├── drivelm_train_augmented_v2/   # VQA增强模板
│   ├── commentary_augmented.json     # Commentary增强模板
│   └── commentary_subsentence.json   # Commentary子句级增强
├── evalset_vqa.json                  # VQA评估集
└── evalset_commentary.json           # Commentary评估集
```

**功能说明：**
- 路由文件：用于数据收集和评估
- 增强模板：训练时加载的数据增强模板
- 评估集：语言能力评估用的数据集

#### 6. **数据集存储目录（需创建）**
```
database/                              # 数据集存储目录（需手动创建）
└── simlingo/                          # 解压后的数据集
    ├── driving_data/                  # 驾驶数据
    ├── vqa_labels/                    # VQA标签
    ├── commentary_labels/             # Commentary标签
    └── dreamer_data/                  # Dreamer数据
```

---

## 🔄 典型工作流程

### 场景1：仅评估预训练模型
1. 环境设置（阶段一）
2. 下载模型（阶段四-1）
3. 运行评估（阶段四-2或四-3）

### 场景2：从头训练模型
1. 环境设置（阶段一）
2. 下载数据集（阶段二-选项A）或生成数据集（阶段二-选项B）
3. 训练模型（阶段三）
4. 评估模型（阶段四）

### 场景3：使用自己的数据集训练
1. 环境设置（阶段一）
2. 生成驾驶数据（阶段二-选项B-1）
3. 生成语言标签（阶段二-选项B-3）
4. 训练模型（阶段三）
5. 评估模型（阶段四）

---

## 📝 关键配置文件

1. **训练配置**：`simlingo_training/config/` 下的yaml文件
2. **评估配置**：`start_eval_simlingo.py`（需要修改TODO标记的部分）
3. **数据收集配置**：`collect_dataset_slurm.py`（第213-230行）
4. **SLURM配置**：`partition.txt`、`max_num_jobs.txt`

---

## ⚠️ 注意事项

1. **CARLA版本**：必须使用CARLA 0.9.15
2. **数据集路径**：训练前务必检查配置中的数据集路径
3. **环境变量**：运行前必须设置PYTHONPATH等环境变量
4. **Wandb登录**：训练需要Wandb账号登录
5. **OpenAI API**：语言评估需要OpenAI API key
6. **SLURM集群**：数据收集和评估脚本主要针对SLURM集群设计，本地运行需要修改

---

## 🔗 相关资源

- **论文**：https://arxiv.org/abs/2503.09594
- **数据集**：https://huggingface.co/datasets/RenzKa/simlingo
- **模型**：https://huggingface.co/RenzKa/simlingo
- **网站**：https://www.katrinrenz.de/simlingo/
- **视频**：https://www.youtube.com/watch?v=Mpbnz2AKaNA&t=15s

