# SimLingo 项目安装指南

## ✅ 安装状态

- ✅ Conda 环境已创建（simlingo, Python 3.8.20）
- ✅ Python 依赖包已安装（PyTorch 2.2.0, flash-attn 2.7.0.post2 等）
- ✅ CARLA 0.9.15 已安装到 `/local1/mhu/software/carla0915`
- ✅ CARLA Python API 可正常导入
- ✅ 环境变量已配置

---

## 📋 快速开始

### 1. 激活环境并加载环境变量

```bash
conda activate simlingo
source /local1/mhu/doc_drive_search/carla_exports.sh
```

### 2. 验证安装

```bash
cd /local1/mhu/doc_drive_search
bash verify_installation.sh
```

---

## 🔧 关键配置

### 环境变量（已配置在 `carla_exports.sh`）

```bash
export CARLA_ROOT=/local1/mhu/software/carla0915
export WORK_DIR=/local1/mhu/doc_drive_search
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}:${CARLA_ROOT}/PythonAPI/carla:...
```

### 启动 CARLA 服务器（如需要）

```bash
/local1/mhu/software/carla0915/CarlaUE4.sh
```

---

## 📁 重要文件

- `environment_simplified.yaml` - Conda 环境配置（已使用）
- `carla_exports.sh` - 环境变量配置（已更新路径）
- `verify_installation.sh` - 安装验证脚本

---

## 🆘 常见问题

**Q: CARLA Python API 导入失败？**  
A: 确保已加载环境变量：`source carla_exports.sh`

**Q: conda 环境创建失败？**  
A: 使用 `environment_simplified.yaml` 而非 `environment.yaml`

**Q: flash-attn 安装失败？**  
A: 确保 CUDA 开发工具已安装，编译需要较长时间

---

**最后更新**：2025-01-18  
**状态**：✅ 安装完成，可正常使用
