# Transformer 英德翻译模型

本项目实现了一个基于 **PyTorch** 的 Transformer 神经机器翻译系统（英文 → 德文），涵盖从数据预处理到模型训练与推理的完整流程。  
项目特点包括：
- 使用自实现的 Transformer 模型结构；
- 支持标签平滑、早停策略与可视化日志；
- 提供推理脚本，可直接输入英文句子进行翻译。

---

## 环境配置

建议使用 Conda 环境以避免依赖冲突：

```bash
# 创建环境
conda create -n transformer python=3.8

# 激活环境
conda activate transformer

# 安装依赖
pip install -r requirements.txt

```
---

## 项目结构
```bash
project/
│
├── src/
│   ├── model.py          # Transformer 模型实现（含注意力机制、前馈层、位置编码等）
│   ├── dataset.py        # 数据集加载、分词与掩码处理
│   ├── train.py          # 训练脚本（含早停、日志与可视化）
│   ├── translate.py      # 翻译与推理脚本
│   └── utils.py          # 辅助函数（如固定随机种子）
│
├── results/              # 保存训练日志、模型权重与曲线图
│
└── scripts/
    └── run.sh            # 一键运行脚本
```
---

# 训练模型 
```bash
python train.py 
```
---
# 翻译示例（训练完成后可手动执行） 
```bash
python ../src/translate.py \ # --model ../results/best_transformer.pt \ # --sentence "I love you."
```
