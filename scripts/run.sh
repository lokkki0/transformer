#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 创建结果目录
mkdir -p ../results

# 训练模型
python train.py 

# 翻译示例（训练完成后可手动执行）
# echo ">>> 测试翻译示例..."
# python ../src/translate.py \
#     --model ../results/best_transformer.pt \
#     --sentence "I love natural language processing."
