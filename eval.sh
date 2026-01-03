#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export TEXTATTACK_NO_DOWNLOAD=1

# 1. 端口指向共用服务 (11434)
export OLLAMA_HOST=127.0.0.1:11434

# 2. 【关键修改】禁用 Python 脚本的 GPU，强制使用 CPU
# 这样可以避开 Contriever 模型的崩溃，而 Ollama 服务依然会在后台用显卡加速
export CUDA_VISIBLE_DEVICES="" 

python attack.py \
    --is_genetic \
    --name="ollama_demo" \
    --dataset="demo" \
    --split="test" \
    --reader="ollama-vicuna" \
    --retriever="contriever" \
    --method="typo" \
    --perturbation_level=0.2 \
    --transformations_per_example=25 \
    --is_save \
    --max_iters=25