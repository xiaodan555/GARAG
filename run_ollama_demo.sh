#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 

# 注意 reader 参数格式： ollama-<模型关键词>
# 比如 ollama-vicuna 会匹配到 vicuna:7b
# 比如 ollama-llama3 会匹配到 llama3.1:latest

python attack.py \
    --is_genetic \
    --name="ollama_test" \
    --dataset="demo" \
    --split="test" \
    --reader="ollama-vicuna" \
    --retriever="contriever" \
    --method="typo" \
    --perturbation_level=0.2 \
    --transformations_per_example=25 \
    --is_save \
    --max_iters=5 \
    --gpus=1