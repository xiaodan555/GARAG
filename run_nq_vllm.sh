#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 

# 使用处理后的 NQ 数据集
DATA_PATH="./data/beir/nq/nq_garag_ready.json"

# Reader 切换为本地 HuggingFace 加载模式 (vicuna-7b)
# 这将加载 /data/longyulei/jamming_attack/cache/vicuna-7b-v1.5 下的原始权重 (FP16)
# 从而避免 Ollama 4-bit 量化带来的 ASR 虚高问题

python attack.py \
    --is_genetic \
    --name="nq_vllm_attack" \
    --task="attack" \
    --data_dir=$DATA_PATH \
    --reader="vicuna-7b" \
    --retriever="contriever" \
    --method="typo" \
    --perturbation_level=0.2 \
    --transformations_per_example=25 \
    --is_save \
    --max_iters=5 \
    --gpus=1