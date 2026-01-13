#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export TEXTATTACK_NO_DOWNLOAD=1
export OLLAMA_HOST=127.0.0.1:11434

echo "🚀 开始运行 GARAG 攻击实验 (TEST RUN)..."

python attack.py \
    --is_genetic \
    --name="nq_garag_test_run" \
    --dataset="nq" \
    --data_dir="data/beir/nq/nq_garag_ready.json" \
    --split="test" \
    --reader="ollama-qwen3:8b" \
    --retriever="contriever" \
    --method="typo" \
    --perturbation_level=0.1 \
    --is_save \
    --transformations_per_example=2 \
    --max_iters=1
