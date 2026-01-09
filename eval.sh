#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export TEXTATTACK_NO_DOWNLOAD=1
export OLLAMA_HOST=127.0.0.1:11434

# 不要在这行后面加空格！
echo "🚀 开始运行 GARAG 攻击实验..."

# 注意：每行末尾的 \ 后面不能有任何空格！
python attack.py \
    --is_genetic \
    --name="nq_garag_test" \
    --dataset="nq" \
    --data_dir="data/beir/nq/nq_garag_ready.json" \
    --split="test" \
    --reader="ollama-vicuna" \
    --retriever="contriever" \
    --method="typo" \
    --perturbation_level=0.2 \
    --is_save \
    --transformations_per_example=25 \
    --max_iters=25
    # --transformations_per_example=10 \
    # --max_iters=2
    