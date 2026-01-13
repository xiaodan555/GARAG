#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export TEXTATTACK_NO_DOWNLOAD=1
export OLLAMA_HOST=127.0.0.1:11434
# export GARAG_DEBUG_LIMIT=10 跑10个样本测试
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

mkdir -p logs
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="logs/eval_ollama_attack_${timestamp}.log"
echo "Output redirected to $log_file"
echo "You can view progress with: tail -f $log_file"

# 不要在这行后面加空格！
echo "🚀 开始运行 GARAG 攻击实验..."

# 注意：每行末尾的 \ 后面不能有任何空格！
python attack.py \
    --is_genetic \
    --name="nq_ollama_attack" \
    --dataset="nq" \
    --data_dir="data/beir/nq/nq_garag_ready.json" \
    --split="test" \
    --reader="ollama-llama3.2:latest" \
    --retriever="contriever" \
    --method="typo" \
    --perturbation_level=0.2 \
    --is_save \
    --transformations_per_example=25 \
    --max_iters=25 > "$log_file" 2>&1
    # --transformations_per_example=10 \
    # --max_iters=2
    