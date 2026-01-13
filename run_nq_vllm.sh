#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 

# 使用处理后的 NQ 数据集
DATA_PATH="./data/beir/nq/nq_garag_ready.json"

# Reader 切换为本地 vLLM 加载模式 (Llama-3.2-1B-Instruct)
# 这将加载 /data/longyulei/jamming_attack/cache/Llama-3.2-1B-Instruct 下的原始权重
# 从而避免 Ollama 4-bit 量化带来的 ASR 虚高问题

mkdir -p logs
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="logs/nq_vllm_attack_${timestamp}.log"
echo "Output redirected to $log_file"
echo "You can view progress with: tail -f $log_file"

python attack.py \
    --is_genetic \
    --name="nq_vllm_attack" \
    --task="attack" \
    --data_dir=$DATA_PATH \
    --reader="Llama-3.2-1B-Instruct" \
    --is_vllm \
    --retriever="contriever" \
    --method="typo" \
    --perturbation_level=0.2 \
    --transformations_per_example=25 \
    --is_save \
    --max_iters=5 \
    --gpus=1 > "$log_file" 2>&1