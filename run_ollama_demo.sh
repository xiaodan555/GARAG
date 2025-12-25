#!/bin/bash
# GARAG Demo with Ollama and Contriever

# Ensure we are in the GARAG environment
# source activate GARAG 

# Set visible devices if needed, though Ollama runs as a service
export CUDA_VISIBLE_DEVICES=0 

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
    --max_iters=5 \
    --model_dir="../models" \
    --gpus=1
