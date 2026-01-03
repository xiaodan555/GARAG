import json
import random
import os

# === 配置区域 ===
# 假设你下载的标准大数据集文件路径如下（你需要先下载好）：
# 建议去 https://github.com/facebookresearch/DPR/tree/main/data/retriever 下载
SOURCE_FILES = {
    "nq": "data/ODQA/contriever/nq-test_full.json",          # 3610条的标准版
    # "hotpotqa": "data/ODQA/contriever/hotpotqa-test_full.json", # 如果你有的话
    # "msmarco": "data/ODQA/contriever/msmarco-test_full.json"    # 如果你有的话
}

OUTPUT_DIR = "data/ODQA/contriever"
SAMPLE_SIZE = 100  # 你的实验设计要求：随机攻击100条
SEED = 2026        # 固定随机种子，保证你的Baseline是可以复现的

def sample_dataset(dataset_name, input_path):
    print(f"正在处理 {dataset_name} ...")
    
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到文件 {input_path}。请先下载标准数据集！")
        return

    with open(input_path, 'r') as f:
        full_data = json.load(f)
    
    total_len = len(full_data)
    print(f"  - 原始数据集大小: {total_len} 条")
    
    # 随机采样
    random.seed(SEED)
    if total_len > SAMPLE_SIZE:
        sampled_data = random.sample(full_data, SAMPLE_SIZE)
        print(f"  - ✅ 已随机抽取 {SAMPLE_SIZE} 条")
    else:
        sampled_data = full_data
        print(f"  - ⚠️ 数据量不足 {SAMPLE_SIZE}，取全量。")

    # 保存
    output_filename = f"{dataset_name}-random{SAMPLE_SIZE}_100.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(sampled_data, f)
    
    print(f"  - 💾 已保存至: {output_path}")
    print(f"  - 💡 接下来请修改 eval.sh: --dataset='{dataset_name}-random{SAMPLE_SIZE}'")

if __name__ == "__main__":
    for name, path in SOURCE_FILES.items():
        sample_dataset(name, path)