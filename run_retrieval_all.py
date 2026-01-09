import random
import json
import os
import logging
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DenseRetrieval
# === 关键修正：引入正确的数据加载器 ===
from beir.datasets.data_loader import GenericDataLoader

# ================= 核心配置区域 (只改这里) =================

# 1. 你当前要跑的数据集名字
# 选项: "nq", "hotpotqa", "msmarco" (注意文件夹名字要和你解压的一致)
DATASET_NAME = "msmarco"  

# 2. 你的 BEIR 数据根目录 (父目录)
BEIR_ROOT_DIR = "data/beir"

# 3. 采样设置
SAMPLE_SIZE = 100
SEED = 2026

# ================= 自动生成路径 (不用改) =================
DATA_PATH = os.path.join(BEIR_ROOT_DIR, DATASET_NAME)
OUTPUT_RUN_FILE = os.path.join(DATA_PATH, f"run_contriever_{DATASET_NAME}_top100.json")
OUTPUT_SAMPLED_QIDS = os.path.join(DATA_PATH, f"sampled_{DATASET_NAME}_100_qids.json")
MODEL_NAME = "facebook/contriever"
# ========================================================

def main():
    # 设置日志
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    
    print(f"\n🚀 正在启动检索任务: [ {DATASET_NAME} ]")
    print(f"📂 数据目录: {DATA_PATH}")
    print(f"💾 输出文件将保存为: {OUTPUT_RUN_FILE}\n")

    # ---------------------------------------------------------
    # 第一步：加载 BEIR 数据 (已修正 API)
    # ---------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        logging.error(f"❌ 错误：找不到目录 {DATA_PATH}，请检查 DATASET_NAME 是否写对！")
        return

    logging.info(f"正在加载数据集: {DATASET_NAME} ...")
    
    try:
        # === 自动判断加载 test 还是 dev ===
        # 有些数据集(如MSMARCO)可能只有 dev.tsv，没有 test.tsv
        # === 智能判断加载 split ===
        split_to_load = "test"
        
        # 🔧 特殊修正：MS MARCO 必须强制用 dev，因为它的 test 集通常无效或无答案
        if DATASET_NAME == "msmarco":
            split_to_load = "dev"
            logging.info("🔧 检测到 MS MARCO，强制切换为 [dev] 集模式")
            
        # 其他数据集如果找不到 test，才回退到 dev
        elif not os.path.exists(os.path.join(DATA_PATH, "qrels", "test.tsv")):
            if os.path.exists(os.path.join(DATA_PATH, "qrels", "dev.tsv")):
                split_to_load = "dev"
            else:
                logging.warning("⚠️ 既没找到 test 也没找到 dev qrels，将尝试加载 test (可能会报错)...")
        
        # === 使用 GenericDataLoader 加载 ===
        corpus, queries, qrels = GenericDataLoader(data_folder=DATA_PATH).load(split=split_to_load)
        
    except Exception as e:
        logging.error(f"❌ 加载失败！请确认该目录下有 corpus.jsonl, queries.jsonl 和 qrels 文件夹。\n错误信息: {e}")
        return
    
    logging.info(f"  - Corpus (文档库) 大小: {len(corpus)} 条")
    logging.info(f"  - Queries (问题集) 大小: {len(queries)} 条")

    # ---------------------------------------------------------
    # 第二步：随机抽取 100 个问题
    # ---------------------------------------------------------
    logging.info(f"正在随机抽取 {SAMPLE_SIZE} 个问题...")
    random.seed(SEED)
    all_qids = list(queries.keys())
    
    # 过滤：只保留有标准答案的问题
    valid_qids = [qid for qid in all_qids if qid in qrels]
    
    if len(valid_qids) < SAMPLE_SIZE:
        logging.warning(f"⚠️ 警告：有效问题数 ({len(valid_qids)}) 少于采样数，将使用所有问题。")
        sampled_qids = valid_qids
    else:
        sampled_qids = random.sample(valid_qids, SAMPLE_SIZE)
        
    # 构建小 Queries 字典
    small_queries = {qid: queries[qid] for qid in sampled_qids}
    
    # 备份抽样的 ID
    with open(OUTPUT_SAMPLED_QIDS, 'w') as f:
        json.dump(sampled_qids, f)
    logging.info(f"  - 已锁定 {len(small_queries)} 个测试问题 (ID已备份)")

    # ---------------------------------------------------------
    # 第三步：加载 Contriever 模型
    # ---------------------------------------------------------
    logging.info(f"正在加载模型: {MODEL_NAME} ...")
    model = DenseRetrieval(models.SentenceBERT(MODEL_NAME), batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="dot")

    # ---------------------------------------------------------
    # 第四步：全库检索
    # ---------------------------------------------------------
    logging.info("🔥 开始全库检索 (Indexing Corpus)...")
    
    results = retriever.retrieve(corpus, small_queries)

    # ---------------------------------------------------------
    # 第五步：保存 Top-100 结果
    # ---------------------------------------------------------
    logging.info(f"正在保存检索结果到: {OUTPUT_RUN_FILE}")
    
    top_k_results = {}
    for qid, docs in results.items():
        # 排序并截取 Top-100
        sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)[:100]
        top_k_results[qid] = {k: v for k, v in sorted_docs}

    with open(OUTPUT_RUN_FILE, 'w') as f:
        json.dump(top_k_results, f, indent=4)
        
    logging.info(f"✅ [ {DATASET_NAME} ] 任务完成！")
    logging.info(f"结果已生成: {OUTPUT_RUN_FILE}")

if __name__ == "__main__":
    main()