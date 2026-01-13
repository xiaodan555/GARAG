import random
import json
import os
import logging
from beir import util, LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DenseRetrieval
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union
import numpy as np

# === Custom SentenceBERT Wrapper to bypass beir.retrieval.models import issues ===
class SentenceBERT:
    def __init__(self, model_path: str, sep: str = " ", **kwargs):
        self.sep = sep
        self.q_model = SentenceTransformer(model_path)
        self.doc_model = self.q_model
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)

# ================= 核心配置区域 (只改这里) =================

# 1. 你当前要跑的数据集名字列表
# 选项: "nq", "hotpotqa", "msmarco" (注意文件夹名字要和你解压的一致)
DATASETS = ["nq", "hotpotqa", "msmarco"]

# 2. 你的 BEIR 数据根目录 (父目录)
BEIR_ROOT_DIR = "data/beir"

# 3. 采样设置
SAMPLE_SIZE = 100
SEED = 2026

# ================= 自动生成路径 (不用改) =================
MODEL_NAME = "facebook/contriever"
# ========================================================

def process_dataset(dataset_name, retriever):
    data_path = os.path.join(BEIR_ROOT_DIR, dataset_name)
    output_run_file = os.path.join(data_path, f"run_contriever_{dataset_name}_top100.json")
    output_sampled_qids = os.path.join(data_path, f"sampled_{dataset_name}_100_qids.json")

    print(f"\n🚀 正在启动检索任务: [ {dataset_name} ]")
    print(f"📂 数据目录: {data_path}")
    print(f"💾 输出文件将保存为: {output_run_file}\n")

    # ---------------------------------------------------------
    # 第一步：加载 BEIR 数据 (已修正 API)
    # ---------------------------------------------------------
    if not os.path.exists(data_path):
        logging.error(f"❌ 错误：找不到目录 {data_path}，跳过该数据集！")
        return

    logging.info(f"正在加载数据集: {dataset_name} ...")
    
    try:
        # === 自动判断加载 test 还是 dev ===
        # 有些数据集(如MSMARCO)可能只有 dev.tsv，没有 test.tsv
        # === 智能判断加载 split ===
        split_to_load = "test"
        
        # 🔧 特殊修正：MS MARCO 必须强制用 dev，因为它的 test 集通常无效或无答案
        if dataset_name == "msmarco":
            split_to_load = "dev"
            logging.info("🔧 检测到 MS MARCO，强制切换为 [dev] 集模式")
            
        # 其他数据集如果找不到 test，才回退到 dev
        elif not os.path.exists(os.path.join(data_path, "qrels", "test.tsv")):
            if os.path.exists(os.path.join(data_path, "qrels", "dev.tsv")):
                split_to_load = "dev"
            else:
                logging.warning("⚠️ 既没找到 test 也没找到 dev qrels，将尝试加载 test (可能会报错)...")
        
        # === 使用 GenericDataLoader 加载 ===
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split_to_load)
        
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
    with open(output_sampled_qids, 'w') as f:
        json.dump(sampled_qids, f)
    logging.info(f"  - 已锁定 {len(small_queries)} 个测试问题 (ID已备份)")

    # ---------------------------------------------------------
    # 第三步：全库检索 (模型已在外部加载)
    # ---------------------------------------------------------
    logging.info("🔥 开始全库检索 (Indexing Corpus)...")
    
    results = retriever.retrieve(corpus, small_queries)

    # ---------------------------------------------------------
    # 第四步：保存 Top-100 结果
    # ---------------------------------------------------------
    logging.info(f"正在保存检索结果到: {output_run_file}")
    
    top_k_results = {}
    for qid, docs in results.items():
        # 排序并截取 Top-100
        sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)[:100]
        top_k_results[qid] = {k: v for k, v in sorted_docs}

    with open(output_run_file, 'w') as f:
        json.dump(top_k_results, f, indent=4)
        
    logging.info(f"✅ [ {dataset_name} ] 任务完成！")
    logging.info(f"结果已生成: {output_run_file}")

def main():
    # 设置日志
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    
    # ---------------------------------------------------------
    # 加载 Contriever 模型 (只加载一次)
    # ---------------------------------------------------------
    logging.info(f"正在加载模型: {MODEL_NAME} ...")
    model = DenseRetrieval(SentenceBERT(MODEL_NAME), batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="dot")

    # ---------------------------------------------------------
    # 循环处理每个数据集
    # ---------------------------------------------------------
    for dataset_name in DATASETS:
        process_dataset(dataset_name, retriever)
        
    logging.info("🎉 所有任务全部完成！")

if __name__ == "__main__":
    main()