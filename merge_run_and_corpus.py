import json
import os
import csv
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 数据集名称 (跑哪个改哪个: "nq", "hotpotqa", "msmarco")
DATASET_NAME = "msmarco"

# 2. 你的 BEIR 根目录
BEIR_ROOT = "data/beir"

# 3. 自动生成的文件路径 (对应你之前跑出来的文件名)
BASE_PATH = os.path.join(BEIR_ROOT, DATASET_NAME)
RUN_FILE = os.path.join(BASE_PATH, f"run_contriever_{DATASET_NAME}_top100.json")
OUTPUT_FILE = os.path.join(BASE_PATH, f"{DATASET_NAME}_garag_ready.json")

# ===========================================

def load_qrels(path):
    """加载标准答案映射 (QID -> List[DocID])"""
    qrels = {}
    if not os.path.exists(path):
        return qrels
    print(f"   - 正在加载 Qrels: {path}")
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None) # 跳过表头
        for row in reader:
            qid, doc_id = row[0], row[1]
            if qid not in qrels: qrels[qid] = []
            qrels[qid].append(doc_id)
    return qrels

def main():
    print(f"🚀 开始拼接数据集: {DATASET_NAME}")
    print(f"📂 读取 Run File: {RUN_FILE}")
    
    if not os.path.exists(RUN_FILE):
        print(f"❌ 错误：找不到 Run File！请确认你是否运行了检索脚本。")
        return

    # 1. 加载 Run File (检索结果)
    with open(RUN_FILE, 'r') as f:
        run_data = json.load(f)
    
    target_qids = list(run_data.keys())
    print(f"   - 包含 {len(target_qids)} 个问题")

    # 2. 收集所有需要提取的文档 ID (包括 Top-100 和 标准答案文档)
    # 我们只需要加载这些文档的内容，不需要加载整个 200万 Corpus，省内存
    needed_doc_ids = set()
    for qid, docs in run_data.items():
        for doc_id in docs.keys():
            needed_doc_ids.add(doc_id)
            
    # 3. 加载 Qrels (为了标记 has_answer)
    # 🔧 修改：针对 MS MARCO 强制使用 dev，防止读到空的 test 文件
    if DATASET_NAME == "msmarco":
        qrels_path = os.path.join(BASE_PATH, 'qrels', 'dev.tsv')
        print("🔧 检测到 MS MARCO，强制加载 Qrels: dev.tsv")
    else:
        # 其他数据集优先找 test
        qrels_path = os.path.join(BASE_PATH, 'qrels', 'test.tsv')
        if not os.path.exists(qrels_path):
            qrels_path = os.path.join(BASE_PATH, 'qrels', 'dev.tsv')
            
    qrels = load_qrels(qrels_path)
    
    # 把标准答案的 Doc ID 也加进去，防止检索没召回导致报错
    for qid in target_qids:
        if qid in qrels:
            for gold_doc_id in qrels[qid]:
                needed_doc_ids.add(gold_doc_id)

    print(f"   - 需要提取的文档总数: {len(needed_doc_ids)}")

    # 4. 扫描 Corpus (提取内容)
    doc_lookup = {}
    corpus_path = os.path.join(BASE_PATH, 'corpus.jsonl')
    print(f"📂 扫描 Corpus: {corpus_path} (请稍候)...")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        # 使用 tqdm 显示进度，因为 MSMARCO 很大
        for line in tqdm(f, desc="Reading Corpus"):
            # 快速检查：如果这一行包含我们需要的ID，再解析 JSON (极大提升速度)
            # 这是一个简单的字符串匹配优化，防止 json.loads 每一行
            # 虽然有误判可能，但在 doc_id 较长时很有效。
            # 为稳妥起见，我们还是老老实实解析，但只存需要的
            item = json.loads(line)
            if item['_id'] in needed_doc_ids:
                doc_lookup[item['_id']] = {
                    "title": item.get("title", ""),
                    "text": item.get("text", "")
                }
    
    # 5. 加载 Queries (获取问题文本)
    print("📂 加载 Queries...")
    query_lookup = {}
    with open(os.path.join(BASE_PATH, 'queries.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item['_id'] in target_qids:
                query_lookup[item['_id']] = item['text']

    # 6. 组装最终数据
    print("🔨 正在组装最终 JSON...")
    final_data = []
    
    for qid in target_qids:
        if qid not in query_lookup:
            continue
            
        question_text = query_lookup[qid]
        gold_doc_ids = qrels.get(qid, [])
        
        # 构建 ctxs 列表
        ctxs = []
        top_docs = run_data[qid] # 这是一个 dict: {doc_id: score}
        
        # 按分数排序确保顺序正确
        sorted_docs = sorted(top_docs.items(), key=lambda x: x[1], reverse=True)
        
        for doc_id, score in sorted_docs:
            if doc_id in doc_lookup:
                doc_content = doc_lookup[doc_id]
                is_gold = doc_id in gold_doc_ids
                
                ctxs.append({
                    "id": doc_id,
                    "title": doc_content['title'],
                    "text": doc_content['text'],
                    "score": score,
                    "has_answer": is_gold # 这一项对 GARAG 很重要
                })
        
        # ⚠️ BEIR 数据集通常只有 Doc ID 作为答案，没有短语文本答案
        # GARAG 这里的 answers 字段如果不填可能会报错，或者评估为 0
        # 我们这里填入 "Unknown" 占位。
        # (GARAG 的攻击通常关注检索排序，只要 has_answer 标记对就行)
        final_data.append({
            "question": question_text,
            "answers": ["Unknown"], 
            "ctxs": ctxs
        })

    # 7. 保存
    print(f"💾 保存结果至: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"✅ 成功！已生成 {len(final_data)} 条完整数据。")
    print(f"➡️  下一步：在 eval.sh 中设置 --dataset={DATASET_NAME}_garag_ready")

if __name__ == "__main__":
    main()