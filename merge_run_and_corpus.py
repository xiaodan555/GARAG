import json
import os
import csv
from tqdm import tqdm

# ================= 配置区域 =================
# BEIR 根目录
BEIR_ROOT = "data/beir"
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

def process_dataset(dataset_name):
    # 路径配置
    base_path = os.path.join(BEIR_ROOT, dataset_name)
    run_file = os.path.join(base_path, f"run_contriever_{dataset_name}_top100.json")
    output_file = os.path.join(base_path, f"{dataset_name}_garag_ready.json")

    print(f"\n🚀 开始拼接数据集: {dataset_name}")
    print(f"📂 读取 Run File: {run_file}")
    
    if not os.path.exists(run_file):
        print(f"❌ 错误：找不到 Run File！请确认你是否运行了检索脚本 ({run_file})。")
        return

    # 1. 加载 Run File (检索结果)
    with open(run_file, 'r') as f:
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
    if dataset_name == "msmarco":
        qrels_path = os.path.join(base_path, 'qrels', 'dev.tsv')
        print("🔧 检测到 MS MARCO，强制加载 Qrels: dev.tsv")
    else:
        # 其他数据集优先找 test
        qrels_path = os.path.join(base_path, 'qrels', 'test.tsv')
        if not os.path.exists(qrels_path):
            qrels_path = os.path.join(base_path, 'qrels', 'dev.tsv')
            
    qrels = load_qrels(qrels_path)
    
    # 把标准答案的 Doc ID 也加进去，防止检索没召回导致报错
    for qid in target_qids:
        if qid in qrels:
            for gold_doc_id in qrels[qid]:
                needed_doc_ids.add(gold_doc_id)

    print(f"   - 需要提取的文档总数: {len(needed_doc_ids)}")

    # 4. 扫描 Corpus (提取内容)
    doc_lookup = {}
    corpus_path = os.path.join(base_path, 'corpus.jsonl')
    print(f"📂 扫描 Corpus: {corpus_path} (请稍候)...")
    
    if not os.path.exists(corpus_path):
         print(f"❌ 错误：找不到 Corpus File ({corpus_path})")
         return

    with open(corpus_path, 'r', encoding='utf-8') as f:
        # 使用 tqdm 显示进度，因为 MSMARCO 很大
        for line in tqdm(f, desc=f"Reading Corpus ({dataset_name})"):
            # 快速检查：如果这一行包含我们需要的ID，再解析 JSON (极大提升速度)
            # 这是一个简单的字符串匹配优化，防止 json.loads 每一行
            # 虽然有误判可能，但在 doc_id 较长时很有效。
            # 为稳妥起见，我们还是老老实实解析，但只存需要的
            # 为了性能，可以尝试简单字符串 check，但这里为了保险直接 json.loads
            # 如果觉得慢，可以先 check string in line
            item = json.loads(line)
            if item['_id'] in needed_doc_ids:
                doc_lookup[item['_id']] = {
                    "title": item.get("title", ""),
                    "text": item.get("text", "")
                }
    
    # 5. 加载 Queries (获取问题文本)
    print("📂 加载 Queries...")
    query_lookup = {}
    queries_path = os.path.join(base_path, 'queries.jsonl')
    if not os.path.exists(queries_path):
         print(f"❌ 错误：找不到 Queries File ({queries_path})")
         return

    with open(queries_path, 'r', encoding='utf-8') as f:
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
    print(f"💾 保存结果至: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
        
    print(f"✅ [{dataset_name}] 处理完成！已生成 {len(final_data)} 条完整数据。")


def main():
    datasets = ["nq", "hotpotqa", "msmarco"]
    for ds in datasets:
        process_dataset(ds)

if __name__ == "__main__":
    main()
