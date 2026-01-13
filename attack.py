# import random
# import torch
# import numpy as np
# from src.option import Options
# from src.util import init_logger, timestr
# from src.task import ReaderDataset, evaluate
# from src.attacker import build_attack
# from textattack.augmentation import Augmenter
# from textattack.attack_args import AttackArgs

# import tqdm
# import os
# import json
# import logging

# from textattack.metrics.quality_metrics import Perplexity, USEMetric
# from textattack.shared import AttackedText, utils

# logger = logging.getLogger(__name__)

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # def main():
# #     option = Options("attack")
# #     opt, message = option.parse(timestr())
# #     logger = init_logger(opt)
# #     logger.info(message)
# #     logger.info("The name of experiment is {}".format(opt.name))
# #     logger.info("Attack type is {}".format(opt.method))

# #     dataset = ReaderDataset(opt)
# #     attack, dataset = build_attack(opt, dataset)

# #     if opt.is_black:
# #         result = attack.augment_dataset(dataset)

# #     elif opt.is_genetic:
# #         result = attack.attack_dataset(dataset)

# #     elif opt.is_hotflip:
# #         result = attack.attack_dataset(dataset)
# #     else:
# #         result = attack.attack_dataset(dataset)
# #     logger.info("Attack finished")
# #     evaluate(result)
# #     if opt.is_save:
# #         # data_dir = os.path.join(os.path.split(opt.data_dir)[0], "noise", "g_p_{}_seq_{}".format(opt.perturbation_level, opt.transformations_per_example))
# #         # os.makedirs(data_dir, exist_ok=True)
# #         with open(os.path.join(opt.output_dir, "{}.json".format(opt.method)), 'w') as f: json.dump(result,f)
    

# # if __name__=="__main__":
# #     main()

# # 辅助函数：把 Tensor 变成普通数字，防止 JSON 报错
# def make_serializable(obj):
#     if isinstance(obj, torch.Tensor):
#         return obj.item() if obj.numel() == 1 else obj.tolist()
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, (np.float32, np.float64)):
#         return float(obj)
#     elif isinstance(obj, (np.int32, np.int64)):
#         return int(obj)
#     elif isinstance(obj, list):
#         return [make_serializable(i) for i in obj]
#     elif isinstance(obj, dict):
#         return {k: make_serializable(v) for k, v in obj.items()}
#     return obj

# def main():
#     # 1. 初始化配置
#     t = timestr()
#     op = Options("attack")
#     opt, message = op.parse(t)
    
#     # 2. 初始化日志 (恢复原作者逻辑，这样日志会存到 output 文件夹)
#     global logger
#     logger = init_logger(opt)
#     logger.info(message)
#     logger.info(f"Experiment Name: {opt.name}")
#     logger.info(f"Attack Method: {opt.method}")

#     # 3. 设置随机种子 (保留你的修复)
#     if hasattr(opt, 'seed'):
#         seed = opt.seed
#     else:
#         seed = 42
    
#     logger.info(f"Setting Random Seed: {seed}")
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)

#     # 4. 加载数据 (保留你的修复，手动加载 JSON)
#     logger.info(f"📂 Loading dataset from: {opt.data_dir}")
#     with open(opt.data_dir, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)

#     # 👇 临时加上这一行，只跑前 5 个，用来调试
#     dataset = dataset[:5]
    
#     # ==========================================
#     # 🔧【关键修复】格式兼容性处理 (Hotfix)
#     # GARAG 代码只认 "context" 字段，但 BEIR 数据集通常叫 "text"
#     # 我们在这里遍历一遍，把 text 的内容复制给 context
#     # ==========================================
#     logger.info("🔧 Pre-processing data: Mapping 'text' to 'context'...")
#     fixed_count = 0
#     for item in dataset:
#         if 'ctxs' in item:
#             for ctx in item['ctxs']:
#                 # 如果有 text 但没 context，就补上 context
#                 if 'text' in ctx and 'context' not in ctx:
#                     ctx['context'] = ctx['text']
#                     fixed_count += 1
#     logger.info(f"✅ Data fixed! Updated {fixed_count} documents.")
#     # ==========================================

#     # 5. 构建攻击器
#     # 注意：build_attack 返回 (attacker, dataset)
#     attack, dataset = build_attack(opt, dataset)

#     # 6. 执行攻击 (恢复原作者的分支逻辑，更健壮)
#     logger.info("🚀 Starting Attack...")
#     if opt.is_black:
#         # 黑盒攻击通常调用 augment_dataset
#         result = attack.augment_dataset(dataset)
#     elif opt.is_genetic:
#         # 遗传算法调用 attack_dataset
#         result = attack.attack_dataset(dataset)
#     elif opt.is_hotflip:
#         # HotFlip 也调用 attack_dataset
#         result = attack.attack_dataset(dataset)
#     else:
#         # 默认情况
#         result = attack.attack_dataset(dataset)

#     logger.info(f"Attack finished. Generated {len(result)} samples.")

#     # 7. 评估结果
#     if len(result) > 0:
#         # 尝试捕获评估时的错误，防止最后一步崩了
#         try:
#             evaluate(result)
#         except Exception as e:
#             logger.error(f"Evaluation failed: {e}")

#     # 8. 保存结果 (保留你的序列化修复)
#     if opt.is_save:
#         logger.info("Processing data for saving...")
#         clean_result = make_serializable(result)
        
#         # 确保输出目录存在
#         os.makedirs(opt.output_dir, exist_ok=True)
        
#         output_file = os.path.join(opt.output_dir, "{}.json".format(opt.method))
#         logger.info(f"💾 Saving results to: {output_file}")
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(clean_result, f, indent=4)

# if __name__ == "__main__":
#     main()


import random
import torch
import numpy as np
from src.option import Options
from src.util import init_logger, timestr
from src.attacker import build_attack
from src.task import evaluate
import logging
import os
import json

# 设置 logger
logger = logging.getLogger(__name__)

# 禁用 Tokenizers 并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 🔧【核武器级修复】自定义 JSON 编码器
# 专门解决 Tensor/Numpy 嵌套过深无法保存的问题
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)
# ==========================================

def main():
    # 1. 解析参数
    t = timestr()
    op = Options("attack")
    opt, message = op.parse(t)

    # 2. 初始化日志
    global logger
    logger = init_logger(opt)
    logger.info(message)
    logger.info(f"Experiment Name: {opt.name}")
    logger.info(f"Attack Method: {opt.method}")

    # 3. 设置随机种子
    seed = getattr(opt, 'seed', 42)
    logger.info(f"Setting Random Seed: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 4. 加载数据
    logger.info(f"📂 Loading dataset from: {opt.data_dir}")
    if not os.path.exists(opt.data_dir):
        logger.error(f"❌ Error: Dataset file not found at {opt.data_dir}")
        return

    with open(opt.data_dir, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    # 🔧 调试模式：如果设置了环境变量 GARAG_DEBUG_LIMIT，则截取部分数据
    debug_limit = os.environ.get("GARAG_DEBUG_LIMIT")
    if debug_limit:
        try:
            limit = int(debug_limit)
            logger.info(f"🐛 Debug mode active: Limiting dataset to first {limit} examples.")
            dataset = dataset[:limit]
        except ValueError:
            logger.warning(f"⚠️ Invalid GARAG_DEBUG_LIMIT value: {debug_limit}. Ignoring.")

    # 5. 数据格式热修复 (text -> context)
    logger.info("🔧 Pre-processing data: Mapping 'text' to 'context'...")
    for item in dataset:
        if 'ctxs' in item:
            for ctx in item['ctxs']:
                if 'text' in ctx and 'context' not in ctx:
                    ctx['context'] = ctx['text']

    # 6. 构建攻击器
    attack, dataset = build_attack(opt, dataset)

    # 7. 执行攻击
    logger.info("🚀 Starting Attack...")
    if opt.is_black:
        result = attack.augment_dataset(dataset)
    elif opt.is_genetic:
        result = attack.attack_dataset(dataset)
    elif opt.is_hotflip:
        result = attack.attack_dataset(dataset)
    else:
        result = attack.attack_dataset(dataset)

    logger.info(f"Attack finished. Generated {len(result)} samples.")

    # 8. 评估结果
    if len(result) > 0:
        try:
            evaluate(result)
        except Exception as e:
            logger.error(f"Evaluation warning: {e}")

    # 9. 保存结果 (使用 NumpyEncoder)
    if opt.is_save:
        logger.info("Processing data for saving...")
        
        os.makedirs(opt.output_dir, exist_ok=True)
        output_file = os.path.join(opt.output_dir, "{}.json".format(opt.method))
        logger.info(f"💾 Saving results to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # cls=NumpyEncoder 是关键！
            json.dump(result, f, indent=4, cls=NumpyEncoder) 
            
    logger.info("✅ All Done!")

if __name__ == "__main__":
    main()