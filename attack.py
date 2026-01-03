import random
from src.option import Options
from src.util import init_logger, timestr
from src.task import ReaderDataset, evaluate
from src.attacker import build_attack
from textattack.augmentation import Augmenter
from textattack.attack_args import AttackArgs

import tqdm
import os
import json
import logging

from textattack.metrics.quality_metrics import Perplexity, USEMetric
from textattack.shared import AttackedText, utils

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def main():
#     option = Options("attack")
#     opt, message = option.parse(timestr())
#     logger = init_logger(opt)
#     logger.info(message)
#     logger.info("The name of experiment is {}".format(opt.name))
#     logger.info("Attack type is {}".format(opt.method))

#     dataset = ReaderDataset(opt)
#     attack, dataset = build_attack(opt, dataset)

#     if opt.is_black:
#         result = attack.augment_dataset(dataset)

#     elif opt.is_genetic:
#         result = attack.attack_dataset(dataset)

#     elif opt.is_hotflip:
#         result = attack.attack_dataset(dataset)
#     else:
#         result = attack.attack_dataset(dataset)
#     logger.info("Attack finished")
#     evaluate(result)
#     if opt.is_save:
#         # data_dir = os.path.join(os.path.split(opt.data_dir)[0], "noise", "g_p_{}_seq_{}".format(opt.perturbation_level, opt.transformations_per_example))
#         # os.makedirs(data_dir, exist_ok=True)
#         with open(os.path.join(opt.output_dir, "{}.json".format(opt.method)), 'w') as f: json.dump(result,f)
    

# if __name__=="__main__":
#     main()

# 辅助函数：把 Tensor 变成普通数字，防止 JSON 报错
def make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    return obj

def main():
    opt = parse_option()
    
    # 设置随机种子
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    
    attack, dataset = build_attack(opt, dataset_name=opt.dataset)
    
    # 开始攻击
    result = attack.attack_dataset(dataset)
    
    # 打印评估结果
    if len(result) > 0:
        evaluate(result)

    # === 关键修复：保存前先清洗数据 ===
    print("正在处理数据格式...")
    clean_result = make_serializable(result)

    # 保存文件
    output_file = os.path.join(opt.output_dir, "{}.json".format(opt.method))
    print(f"正在保存结果到: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(clean_result, f, indent=4) # 加个缩进，方便你看文件内容

if __name__ == "__main__":
    main()