# GARAG 项目复现指南

本指南详细说明了如何配置环境、准备数据以及运行 GARAG 项目的实验。

## 1. 环境配置

首先，请确保您的系统中已安装 Python（推荐 3.10+）。然后，按照以下步骤安装依赖：

```bash
# 创建虚拟环境（可选但推荐）
conda create -n garag python=3.10
conda activate garag

# 安装项目依赖
pip install -r requirements_merged.txt
```

> **注意：** 如果安装 `vllm` 或 `textattack` 遇到问题，请确保其版本与 `requirements_merged.txt` 中指定的版本一致。

## 2. 数据准备

### 步骤 2.1: 上传并解压数据集
您需要提供原始的 BEIR 数据集。请将以下压缩包上传到项目根目录的 `data/beir/` 目录下：

*   `nq.zip`
*   `hotpotqa.zip`
*   `msmarco.zip`

上传完成后，请**解压**这些文件，确保目录结构如下：

```text
data/
└── beir/
    ├── nq/
    │   ├── corpus.jsonl
    │   ├── queries.jsonl
    │   └── qrels/
    ├── hotpotqa/
    │   └── ...
    └── msmarco/
        └── ...
```

### 步骤 2.2: 数据预处理
我们提供了两个脚本用于将原始 BEIR 数据转换成 GARAG 模型可读取的格式：`run_retrieval_all.py` 和 `merge_run_and_corpus.py`。

您需要为您想测试的**每个数据集**分别运行这两个脚本。

**配置说明：**
在运行脚本之前，请打开脚本文件，修改文件顶部的 `DATASET_NAME` 变量，使其与您要处理的数据集名称匹配（例如 `"nq"`、`"hotpotqa"` 或 `"msmarco"`）。

**以处理 'nq' 数据集为例：**

1.  **运行检索与采样：**
    打开 `run_retrieval_all.py`，设置 `DATASET_NAME = "nq"`，然后运行：
    ```bash
    python run_retrieval_all.py
    ```
    *该脚本会在数据集目录下生成 `run_contriever_nq_top100.json`。*

2.  **生成 GARAG 运行数据：**
    打开 `merge_run_and_corpus.py`，设置 `DATASET_NAME = "nq"`，然后运行：
    ```bash
    python merge_run_and_corpus.py
    ```
    *该脚本会生成最终的实验输入文件 `nq_garag_ready.json`。*

> **如需处理其他数据集，请重复上述步骤并修改对应的 `DATASET_NAME`。**

## 3. 运行实验

当数据准备就绪（例如 `nq_garag_ready.json` 已生成）后，即可启动攻击/评估实验。

使用提供的 Shell 脚本启动实验：

```bash
bash run_nq_vllm.sh
```

**注意事项：**
*   请检查 `run_nq_vllm.sh` 中的 `--dataset` 参数，确保它指向了您生成的 JSON 文件。
*   实验日志将自动保存到 `output/` 目录下。