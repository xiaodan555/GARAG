from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from openai import OpenAI
import requests
from .util import f1

import lightning.pytorch as pl

import os
import math
import torch
import logging
from src.util import f1

cls_mapping = {
    "Llama-7b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-7b-chat-hf"),
    "Llama-13b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-13b-chat-hf"),
    "Mistral-7b": (MistralForCausalLM, AutoTokenizer, True, "Mistral-7B-Instruct-v0.2"),
    "vicuna-7b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-7b-v1.5"),
    "vicuna-13b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-13b-v1.5"),
    "gemma-7b": (AutoModelForCausalLM, AutoTokenizer, True, "gemma-7b-it"),
    "qwen-8b": (AutoModelForCausalLM, AutoTokenizer, True, "Qwen3-8B"),
    "deepseek": (AutoModelForCausalLM, AutoTokenizer, True, "DeepSeek-R1-Distill-Llama-8B"),
    "llama3.2-1b": (AutoModelForCausalLM, AutoTokenizer, True, "Llama-3.2-1B-Instruct"),
    "Llama-3.2-1B-Instruct": (AutoModelForCausalLM, AutoTokenizer, True, "Llama-3.2-1B-Instruct")
}

logger = logging.getLogger(__name__)

save_keys = [
    "question", "doc_id", "question", "answers"
]

def load_reader(opt):
    if "ollama" in opt.reader:
        return Reader_Ollama(opt)
    elif opt.reader == "chatgpt":
        return Reader_GPT(opt)
    elif opt.is_vllm:
        return Reader_vLLM(opt)
    else:
        return Reader(opt)

def _load_model(opt):
    reader_name = opt.reader
    if reader_name in cls_mapping:
        return cls_mapping[reader_name]
    else:
        NotImplementedError()

class Reader(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        model_cls, tokenizer_cls, self.is_decoder, hf_name = _load_model(opt)
        self.model = model_cls.from_pretrained(os.path.join(opt.model_dir, hf_name)).to("cuda:0")
        self.tokenizer = tokenizer_cls.from_pretrained(os.path.join(opt.model_dir, hf_name))
        self.generate_kwargs = dict(
            max_new_tokens=opt.max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True
        )
        if self.is_decoder:
            self.tokenizer.padding_side = "left"
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model.generate(input_ids=input_ids.to(self.model.device), attention_mask=attention_mask.to(self.model.device), **self.generate_kwargs)
        preds = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return preds

    def _cal_label_prob(self, probs, labels):
        result = []
        for prob, label in zip(probs, labels):
            mask = label > 0
            prob, label = prob[mask], label[mask]
            log_softmax = torch.nn.functional.log_softmax(prob, dim=-1)
            nll = -log_softmax.gather(1, label.unsqueeze(0).transpose(0, 1))
            avg_nll = torch.sum(nll, dim=0) * -1
            result.append(float(torch.exp(avg_nll / float(label.shape[0]))))
        return result
    
    def get_scores(self, input_ids, label_ids):
        outputs = self.model(input_ids=input_ids.to(self.model.device), labels=label_ids.to(self.model.device))
        scores = self._cal_label_prob(outputs.logits, label_ids.to(self.model.device))
        return scores
    
    def get_tokenizer(self):
        return self.tokenizer

class Reader_vLLM(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # 获取模型路径
        _, _, _, hf_name = _load_model(opt)
        model_path = os.path.join(opt.model_dir, hf_name)
        
        logger.info(f"🚀 Initializing vLLM with model: {model_path}")
        
        # 显存控制：给 Ollama 留出空间 (4090 24GB: 0.65 * 24 ≈ 15.6GB 给 vLLM)
        # 剩下 ~8GB 给 Ollama (5-6GB) 和系统
        self.llm = LLM(
            model=model_path, 
            trust_remote_code=True,
            gpu_memory_utilization=0.65, 
            max_model_len=2048
        )
        
        self.sampling_params = SamplingParams(
            temperature=0, 
            max_tokens=opt.max_new_tokens
        )
        
        # vLLM 不需要显式 tokenizer，但为了兼容性，可以加载一个
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(self, inputs):
        # inputs 必须是 list of strings
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str):
            outputs = self.llm.generate(inputs, self.sampling_params)
            preds = [output.outputs[0].text.strip() for output in outputs]
            return preds
        else:
            logger.error("vLLM forward received invalid inputs (expected list of strings)")
            return []

    def get_scores(self, input_ids, label_ids):
        # vLLM 主要用于生成，获取 logprobs 计算 score 比较麻烦
        # 这里暂时抛出异常或返回伪造数据，因为 GARAG 攻击阶段主要依赖 generate
        # 如果需要 white-box gradient attack，vLLM 是不支持的（vLLM 不支持反向传播）
        # 但既然你原本用 Ollama，说明你跑的是 Black-box 攻击，所以这里只需 generate
        logger.warning("vLLM does not support gradient-based score calculation (get_scores). Returning dummy scores.")
        return [0.0] * len(input_ids)

    def get_tokenizer(self):
        return self.tokenizer

class Reader_Ollama(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        # === 恢复 Ollama 初始化逻辑 ===
        raw_name = opt.reader.replace("ollama-", "")
        
        # 1. 优先使用用户明确指定的 tag (e.g. "qwen3:8b")
        if ":" in raw_name:
            self.model_name = raw_name
            
        # 2. Qwen 系列 (Qwen3, Qwen2.5)
        elif "qwen" in raw_name:
            if "2.5" in raw_name:
                if "14b" in raw_name:
                    self.model_name = "qwen2.5:14b"
                elif "32b" in raw_name:
                    self.model_name = "qwen2.5:32b"
                else:
                    self.model_name = "qwen2.5:7b"
            else:
                # 默认视为 Qwen3
                self.model_name = "qwen3:8b"

        # 3. Vicuna 系列
        elif "vicuna" in raw_name:
            if "13b" in raw_name:
                self.model_name = "vicuna:13b"
            else:
                self.model_name = "vicuna:7b"
        
        # 4. Llama 系列 (Llama 3.2, Llama 2, Llama 3.1)
        elif "llama" in raw_name:
            if "3.2" in raw_name:
                if "3b" in raw_name:
                    self.model_name = "llama3.2:3b"
                else:
                    self.model_name = "llama3.2:1b"
            elif "2" in raw_name:
                 if "13b" in raw_name:
                     self.model_name = "llama2:13b"
                 else:
                     self.model_name = "llama2:7b"
            else:
                # 默认 Llama 3.1
                self.model_name = "llama3.1:latest"

        # 5. DeepSeek
        elif "deepseek" in raw_name:
            self.model_name = "deepseek-r1:8b"
            
        else:
            self.model_name = raw_name
            
        env_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
        base_url = f"http://{env_host}" if "http" not in env_host else env_host
        self.api_url = f"{base_url}/api/generate"
        
        self.system_prompt = "You are a QA assistant. Read the document and answer the question. Your answer should be concise and short phrase, not sentence."
        
        print(f"[Ollama] 使用模型: {self.model_name} 地址: {self.api_url}")

        # 用于解码 input_ids 的 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except:
            self.tokenizer = None
        
        # === vLLM 逻辑已注释 ===
        # self.model = LLM(model=model_path, ...)

    def _call_ollama(self, prompt):
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "system": self.system_prompt,
            "options": {"temperature": 0.0, "num_predict": 50}
        }
        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except Exception as e:
            logger.error(f"Ollama API 掉用失败: {e}")
            return ""

    def forward(self, contexts=None, question=None, input_ids=None, **kwargs):
        preds = []
        # 情况 A: 传入 input_ids (Tensor)
        if input_ids is not None:
            decoded_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            for text in decoded_texts:
                preds.append(self._call_ollama(text))
            return preds

        # 情况 B: 传入 contexts (List[str])
        if question is None:
             for prompt in contexts:
                  preds.append(self._call_ollama(prompt))
             return preds

        for context in contexts:
            prompt = f"Document: {context}\nQuestion: {question}"
            preds.append(self._call_ollama(prompt))
        return preds

    def get_scores(self, input_ids, label_ids, **kwargs):
        """
        使用 F1 分数模拟概率分数
        """
        scores = []
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        gold_answers = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        for prompt, gold in zip(prompts, gold_answers):
            pred = self._call_ollama(prompt)
            score = f1(pred, gold)
            scores.append(score)
            
        return torch.tensor(scores, device=input_ids.device if isinstance(input_ids, torch.Tensor) else 'cpu')
    
    def get_tokenizer(self):
        return self.tokenizer

class Reader_GPT(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        OPENAI_API_KEY = opt.openai_key
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.system_prompt = "You are a QA assistant. Read the document and answer the question. Your answer should be concise and short phrase, not sentence."
    
    def forward(self, contexts, question):
        preds = []
        for context in contexts:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Document: {}\nQuestion: {}".format(context, question)}
                ]
            )
            preds.append(completion.choices[0].message.content)
        return preds
    
    def get_scores(self, contexts, question, answers):
        from math import exp
        scores = []
        for context in contexts:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                n=10,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Document: {}\nQuestion: {}".format(context, question)}
                ],
                logprobs=True
            )
            score = 0
            for choice in completion.choices:
                pred = choice.message.content
                if f1(answers, pred) > 0.5:
                    for token in choice.logprobs.content:
                        score += token.logprob
                    score = exp(score)
                    break
            scores.append(score)
        return scores

class Read_Module(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = load_reader(opt)
        self.is_vllm = opt.is_vllm
        logger.info("Model Load Done")

    def predict_step(self, batch, batch_idx):
        # 统一使用 model 的 forward
        if self.is_vllm:
            preds = self.model(batch['inputs'])
        else:
            # 兼容 Reader 类
            if hasattr(self.model, 'forward'):
                 if 'input_ids' in batch:
                     preds = self.model(batch['input_ids'], batch['attention_mask'])
                 else:
                     preds = self.model(batch['inputs'])
            else:
                 preds = []
        return self._process_output(preds, batch)
    
    def _process_output(self, preds, batch):
        keys = list(batch.keys())
        result = []
        for i in range(len(preds)):
            instance = {}
            for key in keys:
                if not isinstance(batch[key][i],torch.Tensor) and key in save_keys:
                    instance[key] = batch[key][i]
            instance["pred"] = preds[i]
            result.append(instance)
        return result
