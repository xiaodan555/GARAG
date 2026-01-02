from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM
# from vllm import LLM, SamplingParams
# === 修改开始 ===
try:
    from vllm import LLM, SamplingParams
except ImportError:
    # 如果没安装 vllm，就设为 None，防止报错
    LLM = None
    SamplingParams = None
# === 修改结束 ===
from openai import OpenAI
import requests
from .util import f1

import lightning.pytorch as pl

import os
import math
import torch
import logging

cls_mapping = {
    "Llama-7b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-7b-chat-hf"),
    "Llama-13b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-13b-chat-hf"),
    "Mistral-7b": (MistralForCausalLM, AutoTokenizer, True, "Mistral-7B-Instruct-v0.2"),
    "vicuna-7b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-7b-v1.5"),
    "vicuna-13b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-13b-v1.5"),
    "gemma-7b": (AutoModelForCausalLM, AutoTokenizer, True, "gemma-7b-it")
}

logger = logging.getLogger(__name__)

save_keys = [
    "question", "doc_id", "question", "answers"
]

def load_reader(opt):
    # 只要参数里包含 "ollama"，就启用我们的 Ollama Reader
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
    
    # def get_loss(self, input_ids, attention_mask):


    def _cal_label_prob(self, probs, labels):
        result = []
        for prob, label in zip(probs, labels):
            mask = label > 0
            prob, label = prob[mask], label[mask]
            log_softmax = torch.nn.functional.log_softmax(prob, dim=-1)
            # from IPython import embed; embed(); exit(0)
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
        _, tokenizer_cls, _, hf_name = _load_model(opt)
        self.model = LLM(model=os.path.join(opt.model_dir, hf_name), gpu_memory_utilization=0.70, kv_cache_dtype="fp8_e5m2")
        self.tokenizer = tokenizer_cls.from_pretrained(os.path.join(opt.model_dir, hf_name))
        self.gen_sampling = SamplingParams(temperature=1, top_p=1, max_tokens=30)
        self.score_sampling = SamplingParams(temperature=1, top_p=1, prompt_logprobs=0, max_tokens=1)

    def _cal_label_prob(self, outputs, labels):
        labels = [input_id[1:] for input_id in self.tokenizer(labels).input_ids]
        probs = [output.prompt_logprobs for output in outputs]
        result = []
        for prob, label in zip(probs, labels):
            prs = []
            for pr, l in zip(prob[-1 * len(label):], label):
                k,v = list(pr.items())[0]
                assert k == l
                prs.append(v)
            avg_nll = sum(prs)
            result.append(math.exp(avg_nll)/len(label))
        return result

    def forward(self, inputs):
        preds= [output.outputs[0].text.strip() for output in self.model.generate(inputs, use_tqdm=False, sampling_params=self.gen_sampling)]
        return preds
    
    def get_scores(self, inputs, labels):
        outputs = self.model.generate(inputs, use_tqdm=False, sampling_params=self.score_sampling)
        scores = self._cal_label_prob(outputs, labels)
        return scores
    
    def get_tokenizer(self):
        return self.tokenizer
    
# class Reader_Ollama(torch.nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         # Extract model name from reader string, e.g., "ollama-vicuna" -> "vicuna:7b"
#         # We need a mapping or assume user passes "ollama-vicuna:7b"
#         # For simplicity, let's map known ones or use what comes after "ollama-"
        
#         self.model_name = opt.reader.replace("ollama-", "")
#         if self.model_name == "vicuna": self.model_name = "vicuna:7b"
#         if self.model_name == "llama3": self.model_name = "llama3.1:latest"
        
#         self.api_url = "http://localhost:11434/api/generate"
#         self.system_prompt = "You are a QA assistant. Read the document and answer the question. Your answer should be concise and short phrase, not sentence."

#     def _call_ollama(self, prompt):
#         data = {
#             "model": self.model_name,
#             "prompt": prompt,
#             "stream": False,
#             "system": self.system_prompt,
#             "options": {
#                 "temperature": 0.0, # Deterministic
#                 "num_predict": 30
#             }
#         }
#         try:
#             response = requests.post(self.api_url, json=data)
#             response.raise_for_status()
#             return response.json()['response'].strip()
#         except Exception as e:
#             logger.error(f"Ollama API call failed: {e}")
#             return ""

#     def forward(self, contexts, question=None):
#         # GARAG passes contexts (list of strings) and question (string)
#         # But sometimes it passes just inputs if it's unified.
#         # Check Reader_GPT usage: forward(contexts, question)
        
#         # If input is already formatted (Reader_vLLM style), handle that
#         # But attack.py calls reader.generate(contexts, question) -> wrapper -> reader.forward
        
#         # Let's align with Reader_GPT signature
#         preds = []
#         if question is None:
#              # Assume contexts are list of full prompts? 
#              # No, looking at wrapper, inputs are formatted.
#              # But Reader_GPT signature is specific.
#              pass

#         # If called from generate(contexts, question) in Reader_Wrapper
#         if isinstance(contexts, list) and question is not None:
#              for context in contexts:
#                 prompt = f"Document: {context}\nQuestion: {question}"
#                 preds.append(self._call_ollama(prompt))
#         # If called differently (e.g. vLLM style), adapt
#         elif isinstance(contexts, list) and question is None:
#              # Assume contexts are prompts
#              for prompt in contexts:
#                   preds.append(self._call_ollama(prompt))
                  
#         return preds

#     def get_scores(self, contexts, question, answers):
#         # We simulate probability score using F1 overlap
#         # Higher F1 -> Higher probability/score
#         scores = []
#         for context in contexts:
#             prompt = f"Document: {context}\nQuestion: {question}"
#             pred = self._call_ollama(prompt)
            
#             # Calculate max F1 against any of the answers
#             max_f1 = 0
#             # answers is usually a list of valid answers [ans1, ans2]
#             # But sometimes it might be passed differently. 
#             # In Reader_GPT it receives (contexts, question, answers)
            
#             # If answers is a single string, wrap it
#             current_answers = answers if isinstance(answers, list) else [answers]
            
#             for ans in current_answers:
#                 score = f1(ans, pred)
#                 if score > max_f1:
#                     max_f1 = score
            
#             # GARAG expects a probability-like score (0-1, or logprob)
#             # If we return F1 directly (0-1), it might work.
#             # Reader_GPT returns exp(logprob), which is prob [0, 1].
#             # So F1 [0, 1] is a reasonable proxy.
#             scores.append(max_f1)
            
#         return scores
    
#     def get_tokenizer(self):
#         # Ollama manages tokenization internally
#         return None
class Reader_Ollama(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # 解析模型名称：比如 "ollama-vicuna" -> "vicuna:7b"
        # 这里的逻辑是：去掉 "ollama-" 前缀，然后根据关键词匹配具体的 Ollama 模型 tag
        raw_name = opt.reader.replace("ollama-", "")
        
        if "vicuna" in raw_name:
            self.model_name = "vicuna:7b"
        elif "llama3" in raw_name:
            self.model_name = "llama3.1:latest" # 请确保这和你 ollama list 里的名字一致
        elif "qwen" in raw_name:
            self.model_name = "qwen2.5:7b"
        else:
            self.model_name = raw_name # 如果没匹配到，就直接用传进来的名字
            
        self.api_url = "http://localhost:11434/api/generate"
        self.system_prompt = "You are a QA assistant. Read the document and answer the question. Your answer should be concise and short phrase, not sentence."
        
        print(f"[Ollama] Initialized with model: {self.model_name}")

    def _call_ollama(self, prompt):
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "system": self.system_prompt,
            "options": {
                "temperature": 0.0, # 设置为0，保证实验可复现
                "num_predict": 50   # 限制回答长度
            }
        }
        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return ""

    def forward(self, contexts, question=None):
        # 模仿 Reader_GPT 的接口
        # contexts 是检索到的文档列表，question 是问题
        preds = []
        
        # 兼容性处理：如果只传了 contexts（某些攻击步骤可能只传 input list）
        if question is None:
             # 假设 contexts 已经是拼接好的 Prompt
             for prompt in contexts:
                  preds.append(self._call_ollama(prompt))
             return preds

        # 正常 RAG 流程
        for context in contexts:
            prompt = f"Document: {context}\nQuestion: {question}"
            preds.append(self._call_ollama(prompt))
        return preds

    def get_scores(self, contexts, question, answers):
        # GARAG 需要一个 "分数" 来评估攻击效果（分数越高代表回答越准，攻击者想让这个分数变低）
        # 由于 Ollama 很难方便地取到 token 级的 logprobs，我们用 F1 分数作为近似概率
        # F1 = 1.0 (完全正确) -> 类似概率 1.0
        # F1 = 0.0 (完全错误) -> 类似概率 0.0
        
        scores = []
        
        # 确保 answers 是列表
        current_answers = answers if isinstance(answers, list) else [answers]
            
        for context in contexts:
            prompt = f"Document: {context}\nQuestion: {question}"
            pred = self._call_ollama(prompt)
            
            # 计算预测结果和标准答案的 F1 相似度
            max_f1 = 0
            for ans in current_answers:
                score = f1(ans, pred) # 调用 util 里的 f1 函数
                if score > max_f1:
                    max_f1 = score
            
            # 攻击算法希望这个 score 越小越好
            scores.append(max_f1)
            
        return scores
    
    def get_tokenizer(self):
        # 返回 None，TextAttack 会使用默认的空格分词，这通常足够了
        return None

class Reader_GPT(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        OPENAI_API_KEY = opt.openai_key
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.system_prompt = "You are a QA assistant. Read the document and answer the question. Your answer should be concise and short phrase, not sentence."
    

    def _cal_label_prob(self, outputs, labels):
        raise NotImplementedError
    
    def forward(self, contexts, question):
        preds = []
        for context in contexts:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Document: {}\nQuestion: {}".format(context, question)}
                ],
                logprobs=True
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
    
    def get_tokenizer(self):
        raise NotImplementedError

class Read_Module(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        if opt.is_vllm:
            self.model = Reader(opt)
        else:
            self.model = Reader_vLLM(opt)
        self.is_vllm = opt.is_vllm
        logger.info("Model Load Done")

    # def forward(self, input_ids, attention_mask):
    #     preds = self.model(input_ids, attention_mask)
    #     return preds

    def predict_step(self, batch, batch_idx):
        if self.is_vllm:
            preds = self.model(batch['inputs'])
        else:
            preds = self.model(batch['input_ids'], batch['attention_mask'])
        result = self._process_output(preds, batch)
        return result
    
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
        # result = [{
        #     "question": batch["question"][i],
        #     "context": batch["context"][i],
        #     "answers": batch["answers"][i],
        #     "pred": preds[i],
        # }  for i in range(len(preds))]
        return result
    