
try:
    import textattack
    print("textattack: installed")
except ImportError:
    print("textattack: missing")

try:
    import vllm
    print("vllm: installed")
except ImportError:
    print("vllm: missing")

try:
    import transformers
    print("transformers: installed")
except ImportError:
    print("transformers: missing")

try:
    import torch
    print("torch: installed")
except ImportError:
    print("torch: missing")
