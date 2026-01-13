import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

try:
    from vllm import LLM
    print("vLLM imported successfully")
except ImportError:
    print("vLLM import failed")
except Exception as e:
    print(f"vLLM error: {e}")
