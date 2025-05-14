
import gc
import torch

def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

