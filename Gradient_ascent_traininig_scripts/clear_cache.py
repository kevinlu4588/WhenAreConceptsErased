import torch

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    else:
        print("CUDA is not available on this device.")

clear_cuda_cache()
