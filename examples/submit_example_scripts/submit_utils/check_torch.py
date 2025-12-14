#!/usr/bin/env python
import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. No GPU names to display.")
