"""
CUDA optimization utilities.

This module provides shared CUDA configuration for training and inference.
"""
import warnings

import torch
import torch._dynamo.config


def setup_cuda_optimizations() -> None:
    """Configure CUDA/PyTorch optimizations for training and inference.

    Enables:
    - TF32 for faster matrix operations on Ampere+ GPUs
    - cuDNN autotuning for optimal convolution algorithms
    - Flash/memory-efficient scaled dot-product attention
    - Increased dynamo cache size for torch.compile

    Also suppresses noisy library warnings (MONAI cache_dir, TorchVision deprecation).
    """
    # Suppress noisy library warnings
    warnings.filterwarnings("ignore", message="Setting cache_dir to")  # MONAI
    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")  # TorchVision
    warnings.filterwarnings("ignore", message="Arguments other than a weight enum")  # TorchVision
    # cuDNN settings
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # Scaled dot-product attention backends
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # TF32 for matmul
    torch.backends.cuda.matmul.allow_tf32 = True

    # torch.compile cache
    torch._dynamo.config.cache_size_limit = 32
