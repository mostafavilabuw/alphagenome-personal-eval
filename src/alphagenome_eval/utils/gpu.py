"""
GPU utilities for AlphaGenome evaluation.

This module provides functions for automatic GPU selection and device management:
- get_best_gpu(): Returns the GPU index with most free memory
- select_device(): Returns optimal device string for PyTorch
- is_cuda_available(): Check CUDA availability

Example:
    >>> from alphagenome_eval.utils.gpu import select_device
    >>> device = select_device()  # Returns 'cuda:0', 'cuda:1', etc. or 'cpu'
    >>> model = model.to(device)
"""

import subprocess
from typing import Optional

import torch


def get_best_gpu() -> Optional[int]:
    """
    Get the GPU index with the most free memory.
    
    Uses nvidia-smi to query GPU memory and returns the index of the GPU
    with the most available memory. This is useful for multi-GPU systems
    where you want to automatically select the least loaded GPU.
    
    Returns:
        GPU index (0, 1, 2, ...) with most free memory, or None if:
        - nvidia-smi is not available
        - No GPUs are detected
        - An error occurs
        
    Example:
        >>> gpu_id = get_best_gpu()
        >>> if gpu_id is not None:
        ...     torch.cuda.set_device(gpu_id)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', 
             '--format=csv,noheader,nounits'],
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0]:
            return None
            
        gpus = []
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 2:
                gpu_idx = int(parts[0].strip())
                free_mem = int(parts[1].strip())
                gpus.append((gpu_idx, free_mem))
        
        if not gpus:
            return None
            
        # Return GPU with most free memory
        best_gpu = max(gpus, key=lambda x: x[1])
        return best_gpu[0]
        
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def is_cuda_available() -> bool:
    """
    Check if CUDA is available for PyTorch.
    
    Returns:
        True if CUDA is available and functional, False otherwise.
        
    Example:
        >>> if is_cuda_available():
        ...     print("GPU acceleration available")
    """
    return torch.cuda.is_available()


def select_device(prefer_cuda: bool = True) -> str:
    """
    Select the optimal device for PyTorch computations.
    
    Automatically selects the best available GPU (one with most free memory)
    or falls back to CPU if CUDA is not available.
    
    Args:
        prefer_cuda: If True, prefer GPU over CPU when available.
                     If False, always return 'cpu'.
                     
    Returns:
        Device string suitable for PyTorch:
        - 'cuda:N' where N is the best GPU index
        - 'cuda' if GPU is available but index detection fails
        - 'cpu' if CUDA is not available or prefer_cuda is False
        
    Example:
        >>> device = select_device()
        >>> print(f"Using device: {device}")
        Using device: cuda:0
        
        >>> model = model.to(device)
        >>> tensor = tensor.to(device)
    """
    if not prefer_cuda:
        return 'cpu'
        
    if not torch.cuda.is_available():
        return 'cpu'
    
    gpu_id = get_best_gpu()
    if gpu_id is not None:
        return f'cuda:{gpu_id}'
    
    # Fallback to default CUDA device if nvidia-smi fails
    return 'cuda'


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information:
        - cuda_available: Whether CUDA is available
        - gpu_count: Number of GPUs
        - best_gpu: Index of GPU with most free memory
        - selected_device: Recommended device string
        
    Example:
        >>> info = get_device_info()
        >>> print(f"GPUs: {info['gpu_count']}, Best: {info['best_gpu']}")
    """
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    best_gpu = get_best_gpu() if cuda_available else None
    selected = select_device()
    
    return {
        'cuda_available': cuda_available,
        'gpu_count': gpu_count,
        'best_gpu': best_gpu,
        'selected_device': selected
    }


if __name__ == '__main__':
    # CLI entry point for bash scripts
    info = get_device_info()
    
    if info['cuda_available'] and info['best_gpu'] is not None:
        print(info['best_gpu'])
    else:
        # Exit with error code if no GPU available
        import sys
        sys.exit(1)

