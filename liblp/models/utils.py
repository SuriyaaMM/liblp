import torch

from typing import Callable, Tuple

def encode_weight(
    w: torch.Tensor, 
    block_size: int,
    encoder_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
):
    n, k = w.shape
    assert k % block_size == 0, f"K={k} must be divisible by {block_size}"
    u8, scales = encoder_fn(w.to(torch.float32).contiguous())
    return (
        u8.reshape(n, k // 2).contiguous(),
        scales.reshape(n, k // block_size).contiguous(),
    )