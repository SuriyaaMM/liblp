import torch

from typing import Tuple

from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import mxi8fp32
from liblp.lpbackend import lpbackend

@lpregistry.register(
    lpop.encode,
    mxi8fp32,
    lpbackend.torch,
)
def encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if __debug__:
        x_f_shape = x.flatten().shape
        if x_f_shape[0] % mxi8fp32.scale_block_size != 0:
            raise ValueError(f"[MXI8FP32 | Torch := x(shape={x_f_shape}[Flattened]) isn't divisible by scale_block_size({mxi8fp32.scale_block_size})")
    
    x_f = x.flatten()
    x_b = x_f.reshape(-1, mxi8fp32.scale_block_size)

    print(x_b.shape)

    x_bs = x_b.abs().amax(dim=1, keepdim=True) / mxi8fp32.int8_max

    x_bq = (x_b / x_bs).round().to(torch.int8)
    x_bq = torch.clamp(x_bq, mxi8fp32.int8_min, mxi8fp32.int8_max)

    return x_bq, x_bs

@lpregistry.register(
    lpop.decode,
    mxi8fp32,
    lpbackend.torch,
)
def decode(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    x_f = x.flatten()
    x_b = x_f.reshape(-1, mxi8fp32.scale_block_size)
    return x_b * scales
