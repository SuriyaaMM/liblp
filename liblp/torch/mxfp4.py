import torch
from typing import Tuple

from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import mxfp4
from liblp.lpbackend import lpbackend


def _scales(blocks: torch.Tensor) -> torch.Tensor:
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    tiny = torch.finfo(torch.float32).tiny
    # if SCALE_MODE == "e8m0":
    #     e = torch.floor(torch.log2(amax.clamp_min(tiny))) - 2
    #     s = torch.exp2(e.clamp(-127, 127))
    # elif SCALE_MODE == "e8m0_ceil":
    #     e = torch.ceil(torch.log2((amax / 6.0).clamp_min(tiny)))
    #     s = torch.exp2(e.clamp(-127, 127))
    # elif SCALE_MODE == "fp32":
    s = amax / 6.0
    # else:
    #     raise ValueError(SCALE_MODE)
    return torch.where(amax == 0, torch.ones_like(s), s)


def _to_code(a: torch.Tensor) -> torch.Tensor:
    i = lambda t: t.to(torch.int32)
    code = (
        i(a > 0.25) + i(a >= 0.75) + i(a > 1.25) + i(a >= 1.75)
        + i(a > 2.5) + i(a >= 3.5) + i(a > 5.0)
    )
    return code


@lpregistry.register(
    lpop.encode, 
    mxfp4, 
    lpbackend.torch
)
def encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    b = x.reshape(-1, mxfp4.scale_block_size).float()
    s = _scales(b)
    y = (b / s).clamp(-6.0, 6.0)
    nib = _to_code(y.abs()) | (torch.signbit(y).to(torch.int32) << 3)
    nib = nib.to(torch.uint8).reshape(-1, mxfp4.scale_block_size // 2, 2)
    u8 = (nib[..., 0] << 4) | nib[..., 1]
    return u8.contiguous(), s.contiguous()


@lpregistry.register(
    lpop.decode, 
    mxfp4, 
    lpbackend.torch
)
def decode(u8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    mags = torch.tensor(mxfp4.magnitudes, dtype=torch.float32, device=u8.device)
    lut = torch.cat([mags, -mags])
    hi = lut[(u8 >> 4).long()]
    lo = lut[(u8 & 0x0F).long()]
    vals = torch.stack([hi, lo], dim=-1).reshape(u8.shape[0], -1)
    return vals * scales.reshape(-1, 1).float()