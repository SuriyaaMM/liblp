import torch
import triton
import triton.language as tl
from typing import Tuple

from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import mxfp4
from liblp.lpbackend import lpbackend
from liblp.torch import mxfp4 as torch_mxfp4


@triton.jit
def _e2m1_decode(nib):
    code = nib & 0x7
    e = (code >> 1).to(tl.float32)
    m = (code & 1).to(tl.float32)
    mag = tl.where(e == 0.0, m * 0.5, (1.0 + 0.5 * m) * tl.exp2(e - 1.0))
    return tl.where(nib >= 8, -mag, mag)


@triton.autotune(
    configs=[
        triton.Config({"tile_n": 8, "tile_k": 128}, num_warps=4, num_stages=2),
        triton.Config({"tile_n": 16, "tile_k": 128}, num_warps=4, num_stages=2),
        triton.Config({"tile_n": 16, "tile_k": 256}, num_warps=4, num_stages=2),
        triton.Config({"tile_n": 32, "tile_k": 64}, num_warps=4, num_stages=2),
        triton.Config({"tile_n": 4, "tile_k": 256}, num_warps=4, num_stages=3),
    ],
    key=["n", "kp"],
)
@triton.jit
def __gemv_w4a16(
    w_ptr, s_ptr, x_ptr, o_ptr,
    n, kp, stride_w, stride_s,
    tile_n: tl.constexpr, tile_k: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    off_n = pid * tile_n + tl.arange(0, tile_n)
    mask_n = off_n < n
    acc = tl.zeros((tile_n,), dtype=tl.float32)

    for kb in tl.range(0, kp, tile_k):
        off_k = kb + tl.arange(0, tile_k)
        mask_k = off_k < kp
        mask = mask_n[:, None] & mask_k[None, :]

        w = tl.load(w_ptr + off_n[:, None] * stride_w + off_k[None, :], mask=mask, other=0)
        s = tl.load(s_ptr + off_n[:, None] * stride_s + (off_k // 16)[None, :],
                    mask=mask, other=0.0)
        x_e = tl.load(x_ptr + 2 * off_k, mask=mask_k, other=0.0).to(tl.float32)
        x_o = tl.load(x_ptr + 2 * off_k + 1, mask=mask_k, other=0.0).to(tl.float32)

        w = w.to(tl.int32)
        hi = _e2m1_decode((w >> 4) & 0x0F)
        lo = _e2m1_decode(w & 0x0F)

        part = hi * x_e[None, :] + lo * x_o[None, :]
        acc += tl.sum(part * s, axis=1)

    tl.store(o_ptr + off_n, acc, mask=mask_n)


@lpregistry.register(
    lpop.encode, 
    mxfp4,
    lpbackend.triton
)
def encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch_mxfp4.encode(x)


@lpregistry.register(
    lpop.decode, 
    mxfp4, 
    lpbackend.triton
)
def decode(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return torch_mxfp4.decode(x, scales)


@lpregistry.register(
    lpop.mv_w4a16, 
    mxfp4, 
    lpbackend.triton
)
def mv_w4a16(
    w_u8: torch.Tensor, 
    w_scales: torch.Tensor, 
    x: torch.Tensor
):
    n, kp = w_u8.shape
    x = x.contiguous()
    out = torch.empty(n, dtype=torch.float32, device=w_u8.device)
    grid = lambda meta: (triton.cdiv(n, meta["tile_n"]),)
    __gemv_w4a16[grid](w_u8, w_scales, x, out, n, kp, w_u8.stride(0), w_scales.stride(0))
    return out