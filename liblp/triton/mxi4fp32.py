import torch
import triton
import triton.language as tl

from typing import Tuple

from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import mxi4fp32
from liblp.lpbackend import lpbackend
from liblp.torch import mxi4fp32 as torch_mxi4fp32

# ------------------------------------------------------------
# kernels
# ------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"tile_y": 64,  "tile_x": 64,  "tile_b": 32}, num_warps=4, num_stages=2),
        triton.Config({"tile_y": 128, "tile_x": 64,  "tile_b": 32}, num_warps=4, num_stages=2),
        triton.Config({"tile_y": 128, "tile_x": 128, "tile_b": 32}, num_warps=8, num_stages=2),
        triton.Config({"tile_y": 64,  "tile_x": 128, "tile_b": 64}, num_warps=4, num_stages=2),
        triton.Config({"tile_y": 16,  "tile_x": 64,  "tile_b": 64}, num_warps=4, num_stages=3),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def __mm(
    a_u8_ptr: torch.Tensor,
    a_fp32_ptr: torch.Tensor,
    b_u8_ptr: torch.Tensor,
    b_fp32_ptr: torch.Tensor,
    o_fp32_ptr: torch.Tensor,
    m: int,
    n: int,
    k: int,
    stride_ay: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_asy: tl.constexpr,
    stride_asb: tl.constexpr,
    stride_bx: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bsx: tl.constexpr,
    stride_bsb: tl.constexpr,
    stride_oy: tl.constexpr,
    stride_ox: tl.constexpr,
    scale_b: tl.constexpr,  
    tile_y: tl.constexpr,
    tile_x: tl.constexpr,
    tile_b: tl.constexpr,
):
    tile_y_id = tl.program_id(axis=0)
    tile_x_id = tl.program_id(axis=1)

    off_y = tile_y_id * tile_y + tl.arange(0, tile_y)
    off_x = tile_x_id * tile_x + tl.arange(0, tile_x)
    mask_y = off_y < m
    mask_x = off_x < n

    tile_acc = tl.zeros(shape=(tile_y, tile_x), dtype=tl.float32)

    num_b_blocks = tl.cdiv(k, tile_b)
    for b_idx in tl.range(0, num_b_blocks):
        b_start = b_idx * tile_b
        off_b = b_start + tl.arange(0, tile_b)
        off_s = off_b // scale_b
        mask_b = off_b < k

        a_tile_ptr = (off_y[:, None] * stride_ay) + (off_b[None, :] * stride_ab)
        b_tile_ptr = (off_x[:, None] * stride_bx) + (off_b[None, :] * stride_bb)
        a_scale_ptr = (off_y[:, None] * stride_asy) + (off_s[None, :] * stride_asb)
        b_scale_ptr = (off_x[:, None] * stride_bsx) + (off_s[None, :] * stride_bsb)

        mask_a = mask_y[:, None] & mask_b[None, :]
        mask_bb = mask_x[:, None] & mask_b[None, :]

        a_u8 = tl.load(pointer=a_u8_ptr + a_tile_ptr, mask=mask_a, other=0)
        b_u8 = tl.load(pointer=b_u8_ptr + b_tile_ptr, mask=mask_bb, other=0)
        a_fp32 = tl.load(pointer=a_fp32_ptr + a_scale_ptr, mask=mask_a, other=0.0)
        b_fp32 = tl.load(pointer=b_fp32_ptr + b_scale_ptr, mask=mask_bb, other=0.0)

        a_hi = ((a_u8 >> 4) & 0x0F).to(tl.int8)
        a_lo = (a_u8 & 0x0F).to(tl.int8)
        a_hi = tl.where(a_hi >= 8, a_hi - 16, a_hi)
        a_lo = tl.where(a_lo >= 8, a_lo - 16, a_lo)

        b_hi = ((b_u8 >> 4) & 0x0F).to(tl.int8)
        b_lo = (b_u8 & 0x0F).to(tl.int8)
        b_hi = tl.where(b_hi >= 8, b_hi - 16, b_hi)
        b_lo = tl.where(b_lo >= 8, b_lo - 16, b_lo)

        a_hi_f16 = (a_hi.to(tl.float32) * a_fp32).to(tl.float16)
        a_lo_f16 = (a_lo.to(tl.float32) * a_fp32).to(tl.float16)
        b_hi_f16 = (b_hi.to(tl.float32) * b_fp32).to(tl.float16)
        b_lo_f16 = (b_lo.to(tl.float32) * b_fp32).to(tl.float16)

        tile_acc = tl.dot(a_hi_f16, tl.trans(b_hi_f16), tile_acc, out_dtype=tl.float32)
        tile_acc = tl.dot(a_lo_f16, tl.trans(b_lo_f16), tile_acc, out_dtype=tl.float32)

    tile_out_ptr = (off_y[:, None] * stride_oy) + (off_x[None, :] * stride_ox)

    tl.store(
        pointer=o_fp32_ptr + tile_out_ptr,
        value=tile_acc,
        mask=mask_y[:, None] & mask_x[None, :],
    )

@triton.autotune(
    configs=[
        triton.Config({"tile_x": 16, "tile_b": 128}, num_warps=4, num_stages=2),
        triton.Config({"tile_x": 32, "tile_b": 64},  num_warps=4, num_stages=2),
        triton.Config({"tile_x": 32, "tile_b": 128}, num_warps=4, num_stages=2),
        triton.Config({"tile_x": 64, "tile_b": 64},  num_warps=8, num_stages=2),
        triton.Config({"tile_x": 8,  "tile_b": 256}, num_warps=4, num_stages=3),
    ],
    key=["n", "k"],
)
@triton.jit
def __gemv(
    a_u8_ptr: torch.Tensor,
    a_fp32_ptr: torch.Tensor,
    b_u8_ptr: torch.Tensor,
    b_fp32_ptr: torch.Tensor,
    o_fp32_ptr: torch.Tensor,
    n: int,
    k: int,
    stride_bx: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bsx: tl.constexpr,
    stride_bsb: tl.constexpr,
    stride_ox: tl.constexpr,
    scale_b: tl.constexpr,  
    tile_x: tl.constexpr,
    tile_b: tl.constexpr,
):
    tile_x_id = tl.program_id(axis=0)

    off_x = tile_x_id * tile_x + tl.arange(0, tile_x)
    mask_x = off_x < n

    tile_acc = tl.zeros(shape=(tile_x,), dtype=tl.float32)

    num_b_blocks = tl.cdiv(k, tile_b)
    for b_idx in tl.range(0, num_b_blocks):
        b_start = b_idx * tile_b
        off_b = b_start + tl.arange(0, tile_b)
        off_s = off_b // scale_b
        mask_b = off_b < k

        b_tile_ptr = (off_x[:, None] * stride_bx) + (off_b[None, :] * stride_bb)
        b_scale_ptr = (off_x[:, None] * stride_bsx) + (off_s[None, :] * stride_bsb)
        mask_tile = mask_x[:, None] & mask_b[None, :]
        
        a_u8 = tl.load(pointer=a_u8_ptr + off_b, mask=mask_b, other=0)
        a_fp32 = tl.load(pointer=a_fp32_ptr + off_s, mask=mask_b, other=0.0)
        b_u8 = tl.load(pointer=b_u8_ptr + b_tile_ptr, mask=mask_tile, other=0)
        b_fp32 = tl.load(pointer=b_fp32_ptr + b_scale_ptr, mask=mask_tile, other=0.0)

        a_hi = ((a_u8 >> 4) & 0x0F).to(tl.int8)
        a_lo = (a_u8 & 0x0F).to(tl.int8)
        a_hi = tl.where(a_hi >= 8, a_hi - 16, a_hi)
        a_lo = tl.where(a_lo >= 8, a_lo - 16, a_lo)

        b_hi = ((b_u8 >> 4) & 0x0F).to(tl.int8)
        b_lo = (b_u8 & 0x0F).to(tl.int8)
        b_hi = tl.where(b_hi >= 8, b_hi - 16, b_hi)
        b_lo = tl.where(b_lo >= 8, b_lo - 16, b_lo)

        a_hi_fp32 = a_hi.to(tl.float32) * a_fp32
        a_lo_fp32 = a_lo.to(tl.float32) * a_fp32
        b_hi_fp32 = b_hi.to(tl.float32) * b_fp32
        b_lo_fp32 = b_lo.to(tl.float32) * b_fp32

        tile_acc += tl.sum(b_hi_fp32 * a_hi_fp32[None, :], axis=1)
        tile_acc += tl.sum(b_lo_fp32 * a_lo_fp32[None, :], axis=1)

    tl.store(
        pointer=o_fp32_ptr + (off_x * stride_ox),
        value=tile_acc,
        mask=mask_x,
    )

# ------------------------------------------------------------
# utility functions
# ------------------------------------------------------------

@lpregistry.register(
    lpop.encode,
    mxi4fp32,
    lpbackend.triton,
)
def encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch_mxi4fp32.encode(x)

@lpregistry.register(
    lpop.decode,
    mxi4fp32,
    lpbackend.triton,
)
def decode(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return torch_mxi4fp32.decode(x, scales)


@lpregistry.register(
    lpop.matmul,
    mxi4fp32,
    lpbackend.triton,
)
def matmul(
    a_u8: torch.Tensor,
    a_fp32: torch.Tensor,
    a_shape: Tuple[int, int],
    b_u8: torch.Tensor,
    b_fp32: torch.Tensor,
    b_shape: Tuple[int, int],
    tile_y: int = 32,
    tile_b: int = mxi4fp32.scale_block_size // 2,
):
    a_u8 = a_u8.contiguous()
    b_u8 = b_u8.contiguous()
    a_fp32 = a_fp32.contiguous()
    b_fp32 = b_fp32.contiguous()

    m, ka = a_shape
    n, kb = b_shape
    k = ka

    if __debug__:
        # reduction dimension check
        if ka != kb:
            raise ValueError(f"a(shape={a_shape}) & b(shape={b_shape}) are not compatible for matrix multiplication")
        # reshape dimension check
        if m * (k // 2) != a_u8.shape[0] * a_u8.shape[1]:
            raise ValueError(f"a_u8(shape={a_u8.shape}) is not resizable into shape=({m}, {k // 2})")
        if n * (k // 2) != b_u8.shape[0] * b_u8.shape[1]:
            raise ValueError(f"b_u8(shape={a_u8.shape}) is not resizable into shape=({n}, {k // 2})")
        if k % mxi4fp32.scale_block_size:
            raise ValueError(f"k({k}) should be divisible by scale({mxi4fp32.scale_block_size})")

    a_u8 = a_u8.reshape(m, k // 2)
    b_u8 = b_u8.reshape(n, k // 2)
    a_fp32 = a_fp32.reshape(m, k // mxi4fp32.scale_block_size)
    b_fp32 = b_fp32.reshape(n, k // mxi4fp32.scale_block_size)

    if m == 1:
        o_fp32 = torch.empty(
            size=(m, n),
            dtype=torch.float32,
            device=a_u8.device,
            memory_format=torch.contiguous_format,
        )

        launch_parameters = lambda meta: (triton.cdiv(n, meta["tile_x"]),)

        __gemv[launch_parameters](
            a_u8, a_fp32,
            b_u8, b_fp32,
            o_fp32,
            n, k // 2,
            b_u8.stride(0), b_u8.stride(1),
            b_fp32.stride(0), b_fp32.stride(1),
            o_fp32.stride(1),
            scale_b=mxi4fp32.scale_block_size // 2,
        )
        return o_fp32

    o_fp32 = torch.empty(
        size=(m, n),
        dtype=torch.float32,
        device=a_u8.device,
        memory_format=torch.contiguous_format,
    )

    launch_parameters = lambda meta: (
        triton.cdiv(m, meta["tile_y"]),
        triton.cdiv(n, meta["tile_x"]),
    )

    __mm[launch_parameters](
        a_u8, a_fp32,
        b_u8, b_fp32,
        o_fp32,
        m, n, k // 2,
        a_u8.stride(0), a_u8.stride(1),
        a_fp32.stride(0), a_fp32.stride(1),
        b_u8.stride(0), b_u8.stride(1),
        b_fp32.stride(0), b_fp32.stride(1),
        o_fp32.stride(0), o_fp32.stride(1),
        mxi4fp32.scale_block_size // 2,
    )
    return o_fp32