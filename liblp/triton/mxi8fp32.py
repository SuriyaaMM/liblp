import torch
import triton
import triton.language as tl

from typing import Tuple

from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import mxi8fp32
from liblp.lpbackend import lpbackend

import liblp.torch.mxi8fp32 as torch_mxi8fp32

# ------------------------------------------------------------
# kernels
# ------------------------------------------------------------

@triton.jit
def __mm_mxi8fp32_accfp32(
    a_ptr, # [m, k]
    b_ptr, # [n, k]
    o_ptr, # [m, n]
    a_s_ptr, # [m, k // s_blk_size]
    b_s_ptr, # [n, k // s_blk_size]
    m: int,
    n: int,
    k: int,
    k_s: int,
    tile_m: tl.constexpr,
    tile_n: tl.constexpr,
    tile_k: tl.constexpr,
):
    # ty = tile_y
    # ty_b = tile_y's base
    # ty_o1d = tile_y's offset (unbroadcasted)
    # similar acronyms for tile_x

    tm_id = tl.program_id(axis=0)
    tn_id = tl.program_id(axis=1)

    tm_b = tm_id * k * tile_m
    tm_o1d = tm_b + tl.arange(0, tile_m) * k
    tm_m1d = tm_o1d < m * k

    tn_b = tn_id * k * tile_n
    tn_o1d = tn_b + tl.arange(0, tile_n) * k
    tn_m1d = tn_o1d < n * k

    tm_s_b = tm_id * k_s * tile_m
    tm_s_o1d = tm_s_b + tl.arange(0, tile_m) * k_s
    tm_s_m1d = tm_s_o1d < m * k_s

    tn_s_b = tn_id * k_s * tile_n
    tn_s_o1d = tn_s_b + tl.arange(0, tile_n) * k_s
    tn_s_m1d = tn_s_o1d < n * k_s

    acc = tl.zeros(shape=(tile_m, tile_n), dtype=tl.float32)

    for tk_id in range(0, k, tile_k):
        tk_s_id = tk_id // tile_k

        tk_o1d = tk_id + tl.arange(0, tile_k)
        tk_m1d = tk_o1d < k

        tk_s_o1d = tk_s_id + tl.arange(0, 1)
        tk_s_m1d = tk_s_o1d < k_s
        
        a_o2d = tm_o1d[:, None] + tk_o1d[None, :]
        a_m2d = tm_m1d[:, None] & tk_m1d[None, :]

        a_s_o2d = tm_s_o1d[:, None] + tk_s_o1d[None, :]
        a_s_m2d = tm_s_m1d[:, None] & tk_s_m1d[None, :]
        
        b_o2d = tn_o1d[:, None] + tk_o1d[None, :]
        b_m2d = tn_m1d[:, None] & tk_m1d[None, :]

        b_s_o2d = tn_s_o1d[:, None] + tk_s_o1d[None, :]
        b_s_m2d = tn_s_m1d[:, None] & tk_s_m1d[None, :]

        a = tl.load(
            pointer=a_ptr + a_o2d,
            mask=a_m2d,
            other=0,
        )
        a = tl.cast(a, tl.int8)
        
        b = tl.load(
            pointer=b_ptr + b_o2d,
            mask=b_m2d,
            other=0,
        )
        b = tl.cast(b, tl.int8)

        a_s = tl.load(
            pointer=a_s_ptr + a_s_o2d,
            mask=a_s_m2d,
            other=0.0,
        )
        a_s = tl.cast(a_s, tl.float32)

        b_s = tl.load(
            pointer=b_s_ptr + b_s_o2d,
            mask=b_s_m2d,
            other=0.0,
        )
        b_s = tl.cast(b_s, tl.float32)

        p_dot = tl.dot(
            input=a,
            other=tl.trans(b),
        )
        s_dot = a_s * tl.trans(b_s)

        acc += (p_dot) * s_dot

    to_m_o1d = tm_id * tile_m * n + tl.arange(0, tile_m) * n
    to_m_m1d = to_m_o1d < m * n

    to_n_o1d = tn_id * tile_n + tl.arange(0, tile_n)
    to_n_m1d = to_n_o1d < n

    o_o2d = to_m_o1d[:, None] + to_n_o1d[None, :]
    o_m2d = to_m_m1d[:, None] & to_n_m1d[None, :]

    tl.store(
        pointer=o_ptr + o_o2d,
        value=acc,
        mask=o_m2d,
    )
    
@triton.jit
def __mm_mxi8fp32_accfp16(
    a_ptr, # [m, k]
    b_ptr, # [n, k]
    o_ptr, # [m, n]
    a_s_ptr, # [m, k // s_blk_size]
    b_s_ptr, # [n, k // s_blk_size]
    m: int,
    n: int,
    k: int,
    k_s: int,
    tile_m: tl.constexpr,
    tile_n: tl.constexpr,
    tile_k: tl.constexpr,
):
    # ty = tile_y
    # ty_b = tile_y's base
    # ty_o1d = tile_y's offset (unbroadcasted)
    # similar acronyms for tile_x

    tm_id = tl.program_id(axis=0)
    tn_id = tl.program_id(axis=1)

    tm_b = tm_id * k * tile_m
    tm_o1d = tm_b + tl.arange(0, tile_m) * k
    tm_m1d = tm_o1d < m * k

    tn_b = tn_id * k * tile_n
    tn_o1d = tn_b + tl.arange(0, tile_n) * k
    tn_m1d = tn_o1d < n * k

    tm_s_b = tm_id * k_s * tile_m
    tm_s_o1d = tm_s_b + tl.arange(0, tile_m) * k_s
    tm_s_m1d = tm_s_o1d < m * k_s

    tn_s_b = tn_id * k_s * tile_n
    tn_s_o1d = tn_s_b + tl.arange(0, tile_n) * k_s
    tn_s_m1d = tn_s_o1d < n * k_s

    acc = tl.zeros(shape=(tile_m, tile_n), dtype=tl.float16)

    for tk_id in range(0, k, tile_k):
        tk_s_id = tk_id // tile_k

        tk_o1d = tk_id + tl.arange(0, tile_k)
        tk_m1d = tk_o1d < k

        tk_s_o1d = tk_s_id + tl.arange(0, 1)
        tk_s_m1d = tk_s_o1d < k_s
        
        a_o2d = tm_o1d[:, None] + tk_o1d[None, :]
        a_m2d = tm_m1d[:, None] & tk_m1d[None, :]

        a_s_o2d = tm_s_o1d[:, None] + tk_s_o1d[None, :]
        a_s_m2d = tm_s_m1d[:, None] & tk_s_m1d[None, :]
        
        b_o2d = tn_o1d[:, None] + tk_o1d[None, :]
        b_m2d = tn_m1d[:, None] & tk_m1d[None, :]

        b_s_o2d = tn_s_o1d[:, None] + tk_s_o1d[None, :]
        b_s_m2d = tn_s_m1d[:, None] & tk_s_m1d[None, :]

        a = tl.load(
            pointer=a_ptr + a_o2d,
            mask=a_m2d,
            other=0,
        )
        a = tl.cast(a, tl.int8)
        
        b = tl.load(
            pointer=b_ptr + b_o2d,
            mask=b_m2d,
            other=0,
        )
        b = tl.cast(b, tl.int8)

        a_s = tl.load(
            pointer=a_s_ptr + a_s_o2d,
            mask=a_s_m2d,
            other=0.0,
        )
        a_s = tl.cast(a_s, tl.float32)

        b_s = tl.load(
            pointer=b_s_ptr + b_s_o2d,
            mask=b_s_m2d,
            other=0.0,
        )
        b_s = tl.cast(b_s, tl.float32)

        p_dot = tl.dot(
            input=a,
            other=tl.trans(b),
        )
        s_dot = a_s * tl.trans(b_s)

        acc += tl.cast(input=(p_dot) * s_dot, dtype=tl.float16)

    to_m_o1d = tm_id * tile_m * n + tl.arange(0, tile_m) * n
    to_m_m1d = to_m_o1d < m * n

    to_n_o1d = tn_id * tile_n + tl.arange(0, tile_n)
    to_n_m1d = to_n_o1d < n

    o_o2d = to_m_o1d[:, None] + to_n_o1d[None, :]
    o_m2d = to_m_m1d[:, None] & to_n_m1d[None, :]

    tl.store(
        pointer=o_ptr + o_o2d,
        value=acc,
        mask=o_m2d,
    )

# ------------------------------------------------------------
# utility functions
# ------------------------------------------------------------

@lpregistry.register(
    lpop.encode,
    mxi8fp32,
    lpbackend.triton,
)
def encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch_mxi8fp32.encode(x)

@lpregistry.register(
    lpop.decode,
    mxi8fp32,
    lpbackend.triton,
)
def decode(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return torch_mxi8fp32.decode(x, scales)

@lpregistry.register(
    lpop.mm_a32,
    mxi8fp32,
    lpbackend.triton,
)
def mm_a32(
    a: torch.Tensor, a_scales: torch.Tensor,
    b: torch.Tensor, b_scales: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n, _ = b.shape

    tile_m = 32
    tile_n = 32

    launch_params = (triton.cdiv(m, tile_m), triton.cdiv(n, tile_n))

    o = torch.zeros(size=(m, n), dtype=torch.float32, device=a.device)

    __mm_mxi8fp32_accfp32[launch_params](
        a_ptr=a,
        b_ptr=b,
        o_ptr=o,
        a_s_ptr=a_scales,
        b_s_ptr=b_scales,
        m=m, n=n, k=k, k_s=k // mxi8fp32.scale_block_size,
        tile_m=tile_m, tile_n=tile_n, tile_k=mxi8fp32.scale_block_size,
    )
    return o

@lpregistry.register(
    lpop.mm_a16,
    mxi8fp32,
    lpbackend.triton,
)
def mm_a16(
    a: torch.Tensor, a_scales: torch.Tensor,
    b: torch.Tensor, b_scales: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n, _ = b.shape

    tile_m = 32
    tile_n = 32

    launch_params = (triton.cdiv(m, tile_m), triton.cdiv(n, tile_n))

    o = torch.zeros(size=(m, n), dtype=torch.float16, device=a.device)

    __mm_mxi8fp32_accfp16[launch_params](
        a_ptr=a,
        b_ptr=b,
        o_ptr=o,
        a_s_ptr=a_scales,
        b_s_ptr=b_scales,
        m=m, n=n, k=k, k_s=k // mxi8fp32.scale_block_size,
        tile_m=tile_m, tile_n=tile_n, tile_k=mxi8fp32.scale_block_size,
    )
    return o

