from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import mxi4fp32
from liblp.lpbackend import lpbackend

import torch
import logging

logging.getLogger(__name__)

@lpregistry.register(
    lpop.encode, 
    mxi4fp32,
    lpbackend.torch,
)
def encode(x: torch.Tensor):
    if __debug__:
        for i in range(len(x.shape)):
            if (x.shape[i] < 32 or x.shape[i] % 32 != 0) and x.shape[0] != 1:
                logging.error(f"[MXI4FP32 | torch] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
                raise ValueError()
    # bwise := blockwise 
    x_bwise = x.ravel().reshape(-1, mxi4fp32.scale_block_size)
    # shape := scale_block_size, 1]
    bwise_max = torch.amax(torch.abs(x_bwise), dim=-1)[:, None] / mxi4fp32.int4_max
    x_bwise_q = torch.round(x_bwise / (bwise_max + torch.finfo(torch.float32).eps))
    x_bwise_q = torch.clip(x_bwise_q, mxi4fp32.int4_min, mxi4fp32.int4_max).to(torch.int8)
    x_bwise_q_u = x_bwise_q.to(torch.uint8)
    x_bwise_q_u_p = (
        ((x_bwise_q_u[:, 0::2] & 0x0F) << 4) |
        (x_bwise_q_u[:, 1::2] & 0x0F)
    )
    return x_bwise_q_u_p, bwise_max

@lpregistry.register(
    lpop.decode, 
    mxi4fp32,
    lpbackend.torch,
)
def decode(x: torch.Tensor, scales: torch.Tensor):
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | torch] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    x_bwise = x.ravel().reshape(-1, mxi4fp32.scale_block_size // 2)
    x_bwise_up = torch.empty(size=(x_bwise.shape[0], mxi4fp32.scale_block_size), dtype=torch.uint8, device=x.device)

    x_bwise_up[:, 0::2] = (x_bwise >> 4) & 0x0F
    x_bwise_up[:, 1::2] = (x_bwise) & 0x0F
    x_bwise_up = x_bwise_up.to(torch.int8)
    x_bwise_up = torch.where(x_bwise_up >= 8, x_bwise_up - 16, x_bwise_up)
    x_bwise_dequantised = x_bwise_up.to(torch.float32) * scales.to(torch.float32)
    return x_bwise_dequantised.ravel()

@lpregistry.register(
    lpop.add, 
    mxi4fp32,
    lpbackend.torch,
)
def add(a: torch.Tensor, a_scales: torch.Tensor, b: torch.Tensor, b_scales: torch.Tensor):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & torch.uint8
        b: (n_blocks, scales_block_size // 2) & torch.uint8
        a_scales: (n_blocks, 1) & torch.float32
        b_scales: (n_blocks, 1) & torch.float32
    """
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | torch] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    a_dq = decode(a, a_scales)
    b_dq = decode(b, b_scales)

    c_dq = a_dq + b_dq

    return encode(c_dq)

@lpregistry.register(
    lpop.mul, 
    mxi4fp32,
    lpbackend.torch,
)
def mul(a: torch.Tensor, a_scales: torch.Tensor, b: torch.Tensor, b_scales: torch.Tensor):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & torch.uint8
        b: (n_blocks, scales_block_size // 2) & torch.uint8
        a_scales: (n_blocks, 1) & torch.float32
        b_scales: (n_blocks, 1) & torch.float32
    """
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | torch] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    a_dq = decode(a, a_scales)
    b_dq = decode(b, b_scales)

    c_dq = a_dq * b_dq

    return encode(c_dq)

@lpregistry.register(
    lpop.sub, 
    mxi4fp32,
    lpbackend.torch,
)
def sub(a: torch.Tensor, a_scales: torch.Tensor, b: torch.Tensor, b_scales: torch.Tensor):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & torch.uint8
        b: (n_blocks, scales_block_size // 2) & torch.uint8
        a_scales: (n_blocks, 1) & torch.float32
        b_scales: (n_blocks, 1) & torch.float32
    """
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | torch] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    a_dq = decode(a, a_scales)
    b_dq = decode(b, b_scales)

    c_dq = a_dq - b_dq

    return encode(c_dq)

@lpregistry.register(
    lpop.div, 
    mxi4fp32,
    lpbackend.torch,
)
def div(a: torch.Tensor, a_scales: torch.Tensor, b: torch.Tensor, b_scales: torch.Tensor):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & torch.uint8
        b: (n_blocks, scales_block_size // 2) & torch.uint8
        a_scales: (n_blocks, 1) & torch.float32
        b_scales: (n_blocks, 1) & torch.float32
    """
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | torch] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    a_dq = decode(a, a_scales)
    b_dq = decode(b, b_scales)

    c_dq = a_dq / b_dq

    return encode(c_dq)

@lpregistry.register(
    lpop.matmul, 
    mxi4fp32,
    lpbackend.torch,
)
def matmul(a: torch.Tensor, a_scales: torch.Tensor, a_shape, b: torch.Tensor, b_scales: torch.Tensor, b_shape):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & torch.uint8
        b: (n_blocks, scales_block_size // 2) & torch.uint8
        a_scales: (n_blocks, 1) & torch.float32
        b_scales: (n_blocks, 1) & torch.float32
        a_shape: (m, k)
        b_shape: (n, k)
    """
    M, K = a_shape
    N, K = b_shape
    NB = K // mxi4fp32.scale_block_size

    def __unpack(packed_u8:torch.Tensor, shape):
        packed_u8f = packed_u8.flatten()
        shape_f = shape[0] * shape[1]
        out_i8 = torch.empty(size=(shape_f,), dtype=torch.int8, device=packed_u8.device)
        out_i8[0::2] = (packed_u8f >> 4).to(torch.int8)
        out_i8[1::2] = (packed_u8f & 0x0F).to(torch.int8)
        return torch.where(out_i8 >= 8, out_i8 - 16, out_i8).reshape(shape)

    c_shape = (M, N)
    c_fp32 = torch.zeros(size=c_shape, dtype=torch.float32, device=a.device)
    a_int8 = __unpack(a, a_shape)
    b_int8 = __unpack(b, b_shape)

    a_scales = a_scales.reshape(M, NB)
    b_scales = b_scales.reshape(N, NB)

    for i in range(NB):
        scales_mm = a_scales[:, i:i + 1] @ b_scales[:, i:i + 1].T
        bstart = i*mxi4fp32.scale_block_size
        bend = bstart + mxi4fp32.scale_block_size
        int_mm = torch._int_mm(a_int8[:, bstart:bend], b_int8[:, bstart:bend].T)
        c_fp32 += scales_mm * int_mm

    return encode(c_fp32)
