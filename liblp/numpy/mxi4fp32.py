from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import mxi4fp32
from liblp.lpbackend import lpbackend

import numpy as np
import logging

logging.getLogger(__name__)

@lpregistry.register(
    lpop.encode, 
    mxi4fp32,
    lpbackend.np,
)
def encode(x: np.ndarray):
    if __debug__:
        for i in range(len(x.shape)):
            if x.shape[i] < 32 or x.shape[i] % 32 != 0:
                logging.error(f"[MXI4FP32 | np] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
                raise ValueError()
    # bwise := blockwise 
    x_bwise = x.ravel().reshape(-1, mxi4fp32.scale_block_size)
    # shape := scale_block_size, 1]
    bwise_max = np.max(np.abs(x_bwise), axis=-1)[:, np.newaxis] / mxi4fp32.int4_max
    x_bwise_q = np.round(x_bwise / (bwise_max + np.finfo(np.float32).eps))
    x_bwise_q = np.clip(x_bwise_q, mxi4fp32.int4_min, mxi4fp32.int4_max).astype(np.int8)
    x_bwise_q_u = x_bwise_q.astype(np.uint8)
    x_bwise_q_u_p = (
        ((x_bwise_q_u[:, 0::2] & 0x0F) << 4) |
        (x_bwise_q_u[:, 1::2] & 0x0F)
    )
    return x_bwise_q_u_p, bwise_max

@lpregistry.register(
    lpop.decode, 
    mxi4fp32,
    lpbackend.np,
)
def decode(x: np.ndarray, scales: np.ndarray):
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | np] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    x_bwise = x.ravel().reshape(-1, mxi4fp32.scale_block_size // 2)
    x_bwise_up = np.empty(shape=(x_bwise.shape[0], mxi4fp32.scale_block_size), dtype=np.uint8)

    x_bwise_up[:, 0::2] = (x_bwise >> 4) & 0x0F
    x_bwise_up[:, 1::2] = (x_bwise) & 0x0F
    x_bwise_up = x_bwise_up.astype(np.int8)
    x_bwise_up = np.where(x_bwise_up >= 8, x_bwise_up - 16, x_bwise_up)
    x_bwise_dequantised = x_bwise_up.astype(np.float32) * scales.astype(np.float32)
    return x_bwise_dequantised.ravel()

@lpregistry.register(
    lpop.add, 
    mxi4fp32,
    lpbackend.np,
)
def add(a: np.ndarray, a_scales: np.ndarray, b: np.ndarray, b_scales: np.ndarray):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & np.uint8
        b: (n_blocks, scales_block_size // 2) & np.uint8
        a_scales: (n_blocks, 1) & np.float32
        b_scales: (n_blocks, 1) & np.float32
    """
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | np] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    a_dq = decode(a, a_scales)
    b_dq = decode(b, b_scales)

    c_dq = a_dq + b_dq

    return encode(c_dq)

@lpregistry.register(
    lpop.mul, 
    mxi4fp32,
    lpbackend.np,
)
def mul(a: np.ndarray, a_scales: np.ndarray, b: np.ndarray, b_scales: np.ndarray):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & np.uint8
        b: (n_blocks, scales_block_size // 2) & np.uint8
        a_scales: (n_blocks, 1) & np.float32
        b_scales: (n_blocks, 1) & np.float32
    """
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | np] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    a_dq = decode(a, a_scales)
    b_dq = decode(b, b_scales)

    c_dq = a_dq * b_dq

    return encode(c_dq)

@lpregistry.register(
    lpop.sub, 
    mxi4fp32,
    lpbackend.np,
)
def sub(a: np.ndarray, a_scales: np.ndarray, b: np.ndarray, b_scales: np.ndarray):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & np.uint8
        b: (n_blocks, scales_block_size // 2) & np.uint8
        a_scales: (n_blocks, 1) & np.float32
        b_scales: (n_blocks, 1) & np.float32
    """
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | np] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    a_dq = decode(a, a_scales)
    b_dq = decode(b, b_scales)

    c_dq = a_dq - b_dq

    return encode(c_dq)

@lpregistry.register(
    lpop.div, 
    mxi4fp32,
    lpbackend.np,
)
def div(a: np.ndarray, a_scales: np.ndarray, b: np.ndarray, b_scales: np.ndarray):
    """
    Expected shapes & dtype:
        a: (n_blocks, scales_block_size // 2) & np.uint8
        b: (n_blocks, scales_block_size // 2) & np.uint8
        a_scales: (n_blocks, 1) & np.float32
        b_scales: (n_blocks, 1) & np.float32
    """
    # if __debug__:
    #     for i in range(len(x.shape)):
    #         if x.shape[i] < 32 or x.shape[i] % 32 != 0:
    #             logging.error(f"[MXI4FP32 | np] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes ({x.shape})")
    
    a_dq = decode(a, a_scales)
    b_dq = decode(b, b_scales)

    c_dq = a_dq / b_dq

    return encode(c_dq)