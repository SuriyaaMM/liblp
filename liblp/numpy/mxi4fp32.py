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
                logging.error(f"[MXI4FP32 | np] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes")
    
    x_bwise = x.ravel().reshape(-1, 32)
    bwise_max = np.max(np.abs(x_bwise), axis=-1)[:, np.newaxis] / mxi4fp32.int4_max
    x_bwise_q = np.round(x_bwise / (bwise_max + np.finf(np.float32).eps))
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
    if __debug__:
        for i in range(len(x.shape)):
            if x.shape[i] < 32 or x.shape[i] % 32 != 0:
                logging.error(f"[MXI4FP32 | np] : shape at dimension {i} is not divisible by 32 (or) is less than 32, conversion is not defined for such shapes")
    
    x_bwise = x.ravel().reshape(-1, 32)
    x_bwise_up = np.empty(shape=(x.shape[0] * 2, 32), dtype=np.int8)

    x_bwise_up[:, 0::2] = (x_bwise_up & 0x0F) >> 4
    x_bwise_up[:, 1::2] = (x_bwise_up )
    x_bwise_dequantised = x_bwise * scales.astype(np.float32)
    return x_bwise_dequantised