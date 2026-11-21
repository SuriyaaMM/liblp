from abc import ABC, abstractmethod
import torch
import math
import numpy as np

class lpdtype(ABC):
    bits: int = -1
    name: str = "lpdtype"
    microscaling: bool = False

    def __str__(self):
        return f"{self.name}"

class fp4e2m1(lpdtype):
    bits = 4
    name = "fp4e2m1"
    microscaling = True

class mxi4fp32(lpdtype):
    # 32 values share a common fp32 scale
    # (4 * 32 + 32) / 32 => 5 bits per value on average
    bits = 5
    name = "mxi4fp32"
    microscaling = True
    scale_block_size = 32
    int4_min = -8
    int4_max = 7
    fp32_min = np.finfo(np.float32).min
    fp32_max = np.finfo(np.float32).max

class mxi8fp32(lpdtype):
    bits = 9
    name = "mxi8fp32"
    microscaling = True
    scale_block_size = 32
    int8_min = -128
    int8_max = 127
    fp32_min = np.finfo(np.float32).min
    fp32_max = np.finfo(np.float32).max

class mxfp4(lpdtype):
    bits = 5
    name = "mxfp4"
    microscaling = True
    scale_block_size = 32
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class fp8e4m3(lpdtype):
    bits = 8
    name = "fp8e4m3"
    microscaling = False

class fp8e5m2(lpdtype):
    bits = 8
    name = "fp8e5m2"
    microscaling = False

class lns8i4f3(lpdtype):
    bits = 8
    name = "lns8i4f3"
    microscaling = False

    NINF: np.ndarray = np.array(0xFF, dtype=np.uint8)
    INF: np.ndarray = np.array(0x7F, dtype=np.uint8)
    ZERO: np.ndarray = np.array(0x00, dtype=np.uint8)
    
    MAXE: np.ndarray = np.array(126, dtype=np.uint8)
    BIAS: np.ndarray = np.array(7, dtype=np.uint8)
    FBIT: np.ndarray = np.array(3, dtype=np.uint8)

    ADD: np.ndarray = np.round(
        np.log2(1 + 2 ** (-(np.arange(0, 128, dtype=np.float32) / 8.0))) * 8
    ).astype(np.int16)
    SUB: np.ndarray = np.round(
        np.log2(1 - 2 ** (-(np.arange(0, 128, dtype=np.float32) / 8.0))) * 8
    ).astype(np.int16)
    # log2(0) = 0, piecewise function
    SUB[0] = 0