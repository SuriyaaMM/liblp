from abc import ABC, abstractmethod
import torch
import math
import numpy as np

class lpdtype(ABC):
    bits: int = None
    name: str = "lpdtype"

    def __str__(self):
        return f"{self.name}"

class fp4e2m1(lpdtype):
    bits = 4
    name = "fp4e2m1"

class fp8e4m3(lpdtype):
    bits = 8
    name = "fp8e4m3"

class fp8e5m2(lpdtype):
    bits = 8
    name = "fp8e5m2"

class lns8i4f3(lpdtype):
    bits = 8
    name = "lns8i4f3"

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