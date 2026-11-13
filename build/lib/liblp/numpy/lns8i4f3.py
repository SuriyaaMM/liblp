from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import lns8i4f3
from liblp.lpbackend import lpbackend

import numpy as np
import logging

logger = logging.getLogger(__name__)

def __lns8_i4f3_get_sign(x: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    # MSB is the sign bit
    return x >> 7


def __lns8_i4f3_get_exponent(x: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    # mask with 0111 1111
    return x & (0x7F)


@lpregistry.register(
    lpop.encode, 
    lns8i4f3,
    lpbackend.np,
)
def encode(x: np.ndarray):
    # extract the sign bit for all numbers
    s: np.ndarray[np.uint8] = np.where((x > 0), 0, 1).astype(np.uint8)
    # unbiased exponent
    e_raw: np.ndarray[np.float32] = np.log2(np.abs(x))
    # clip the exponent between (0, MAXE)
    e: np.ndarray[np.uint8] = np.clip(
        # extract the exponent & shift up the fractional part
        a=np.round((e_raw + lns8i4f3.BIAS) * (1 << lns8i4f3.FBIT)),
        a_min=0,
        a_max=lns8i4f3.MAXE,
    ).astype(np.uint8)
    # pack into 8-bit integer
    lns: np.ndarray[np.uint8] = (s << 7) | (e)

    # handle special values
    lns = np.where((x == 0), lns8i4f3.ZERO, lns)
    lns = np.where(np.isposinf(x), lns8i4f3.INF, lns)
    lns = np.where(np.isneginf(x), lns8i4f3.NINF, lns)

    return lns, None

@lpregistry.register(
    lpop.decode, 
    lns8i4f3,
    lpbackend.np,
)
def decode(x: np.ndarray):
    # extract the sign bit
    s: np.ndarray[np.float32] = __lns8_i4f3_get_sign(x).astype(np.float32)

    # extract the exponents
    e: np.ndarray[np.float32] = (
        # extract the exponent bit
        __lns8_i4f3_get_exponent(x)
        /
        # fractional scale factor (2 ^ LNBS8_I4F3_FBIT)
        np.array(1 << lns8i4f3.FBIT, dtype=np.float32)
        # rebase
    ) - np.array(lns8i4f3.BIAS, dtype=np.float32)

    # reconstruct the fp32
    fp32: np.ndarray[np.float32] = ((-1) ** s) * (2**e)

    # handle special values
    fp32 = np.where((x == lns8i4f3.ZERO), 0.0, fp32)
    fp32 = np.where((x == lns8i4f3.INF), np.inf, fp32)
    fp32 = np.where((x == lns8i4f3.NINF), -np.inf, fp32)

    return fp32

@lpregistry.register(
    lpop.mul, 
    lns8i4f3,
    lpbackend.np,
)
def mul(x: np.ndarray[np.uint8], y: np.ndarray[np.uint8]):
    s_x: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(x)
    s_y: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(y)

    e_x: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(x).astype(np.int16)
    e_y: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(y).astype(np.int16)

    s_xy = (s_x) ^ (s_y)
    # this operation doesn't fit entirely in np.uint8
    e_xy = np.clip(
        ((e_x) + (e_y) - (lns8i4f3.BIAS << lns8i4f3.FBIT)), 0, lns8i4f3.MAXE
    ).astype(np.uint8)

    result = (s_xy << 7) | e_xy
    # explicitly handle zero propagation
    result = np.where(
        (x == lns8i4f3.ZERO) | (y == lns8i4f3.ZERO), lns8i4f3.ZERO, result
    )

    return result, None

@lpregistry.register(
    lpop.div, 
    lns8i4f3,
    lpbackend.np,
)
def div(x: np.ndarray[np.uint8], y: np.ndarray[np.uint8]):
    s_x: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(x)
    s_y: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(y)

    e_x: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(x).astype(np.int16)
    e_y: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(y).astype(np.int16)

    s_xy = (s_x) ^ (s_y)
    # this operation doesn't fit entirely in np.uint8
    e_xy = np.clip(
        ((e_x) - (e_y) + (lns8i4f3.BIAS << lns8i4f3.FBIT)), 0, lns8i4f3.MAXE
    ).astype(np.uint8)

    result = (s_xy << 7) | e_xy

    # numerator = 0
    result = np.where(x == lns8i4f3.ZERO, lns8i4f3.ZERO, result)
    # denominator = 0
    result = np.where(
        y == lns8i4f3.ZERO, np.where(s_xy == 0, lns8i4f3.INF, lns8i4f3.NINF), result
    )

    return result, None

@lpregistry.register(
    lpop.add, 
    lns8i4f3,
    lpbackend.np,
)
def add(x: np.ndarray[np.uint8], y: np.ndarray[np.uint8]):
    s_x: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(x)
    s_y: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(y)

    e_x: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(x).astype(np.int16)
    e_y: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(y).astype(np.int16)

    s_xy = np.where(e_x >= e_y, s_x, s_y)

    d = np.abs(e_x - e_y)

    # lookuptable addition & subtraction
    e_xy = np.maximum(e_x, e_y) + np.where(
        s_x == s_y, lns8i4f3.ADD[d], lns8i4f3.SUB[d]
    )

    e_xy = np.clip(e_xy, 0, lns8i4f3.MAXE).astype(np.uint8)

    result = (s_xy << 7) | e_xy

    # exact cancellation X + (-X) = 0
    result = np.where((s_x != s_y) & (d == 0), lns8i4f3.ZERO, result)

    # x == 0, then answer is simply y
    result = np.where(x == lns8i4f3.ZERO, y, result)
    # y == 0, then answer is simply x
    result = np.where(y == lns8i4f3.ZERO, x, result)
    return result, None

@lpregistry.register(
    lpop.sub, 
    lns8i4f3,
    lpbackend.np,
)
def sub(x: np.ndarray[np.uint8], y: np.ndarray[np.uint8]):
    y_neg = np.where(
        y == lns8i4f3.ZERO,
        lns8i4f3.ZERO,
        # sign bit mask = 0x80 = 1000 0000
        y ^ 0x80,
    )
    return add(x, y_neg)