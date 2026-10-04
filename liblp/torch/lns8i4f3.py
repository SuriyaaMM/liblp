from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import lns8i4f3
from liblp.lpbackend import lpbackend
import torch

import numpy as np
import logging

logger = logging.getLogger(__name__)

def __lns8_i4f3_get_sign(x: torch.Tensor) -> torch.Tensor:
    # MSB is the sign bit
    return x >> 7


def __lns8_i4f3_get_exponent(x: torch.Tensor) -> torch.Tensor:
    # mask with 0111 1111
    return x & (0x7F)


@lpregistry.register(
    lpop.encode, 
    lns8i4f3,
    lpbackend.torch,
)
def encode(x: torch.Tensor):
    # extract the sign bit for all numbers
    s: torch.Tensor = torch.where((x > 0), 0, 1).to(torch.uint8)
    # unbiased exponent
    e_raw: torch.Tensor = torch.log2(torch.abs(x)).to(device=x.device)
    # clip the exponent between (0, MAXE)
    e: torch.Tensor = torch.clip(
        # extract the exponent & shift up the fractional part
        torch.round((e_raw + torch.from_numpy(lns8i4f3.BIAS).to(device=x.device)) * (1 << torch.from_numpy(lns8i4f3.FBIT).to(device=x.device))),
        torch.tensor(0).to(device=x.device),
        torch.from_numpy(lns8i4f3.MAXE).to(device=x.device),
    ).to(torch.uint8)
    # pack into 8-bit integer
    lns: torch.Tensor = (s << 7) | (e)

    # handle special values
    lns = torch.where((x == 0), torch.from_numpy(lns8i4f3.ZERO).to(device=x.device), lns)
    lns = torch.where(torch.isposinf(x), torch.from_numpy(lns8i4f3.INF).to(device=x.device), lns)
    lns = torch.where(torch.isneginf(x), torch.from_numpy(lns8i4f3.NINF).to(device=x.device), lns)

    return lns, None

@lpregistry.register(
    lpop.decode, 
    lns8i4f3,
    lpbackend.torch,
)
def decode(x: torch.Tensor):
    # extract the sign bit
    s: torch.Tensor = __lns8_i4f3_get_sign(x).to(torch.float32)

    # extract the exponents
    e: torch.Tensor = (
        # extract the exponent bit
        __lns8_i4f3_get_exponent(x)
        /
        # fractional scale factor (2 ^ LNBS8_I4F3_FBIT)
        torch.tensor(1 << torch.from_numpy(lns8i4f3.FBIT).to(device=x.device), dtype=torch.float32, device=x.device)
        # rebase
    ) - torch.tensor(torch.from_numpy(lns8i4f3.BIAS).to(device=x.device), dtype=torch.float32, device=x.device)

    # reconstruct the fp32
    fp32: torch.Tensor = ((-1) ** s) * (2**e)

    # handle special values
    fp32 = torch.where((x == torch.from_numpy(lns8i4f3.ZERO).to(device=x.device)), 0.0, fp32)
    fp32 = torch.where((x == torch.from_numpy(lns8i4f3.INF).to(device=x.device)), torch.inf, fp32)
    fp32 = torch.where((x == torch.from_numpy(lns8i4f3.NINF).to(device=x.device)), -torch.inf, fp32)

    return fp32

@lpregistry.register(
    lpop.mul, 
    lns8i4f3,
    lpbackend.torch,
)
def mul(x: torch.Tensor, y: torch.Tensor):
    s_x: torch.Tensor = __lns8_i4f3_get_sign(x)
    s_y: torch.Tensor = __lns8_i4f3_get_sign(y)

    e_x: torch.Tensor[torch.int16] = __lns8_i4f3_get_exponent(x).to(torch.int16)
    e_y: torch.Tensor[torch.int16] = __lns8_i4f3_get_exponent(y).to(torch.int16)

    s_xy = (s_x) ^ (s_y)
    # this operation doesn't fit entirely in torch.uint8
    e_xy = torch.clip(
        ((e_x) + (e_y) - (torch.from_numpy(lns8i4f3.BIAS).to(device=x.device) << torch.from_numpy(lns8i4f3.FBIT).to(device=x.device))), 0, torch.from_numpy(lns8i4f3.MAXE).to(device=x.device)
    ).to(torch.uint8)

    result = (s_xy << 7) | e_xy
    # explicitly handle zero propagation
    result = torch.where(
        (x == torch.from_numpy(lns8i4f3.ZERO).to(device=x.device)) | (y == torch.from_numpy(lns8i4f3.ZERO).to(device=x.device)), torch.from_numpy(lns8i4f3.ZERO).to(device=x.device), result
    )

    return result, None

@lpregistry.register(
    lpop.div, 
    lns8i4f3,
    lpbackend.torch,
)
def div(x: torch.Tensor, y: torch.Tensor):
    s_x: torch.Tensor = __lns8_i4f3_get_sign(x)
    s_y: torch.Tensor = __lns8_i4f3_get_sign(y)

    e_x: torch.Tensor[torch.int16] = __lns8_i4f3_get_exponent(x).to(torch.int16)
    e_y: torch.Tensor[torch.int16] = __lns8_i4f3_get_exponent(y).to(torch.int16)

    s_xy = (s_x) ^ (s_y)
    # this operation doesn't fit entirely in torch.uint8
    e_xy = torch.clip(
        ((e_x) - (e_y) + (torch.from_numpy(lns8i4f3.BIAS).to(device=x.device) << torch.from_numpy(lns8i4f3.FBIT).to(device=x.device))), 0, torch.from_numpy(lns8i4f3.MAXE).to(device=x.device)
    ).to(torch.uint8)

    result = (s_xy << 7) | e_xy

    # numerator = 0
    result = torch.where(x == torch.from_numpy(lns8i4f3.ZERO).to(device=x.device), torch.from_numpy(lns8i4f3.ZERO).to(device=x.device), result)
    # denominator = 0
    result = torch.where(
        y == torch.from_numpy(lns8i4f3.ZERO).to(device=x.device), torch.where(s_xy == 0, torch.from_numpy(lns8i4f3.INF).to(device=x.device), torch.from_numpy(lns8i4f3.NINF).to(device=x.device)), result
    )

    return result, None

@lpregistry.register(
    lpop.add, 
    lns8i4f3,
    lpbackend.torch,
)
def add(x: torch.Tensor, y: torch.Tensor):
    s_x: torch.Tensor = __lns8_i4f3_get_sign(x)
    s_y: torch.Tensor = __lns8_i4f3_get_sign(y)
    # this int32 is required for which I dont know the exact reason
    # it doesn't seem to work with int16.
    # TODO
    e_x: torch.Tensor = __lns8_i4f3_get_exponent(x).to(torch.int32)
    e_y: torch.Tensor = __lns8_i4f3_get_exponent(y).to(torch.int32)

    s_xy = torch.where(e_x >= e_y, s_x, s_y)

    d = torch.abs(e_x - e_y)

    # lookuptable addition & subtraction
    e_xy = torch.maximum(e_x, e_y) + torch.where(
        s_x == s_y, torch.from_numpy(lns8i4f3.ADD).to(device=x.device)[d], torch.from_numpy(lns8i4f3.SUB).to(device=x.device)[d]
    )

    e_xy = torch.clip(e_xy, 0, torch.from_numpy(lns8i4f3.MAXE).to(device=x.device)).to(torch.uint8)

    result = (s_xy << 7) | e_xy

    # exact cancellation X + (-X) = 0
    result = torch.where((s_x != s_y) & (d == 0), torch.from_numpy(lns8i4f3.ZERO).to(device=x.device), result)

    # x == 0, then answer is simply y
    result = torch.where(x == torch.from_numpy(lns8i4f3.ZERO).to(device=x.device), y, result)
    # y == 0, then answer is simply x
    result = torch.where(y == torch.from_numpy(lns8i4f3.ZERO).to(device=x.device), x, result)
    return result, None

@lpregistry.register(
    lpop.sub, 
    lns8i4f3,
    lpbackend.torch,
)
def sub(x: torch.Tensor, y: torch.Tensor):
    y_neg = torch.where(
        y == torch.from_numpy(lns8i4f3.ZERO).to(device=x.device),
        torch.from_numpy(lns8i4f3.ZERO).to(device=x.device),
        # sign bit mask = 0x80 = 1000 0000
        y ^ 0x80,
    )
    return add(x, y_neg)