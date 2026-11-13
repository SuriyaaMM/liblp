from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import lns8i4f3
from liblp.lpbackend import lpbackend

import numpy as np
import triton.language as tl
import logging
import torch

logger = logging.getLogger(__name__)

@lpregistry.register(
    lpop.encode, 
    lns8i4f3,
    lpbackend.triton,
)
def encode(x: torch.Tensor):
    # extract the sign bit for all numbers
    s: torch.Tensor[torch.uint8] = torch.where((x > 0), 0, 1).to(np.uint8)
    # unbiased exponent
    e_raw: torch.Tensor[torch.float32] = torch.log2(torch.abs(x))
    # clip the exponent between (0, MAXE)
    e: torch.Tensor[torch.uint8] = torch.clip(
        # extract the exponent & shift up the fractional part
        a=torch.round((e_raw + lns8i4f3.BIAS) * (1 << lns8i4f3.FBIT)),
        a_min=0,
        a_max=lns8i4f3.MAXE,
    ).astype(np.uint8)
    # pack into 8-bit integer
    lns: torch.Tensor[torch.uint8] = (s << 7) | (e)

    # handle special values
    lns = torch.where((x == 0), lns8i4f3.ZERO, lns)
    lns = torch.where(np.isposinf(x), lns8i4f3.INF, lns)
    lns = torch.where(np.isneginf(x), lns8i4f3.NINF, lns)

    return lns, None
