from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import fp4e2m1
from liblp.lpbackend import lpbackend
import ml_dtypes
import numpy as np
import logging

logging.getLogger(__name__)

@lpregistry.register(
    lpop.encode, 
    fp4e2m1,
    lpbackend.np,
)
def encode(x: np.ndarray):
    if __debug__:
        # check whether every dimension is even
        # implementation is not yet defined for odd dimension
        for i in range(len(x.shape)):
            if x.shape[i] % 2 != 0:
                logging.error(f"[FP4E2M1 | np] : shape at dimension {i} is not even, conversion is not defined for such shapes")


    return x.astype(ml_dtypes.float4_e2m1fn), None