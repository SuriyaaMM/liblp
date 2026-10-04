from liblp.lpdtype import *
from liblp.lpbackend import lpbackend
from liblp.lpregistry import lpregistry
from liblp.lpop import lpop

from liblp.numpy.fp4e2m1 import *
from liblp.numpy.lns8i4f3 import *
from liblp.numpy.fp8e5m2 import *
from liblp.numpy.fp8e4m3 import *
from liblp.triton.fp8e4m3 import *
from liblp.triton.fp8e5m2 import *

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class lptensor(object):
    def __init__(self, data: np.ndarray | torch.Tensor, dtype: lpdtype):
        self.raw_data = data
        self.backend = self.__infer_backend(data)
        self.enc_data, self.enc_scale = lpregistry.dispatch(
            lpop.encode,
            dtype,
            self.backend,
            data,
        )
        self.dtype = dtype()    

    
    @classmethod
    def __from_encoded(
        cls,
        enc_data,
        enc_scale,
        dtype: lpdtype,
        backend: lpbackend,
    ):
        obj = cls.__new__(cls)

        obj.raw_data = None
        obj.enc_data = enc_data
        obj.enc_scale = enc_scale
        obj.dtype = dtype()
        obj.backend = backend

        return obj

    def __mul__(self, other):
        res_data, res_scale = lpregistry.dispatch(
            lpop.mul, 
            self.dtype.__class__, 
            self.backend, 
            self.enc_data, 
            other.enc_data,
        )
        return lptensor.__from_encoded(
            res_data,
            res_scale,
            self.dtype.__class__,
            self.backend,
        )   
     
    def __add__(self, other):
        res_data, res_scale = lpregistry.dispatch(
            lpop.add, 
            self.dtype.__class__, 
            self.backend, 
            self.enc_data, 
            other.enc_data,
        )
        return lptensor.__from_encoded(
            res_data,
            res_scale,
            self.dtype.__class__,
            self.backend,
        )

    def __truediv__(self, other):
        res_data, res_scale = lpregistry.dispatch(
            lpop.div, 
            self.dtype.__class__, 
            self.backend, 
            self.enc_data, 
            other.enc_data,
        )
        return lptensor.__from_encoded(
            res_data,
            res_scale,
            self.dtype.__class__,
            self.backend,
        )

    def __sub__(self, other):
        res_data, res_scale = lpregistry.dispatch(
            lpop.sub, 
            self.dtype.__class__, 
            self.backend, 
            self.enc_data, 
            other.enc_data,
        )
        return lptensor.__from_encoded(
            res_data,
            res_scale,
            self.dtype.__class__,
            self.backend,
        )

    def get(self):
        return lpregistry.dispatch(lpop.decode, type(self.dtype), self.backend, self.enc_data)

    def __str__(self):
        return f"{self.dtype}({self.enc_data})"

    def __infer_backend(self, data):
        if isinstance(data, np.ndarray):
            return lpbackend.np
        elif isinstance(data, torch.Tensor):
            if torch.cuda.is_available():
                return lpbackend.triton
            else:
                return lpbackend.torch
        else:
            logger.error(f"cannot infer valid backend for {data}")