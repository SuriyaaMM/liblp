from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import fp8e4m3
from liblp.lpbackend import lpbackend
import ml_dtypes
import numpy as np
import logging

logging.getLogger(__name__)

@lpregistry.register(
    lpop.encode, 
    fp8e4m3,
    lpbackend.np,
)
def encode(x: np.ndarray):
    return x.astype(ml_dtypes.float8_e4m3), None

@lpregistry.register(
    lpop.decode, 
    fp8e4m3,
    lpbackend.np,
)
def decode(x: np.ndarray):
    return x.astype(np.float32)

@lpregistry.register(
    lpop.add, 
    fp8e4m3,
    lpbackend.np,
)
def add(x: np.ndarray, y: np.ndarray):
    return x + y, None

@lpregistry.register(
    lpop.sub, 
    fp8e4m3,
    lpbackend.np,
)
def sub(x: np.ndarray, y: np.ndarray):
    return x - y, None

@lpregistry.register(
    lpop.mul, 
    fp8e4m3,
    lpbackend.np,
)
def mul(x: np.ndarray, y: np.ndarray):
    return x * y, None

@lpregistry.register(
    lpop.div, 
    fp8e4m3,
    lpbackend.np,
)
def div(x: np.ndarray, y: np.ndarray):
    return x / y, None