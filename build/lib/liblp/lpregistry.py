from liblp.lpdtype import lpdtype
from liblp.lpbackend import lpbackend
from liblp.lpop import lpop

class lpregistry(object):

    __kernels = {}

    @classmethod
    def register(
        cls,
        op: lpop,
        dtype: lpdtype,
        backend: lpbackend,
    ):
        def decorator(function_pointer):
            cls.__kernels[(op, dtype, backend)] = function_pointer
            return function_pointer
        return decorator
    
    @classmethod
    def dispatch(
        cls,
        op: lpop,
        dtype: lpdtype,
        backend: lpbackend,
        *args,
    ):
        key = (op, dtype, backend)
        if key not in cls.__kernels:
            raise NotImplementedError(f"{op} for {dtype} is not implemented for {backend}")
        
        return cls.__kernels[key](*args)