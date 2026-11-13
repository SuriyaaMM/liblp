import enum

class lpbackend(enum.Enum):
    np = "numpy"
    torch = "torch"
    triton = "triton"
    jax = "jax"