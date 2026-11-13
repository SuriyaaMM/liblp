import torch
import triton
import triton.language as tl

from liblp.lpregistry import lpregistry
from liblp.lpop import lpop
from liblp.lpdtype import fp8e5m2
from liblp.lpbackend import lpbackend

# ------------------------------------------------------------
# kernels
# ------------------------------------------------------------


@triton.jit
def __to(
    fp16_ptr: tl.tensor,
    u8_ptr: tl.tensor,
    n: int,
    block_size: tl.constexpr,
):
    """
    triton kernel for converting from FP16 dtype to fp8e5m2dtype

    Parameters
    ----------
    fp16_ptr: tl.tensor
        valid triton compatible gpu pointer to fp16 data to convert (input pointer)
    u8_ptr: tl.tensor
        valid triton compatible gpu pointer to u8 data (output pointer)
    n: int
        number of elements to convert
    block_size: tl.constexpr
        number of elements per block to handle

    Algorithm
    ---------
    - load the data using the fp16 pointer
    - lossy cast it to fp8_e5m2 using triton's caster
    - bitcast it into u8 using triton's caster
    - store the u8 data using the u8 pointer
    """
    pid = tl.program_id(axis=0)
    jump = block_size * pid + tl.arange(start=0, end=block_size)
    mask = jump < n

    fp16_val = tl.load(
        pointer=fp16_ptr + jump,
        mask=mask,
        other=0.0,
    )
    fp8_e5m2 = tl.cast(
        input=fp16_val,
        dtype=tl.float8e5,
    )
    fp8_e5m2_u8 = tl.cast(
        input=fp8_e5m2,
        dtype=tl.uint8,
        bitcast=True,
    )
    tl.store(
        pointer=u8_ptr + jump,
        value=fp8_e5m2_u8,
        mask=mask,
    )


@triton.jit
def __tor(
    fp16_ptr: tl.tensor,
    u8_ptr: tl.tensor,
    n: int,
    block_size: tl.constexpr,
):
    """
    triton kernel for converting from fp8e5m2 dtype to FP16 dtype

    Parameters
    ----------
    fp16_ptr: tl.tensor
        valid triton compatible gpu pointer to fp16 data to convert (output pointer)
    u8_ptr: tl.tensor
        valid triton compatible gpu pointer to u8 data (input pointer)
    n: int
        number of elements to convert
    block_size: tl.constexpr
        number of elements per block to handle

    Algorithm
    ---------
    - load the data using the u8 pointer
    - bitcast it to fp8_e5m2 using triton's caster
    - lossy cast it into fp16 using triton's caster
    - store the fp16 data using the fp16 pointer
    """
    pid = tl.program_id(axis=0)
    jump = block_size * pid + tl.arange(start=0, end=block_size)
    mask = jump < n

    u8_val = tl.load(
        pointer=u8_ptr + jump,
        mask=mask,
        other=0,
    )
    fp8_e5m2 = tl.cast(
        input=u8_val,
        dtype=tl.float8e5,
        bitcast=True,
    )
    fp8_e5m2_fp16 = tl.cast(
        input=fp8_e5m2,
        dtype=tl.float16,
    )
    tl.store(
        pointer=fp16_ptr + jump,
        value=fp8_e5m2_fp16,
        mask=mask,
    )


@triton.jit
def __add(
    a_u8_ptr: tl.tensor,
    b_u8_ptr: tl.tensor,
    o_u8_ptr: tl.tensor,
    n: int,
    block_size: tl.constexpr,
):
    """
    triton kernel for adding two fp8e5m2(lpa) datatypes

    Parameters
    ----------
    a_u8_ptr: tl.tensor
        valid triton compatible gpu pointer to u8 data, must
        be converting using to function provided by the library (input pointer)
    b_u8_ptr: tl.tensor
        valid triton compatible gpu pointer to u8 data, must
        be converting using to function provided by the library (input pointer)
    o_u8_ptr: tl.tensor
        valid triton compatible gpu pointer to u8 data (output pointer)
    n: int
        number of elements to convert
    block_size: tl.constexpr
        number of elements per block to handle

    Algorithm
    ---------
    - load the data using the u8 pointer
    - bitcast it to fp8_e5m2 using triton's caster
    - lossy cast it into fp16 using triton's caster
    - add the results
    - lossy cast it into fp8_e5m2 using triton's caster
    - bitcast it into u8 using triton's caster
    """
    pid = tl.program_id(axis=0)
    jump = block_size * pid + tl.arange(start=0, end=block_size)
    mask = jump < n

    a_u8 = tl.load(
        pointer=a_u8_ptr + jump,
        mask=mask,
        other=0,
    )
    b_u8 = tl.load(
        pointer=b_u8_ptr + jump,
        mask=mask,
        other=0,
    )
    a_fp8 = tl.cast(
        input=a_u8,
        dtype=tl.float8e5,
        bitcast=True,
    )
    b_fp8 = tl.cast(
        input=b_u8,
        dtype=tl.float8e5,
        bitcast=True,
    )
    a_fp16 = tl.cast(
        input=a_fp8,
        dtype=tl.float16,
    )
    b_fp16 = tl.cast(
        input=b_fp8,
        dtype=tl.float16,
    )

    o_fp16 = a_fp16 + b_fp16

    o_fp8 = tl.cast(
        input=o_fp16,
        dtype=tl.float8e5,
    )
    o_u8 = tl.cast(
        input=o_fp8,
        dtype=tl.uint8,
        bitcast=True,
    )

    tl.store(
        pointer=o_u8_ptr + jump,
        value=o_u8,
        mask=mask,
    )

@triton.jit
def __sub(
    a_u8_ptr: tl.tensor,
    b_u8_ptr: tl.tensor,
    o_u8_ptr: tl.tensor,
    n: int,
    block_size: tl.constexpr,
):
    """
    triton kernel for adding two fp8e5m2(lpa) datatypes

    Parameters
    ----------
    a_u8_ptr: tl.tensor
        valid triton compatible gpu pointer to u8 data, must
        be converting using to function provided by the library (input pointer)
    b_u8_ptr: tl.tensor
        valid triton compatible gpu pointer to u8 data, must
        be converting using to function provided by the library (input pointer)
    o_u8_ptr: tl.tensor
        valid triton compatible gpu pointer to u8 data (output pointer)
    n: int
        number of elements to convert
    block_size: tl.constexpr
        number of elements per block to handle

    Algorithm
    ---------
    - load the data using the u8 pointer
    - bitcast it to fp8_e5m2 using triton's caster
    - lossy cast it into fp16 using triton's caster
    - subtract the results
    - lossy cast it into fp8_e5m2 using triton's caster
    - bitcast it into u8 using triton's caster
    """
    pid = tl.program_id(axis=0)
    jump = block_size * pid + tl.arange(start=0, end=block_size)
    mask = jump < n

    a_u8 = tl.load(
        pointer=a_u8_ptr + jump,
        mask=mask,
        other=0,
    )
    b_u8 = tl.load(
        pointer=b_u8_ptr + jump,
        mask=mask,
        other=0,
    )
    a_fp8 = tl.cast(
        input=a_u8,
        dtype=tl.float8e5,
        bitcast=True,
    )
    b_fp8 = tl.cast(
        input=b_u8,
        dtype=tl.float8e5,
        bitcast=True,
    )
    a_fp16 = tl.cast(
        input=a_fp8,
        dtype=tl.float16,
    )
    b_fp16 = tl.cast(
        input=b_fp8,
        dtype=tl.float16,
    )

    o_fp16 = a_fp16 - b_fp16

    o_fp8 = tl.cast(
        input=o_fp16,
        dtype=tl.float8e5,
    )
    o_u8 = tl.cast(
        input=o_fp8,
        dtype=tl.uint8,
        bitcast=True,
    )

    tl.store(
        pointer=o_u8_ptr + jump,
        value=o_u8,
        mask=mask,
    )


@triton.jit
def __mul(
    a_u8_ptr: tl.tensor,
    b_u8_ptr: tl.tensor,
    o_u8_ptr: tl.tensor,
    n: int,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    jump = block_size * pid + tl.arange(start=0, end=block_size)
    mask = jump < n

    a_u8 = tl.load(
        pointer=a_u8_ptr + jump,
        mask=mask,
        other=0,
    )
    b_u8 = tl.load(
        pointer=b_u8_ptr + jump,
        mask=mask,
        other=0,
    )
    a_fp8 = tl.cast(
        input=a_u8,
        dtype=tl.float8e5,
        bitcast=True,
    )
    b_fp8 = tl.cast(
        input=b_u8,
        dtype=tl.float8e5,
        bitcast=True,
    )
    a_fp16 = tl.cast(
        input=a_fp8,
        dtype=tl.float16,
    )
    b_fp16 = tl.cast(
        input=b_fp8,
        dtype=tl.float16,
    )

    o_fp16 = a_fp16 * b_fp16

    o_fp8 = tl.cast(
        input=o_fp16,
        dtype=tl.float8e5,
    )
    o_u8 = tl.cast(
        input=o_fp8,
        dtype=tl.uint8,
        bitcast=True,
    )

    tl.store(
        pointer=o_u8_ptr + jump,
        value=o_u8,
        mask=mask,
    )


@triton.jit
def __div(
    a_u8_ptr: tl.tensor,
    b_u8_ptr: tl.tensor,
    o_u8_ptr: tl.tensor,
    n: int,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    jump = block_size * pid + tl.arange(start=0, end=block_size)
    mask = jump < n

    a_u8 = tl.load(
        pointer=a_u8_ptr + jump,
        mask=mask,
        other=0,
    )
    b_u8 = tl.load(
        pointer=b_u8_ptr + jump,
        mask=mask,
        other=0,
    )
    a_fp8 = tl.cast(
        input=a_u8,
        dtype=tl.float8e5,
        bitcast=True,
    )
    b_fp8 = tl.cast(
        input=b_u8,
        dtype=tl.float8e5,
        bitcast=True,
    )
    a_fp16 = tl.cast(
        input=a_fp8,
        dtype=tl.float16,
    )
    b_fp16 = tl.cast(
        input=b_fp8,
        dtype=tl.float16,
    )

    o_fp16 = a_fp16 / b_fp16

    o_fp8 = tl.cast(
        input=o_fp16,
        dtype=tl.float8e5,
    )
    o_u8 = tl.cast(
        input=o_fp8,
        dtype=tl.uint8,
        bitcast=True,
    )

    tl.store(
        pointer=o_u8_ptr + jump,
        value=o_u8,
        mask=mask,
    )


# ------------------------------------------------------------
# utility functions
# ------------------------------------------------------------

@lpregistry.register(
    lpop.encode,
    fp8e5m2,
    lpbackend.triton,
)
def encode(
    x: torch.Tensor,
    block_size: int = 256,
) -> torch.Tensor:
    x = x.contiguous()
    n = x.numel()

    u8_out = torch.empty(
        size=x.shape,
        dtype=torch.uint8,
        device=x.device,
        memory_format=torch.contiguous_format,
    )

    launch_parameters = (triton.cdiv(n, block_size),)

    __to[launch_parameters](
        fp16_ptr=x,
        u8_ptr=u8_out,
        n=n,
        block_size=block_size,
    )

    return u8_out, None

@lpregistry.register(
    lpop.decode,
    fp8e5m2,
    lpbackend.triton,
)
def decode(
    x: torch.Tensor,
    block_size: int = 256,
) -> torch.Tensor:
    x = x.contiguous()
    n = x.numel()

    fp16_out = torch.empty(
        size=x.shape,
        dtype=torch.float16,
        device=x.device,
        memory_format=torch.contiguous_format,
    )

    launch_parameters = (triton.cdiv(n, block_size),)

    __tor[launch_parameters](
        fp16_ptr=fp16_out,
        u8_ptr=x,
        n=n,
        block_size=block_size,
    )

    return fp16_out

@lpregistry.register(
    lpop.add,
    fp8e5m2,
    lpbackend.triton,
)
def add(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
):
    if a.shape != b.shape:
        raise ValueError(
            f"a(shape={a.shape}) and b(shape={b.shape}) must be of the same size"
        )
    # TODO: remove this later when added support for cpu
    if a.device == "cpu":
        raise ValueError(f"a(device={a.device}) is not on a valid triton supported gpu")

    a = a.contiguous()
    b = b.contiguous()
    n = a.numel()

    o_u8 = torch.empty(
        size=a.shape,
        dtype=torch.uint8,
        device=a.device,
        memory_format=torch.contiguous_format,
    )

    launch_parameters = (triton.cdiv(n, block_size),)

    __add[launch_parameters](
        a_u8_ptr=a,
        b_u8_ptr=b,
        o_u8_ptr=o_u8,
        n=n,
        block_size=block_size,
    )

    return o_u8, None

@lpregistry.register(
    lpop.sub,
    fp8e5m2,
    lpbackend.triton,
)
def add(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
):
    if a.shape != b.shape:
        raise ValueError(
            f"a(shape={a.shape}) and b(shape={b.shape}) must be of the same size"
        )
    # TODO: remove this later when added support for cpu
    if a.device == "cpu":
        raise ValueError(f"a(device={a.device}) is not on a valid triton supported gpu")

    a = a.contiguous()
    b = b.contiguous()
    n = a.numel()

    o_u8 = torch.empty(
        size=a.shape,
        dtype=torch.uint8,
        device=a.device,
        memory_format=torch.contiguous_format,
    )

    launch_parameters = (triton.cdiv(n, block_size),)

    __sub[launch_parameters](
        a_u8_ptr=a,
        b_u8_ptr=b,
        o_u8_ptr=o_u8,
        n=n,
        block_size=block_size,
    )

    return o_u8, None

@lpregistry.register(
    lpop.mul,
    fp8e5m2,
    lpbackend.triton,
)
def mul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
):
    if a.shape != b.shape:
        raise ValueError(
            f"a(shape={a.shape}) and b(shape={b.shape}) must be of the same size"
        )
    if a.device == "cpu":
        raise ValueError(f"a(device={a.device}) is not on a valid triton supported gpu")

    a = a.contiguous()
    b = b.contiguous()
    n = a.numel()

    o_u8 = torch.empty(
        size=a.shape,
        dtype=torch.uint8,
        device=a.device,
        memory_format=torch.contiguous_format,
    )

    launch_parameters = (triton.cdiv(n, block_size),)

    __mul[launch_parameters](
        a_u8_ptr=a,
        b_u8_ptr=b,
        o_u8_ptr=o_u8,
        n=n,
        block_size=block_size,
    )

    return o_u8, None

@lpregistry.register(
    lpop.div,
    fp8e5m2,
    lpbackend.triton,
)
def div(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 256,
):
    if a.shape != b.shape:
        raise ValueError(
            f"a(shape={a.shape}) and b(shape={b.shape}) must be of the same size"
        )
    if a.device == "cpu":
        raise ValueError(f"a(device={a.device}) is not on a valid triton supported gpu")

    a = a.contiguous()
    b = b.contiguous()
    n = a.numel()

    o_u8 = torch.empty(
        size=a.shape,
        dtype=torch.uint8,
        device=a.device,
        memory_format=torch.contiguous_format,
    )

    launch_parameters = (triton.cdiv(n, block_size),)

    __div[launch_parameters](
        a_u8_ptr=a,
        b_u8_ptr=b,
        o_u8_ptr=o_u8,
        n=n,
        block_size=block_size,
    )

    return o_u8, None
