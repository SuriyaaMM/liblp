# liblp (Lib Lower Precision)

## Overview
Lower Precision Arithmetic Library for performing general purpose computations in precision which is generally not available in popular libraries or allow only limited operations (i.e matrix multiplication).

## Usage

### Converting Existing Tensors/NdArrays & Performing Basic Operations
- We store the original data as `lptensor.raw_data` and the actual encoded data as `lptensor.enc_data`, data is automatically encoded in the constructor itself
```python
from liblp.lptensor import lptensor
from liblp.lpdtype import lns8i4f3

# a is now stored as np.ndarray[np.uint8]
# these raw 8 bits are interpreter as lns8i4f3 format
a = lptensor(data=np.array([1.0, 2.0, 3.0]), dtype=lns8i4f3)
b = lptensor(data=np.array([2.0, 3.0, 4.0]), dtype=lns8i4f3)

# actually prints the raw 8 bit values, makes no sense 
# to end user
print(a + b)

c = a + b
# explicitly invokes the decode function in the registry
# which decodes these raw bits into fp32 values
print(c.get())
```
- Note: Backend is inferred automatically by the function `lptensor.__infer_backend()`, it is a simple `if/else` checker, TLDR if the input is on `device="cuda"` then we simply use the `triton` backend, otherwise if it is a `torch.Tensor` then we use the `torch` backend, otherwise if it is a `np.ndarray` then we use the `numpy` backend.

### Implementing Custom Dispatchers
- `lpregistry.lpregistry` is the heart & brain of the entire library, it stores all the functions for particular data type & operation and invokes based on runtime values provided
- To implement custom data types (or) custom operations it is a matter of adding and file to the existing backends & appending it to the registry
```python
# assume such a thing is defined
from liblp.lpdtype import fp8e7m0

# implement the encode function for fp8e7m0
@lpregistry.register(
    # operation lpop := lp operation
    lpop.encode, 
    # lpdtype
    fp8e7m0,
    # lpbackend
    lpbackend.np,
)
def encode(x: np.ndarray):
    # implement your custom encoder here
    # ...
    # return (encoded_data, scale)
    return encoded_x, None

# in `lptensor.py` add
# this is required for registry to properly 
# register the function in the dictionary
from liblp.numpy.fp8e7m0 import *
```
- One thing to be mindful of is the number of arguments & type of the arguments, there's no type checking or number of argument checking whatsover, dispatcher will simply throw runtime error based on the function being invocated with wrong argument count or wrong datatype

## Testing
### Running Tests
- Running a test is straightforward, simply do `uv run tests/backend/dtype.py`, this will generate all the plots that we're using for analysis purposes.
- Analysis includes Heatmaps, Summary Plots & Data used for generating these plots, simply reading the `dtype_backend_operation_summary.pdf/png` should give a overall idea of how good is the datatype performing in terms of precision & accuracy.

### Creating Tests
- Creating a test is also straightforward, only function to invoke is `__test_normal_distribution` from the `liblp.testing.utils.py`, arguments are whatever they mean literally, you can check the existing test files to infer the meaning if it is difficult to understand.

## Project Structure
```
liblp/
├── __init__.py
├── lpbackend.py # manages backend enumerations
├── lpdtype.py  # manages type definitions
├── lpop.py # manages operation enumerations
├── lpregistry.py # manages registry
├── lptensor.py # manages the tensor class
│
├── numpy/ # numpy backend
└── triton/ # triton backend
```

## Support Matrix

| Format        | CPU   | GPU |
|---------------|:-----:|:---:|
| FP8 E4M3      |✔      |✔    |
| FP8 E5M2      |✔      |✔    |
| FP4 (Limited) |✔      |𝓍    |
| LNS8 I4F3     |✔      |✔    |

## Installation

### Pre-Requisites
- `uv` for installing packages
- `git` for cloning the repository

```bash
# clone the repository
# git clone https://github.com/SuriyaaMM/liblp
# change the directory into liblp
cd liblp
# create virtual environment
uv venv --python 3.13
# installs dependencies
uv sync
```