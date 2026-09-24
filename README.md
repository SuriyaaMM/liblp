# LIBLP (Lib Lower Precision)

## Overview
Lower Precision Arithmetic Library for performing general purpose computations in precision which is generally not available in popular libraries or allow only limited operations (i.e matrix multiplication).

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
│   ├── fp4_e2m1.py 
│   └── lns8_i4f3.py
│
└── triton/ # triton backend
    └── fp8_e4m3.py
```

## Installation

### Pre-Requisites
- `uv` for installing packages
- `git` for cloning the repository

```bash
# clone the repository
# change the directory into liblp
uv sync
```