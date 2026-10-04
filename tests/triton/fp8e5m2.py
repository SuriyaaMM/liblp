import torch
import numpy as np
from liblp.testing.utils import __test_normal_distribution

from liblp.lpdtype import fp8e5m2

if __name__ == "__main__":
    __test_normal_distribution(
        dtype=torch.float16,
        lpdtype=fp8e5m2,
        device="cuda",
        operation_ref=lambda a, b: a + b,
        operation_lp=lambda a, b: a + b,
        savefile_base="fp8e5m2_add",
    )
    __test_normal_distribution(
        dtype=torch.float16,
        lpdtype=fp8e5m2,
        device="cuda",
        operation_ref=lambda a, b: a - b,
        operation_lp=lambda a, b: a - b,
        savefile_base="fp8e5m2_sub",
    )
    __test_normal_distribution(
        dtype=torch.float16,
        lpdtype=fp8e5m2,
        device="cuda",
        operation_ref=lambda a, b: a * b,
        operation_lp=lambda a, b: a * b,
        savefile_base="fp8e5m2_mul",
    )
    __test_normal_distribution(
        dtype=torch.float16,
        lpdtype=fp8e5m2,
        device="cuda",
        operation_ref=lambda a, b: a / b,
        operation_lp=lambda a, b: a / b,
        savefile_base="fp8e5m2_div",
    )
