import torch
import numpy as np
from liblp.testing.utils import __test_normal_distribution

from liblp.lpdtype import fp8e4m3

if __name__ == "__main__":
    __test_normal_distribution(
        dtype=np.float16,
        lpdtype=fp8e4m3,
        device="cpu",
        operation_ref=lambda a, b: a + b,
        operation_lp=lambda a, b: a + b,
        savefile_base="fp8e4m3_np_add",
    )
    __test_normal_distribution(
        dtype=np.float16,
        lpdtype=fp8e4m3,
        device="cpu",
        operation_ref=lambda a, b: a - b,
        operation_lp=lambda a, b: a - b,
        savefile_base="fp8e4m3_np_sub",
    )
    __test_normal_distribution(
        dtype=np.float16,
        lpdtype=fp8e4m3,
        device="cpu",
        operation_ref=lambda a, b: a * b,
        operation_lp=lambda a, b: a * b,
        savefile_base="fp8e4m3_np_mul",
    )
    __test_normal_distribution(
        dtype=np.float16,
        lpdtype=fp8e4m3,
        device="cpu",
        operation_ref=lambda a, b: a / b,
        operation_lp=lambda a, b: a / b,
        savefile_base="fp8e4m3_np_div",
    )