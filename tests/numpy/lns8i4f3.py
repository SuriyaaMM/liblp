import torch
import numpy as np
from liblp.testing.utils import __test_normal_distribution

from liblp.lpdtype import lns8i4f3

if __name__ == "__main__":
    __test_normal_distribution(
        dtype=np.float16,
        lpdtype=lns8i4f3,
        device="cpu",
        operation_ref=lambda a, b: a + b,
        operation_lp=lambda a, b: a + b,
        savefile_base="lns8i4f3_add",
    )
    __test_normal_distribution(
        dtype=np.float16,
        lpdtype=lns8i4f3,
        device="cpu",
        operation_ref=lambda a, b: a * b,
        operation_lp=lambda a, b: a * b,
        savefile_base="lns8i4f3_mul",
    )
    __test_normal_distribution(
        dtype=np.float16,
        lpdtype=lns8i4f3,
        device="cpu",
        operation_ref=lambda a, b: a / b,
        operation_lp=lambda a, b: a / b,
        savefile_base="lns8i4f3_div",
    )