import torch
import numpy as np
from liblp.testing.utils import __test_normal_distribution

from liblp.lpdtype import mxi4fp32
from liblp.lptensor import lptensor

if __name__ == "__main__":
    __test_normal_distribution(
        dtype=torch.float32,
        lpdtype=mxi4fp32,
        device="cuda",
        operation_ref=lambda a, b: a + b,
        operation_lp=lambda a, b: a + b,
        savefile_base="mxi4fp32_add",
    )
    __test_normal_distribution(
        dtype=torch.float32,
        lpdtype=mxi4fp32,
        device="cuda",
        operation_ref=lambda a, b: a * b,
        operation_lp=lambda a, b: a * b,
        savefile_base="mxi4fp32_mul",
    )
    __test_normal_distribution(
        dtype=torch.float32,
        lpdtype=mxi4fp32,
        device="cuda",
        operation_ref=lambda a, b: a / b,
        operation_lp=lambda a, b: a / b,
        savefile_base="mxi4fp32_div",
    )
