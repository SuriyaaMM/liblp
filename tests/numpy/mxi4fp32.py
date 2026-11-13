import torch
import numpy as np
from liblp.testing.utils import __test_normal_distribution

from liblp.lpdtype import mxi4fp32
from liblp.lptensor import lptensor

if __name__ == "__main__":
    a_np = np.random.normal(size=(32*1,)).astype(np.float32)

    a = lptensor(data=a_np, dtype=mxi4fp32)

    a_decoded = a.get()

    print(f"numpy array shape := {a_np.shape}\n")
    print(f"enc array shape := {a.enc_data.shape}")
    print(f"enc scale shape := {a.enc_scale.shape}")
    print(f"dec array shape := {a_decoded.shape}")

    print(f"original array := {a_np}")
    print(f"decoded array := {a_decoded}")
