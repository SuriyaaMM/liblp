import torch

from liblp.lptensor import lptensor
from liblp.lpdtype import *

def main():
    device = torch.device("cuda")
    a = torch.randn(size=(256, 256), dtype=torch.float16).to(device) + 2.00
    b = torch.randn(size=(128, 256), dtype=torch.float16).to(device) + 2.00

    ae, ascales = encode(a)
    be, bscales = encode(b)

    ae = torch.reshape(ae, a.shape)
    be = torch.reshape(be, b.shape)

    print(f"ae(shape=({ae.shape}))")
    print(f"be(shape=({be.shape}))")

    c_torch = a @ b.T
    c_ref = mm_a16(ae, ascales, be, bscales)
    # c_ref = torch_mxi8fp32.decode(ae, ascales).reshape(a.shape) @ torch_mxi8fp32.decode(be, bscales).reshape(b.shape).T
    torch.testing.assert_close(c_ref, c_torch, rtol=1e-2, atol=1.00)

    

if __name__ == "__main__":
    main()