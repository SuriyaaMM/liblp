import torch

from liblp.lptensor import lptensor
from liblp.lpdtype import *

def main():
    at = torch.randn(size=(32 * 256,), dtype=torch.float32)
    a = lptensor(at, mxi8fp32)
    ad = a.get().reshape(at.shape)

    print((at - ad).abs().mean())
    print((at - ad).amax())
    print((at - ad).median())
    

if __name__ == "__main__":
    main()