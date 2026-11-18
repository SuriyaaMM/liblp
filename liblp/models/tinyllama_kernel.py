from typing import Callable, Dict, Optional
import torch
import torch.nn.functional as F

from liblp.models.utils import encode_weight
from liblp.models.tinyllama_mxi4fp32 import (
    TinyLLamaMXI4FP32, TinyLLamaMXI4FP32Config, PROJ_SUFFIXES,
)


def pack_state_dict(
    sd: Dict[str, torch.Tensor],
    encode: Callable,
    block_size: int,
    keep_fp16: Optional[Callable[[str], bool]] = None,
    device="cuda",
) -> Dict[str, torch.Tensor]:
    packed = {}
    for key, t in sd.items():
        is_proj = any(key.endswith(s) for s in PROJ_SUFFIXES)
        if is_proj and not (keep_fp16 and keep_fp16(key)):
            u8, sc = encode_weight(t.to(device), block_size, encode)
            packed[key], packed[key + "_scales"] = u8, sc
            packed[key + "_shape"] = torch.tensor(tuple(t.shape))
        else:
            packed[key] = t.to(torch.float16).to(device)
    return packed


class TinyLlamaKernel(TinyLLamaMXI4FP32):
    """Works for any format whose kernel has the mv_w4a16(w_u8, scales, x) signature."""

    def __init__(self, cfg, packed, mv, device="cuda"):
        super().__init__(cfg, packed, device)
        self.mv = mv

    def _proj(self, key, x):
        sk = key + "_scales"
        if sk not in self.w:  # kept in fp16 (ablation)
            return F.linear(x.to(torch.float16), self.w[key]).float()
        return self.mv(self.w[key], self.w[sk], x)