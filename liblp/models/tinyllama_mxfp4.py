from typing import Dict, Optional
import torch

from liblp.lpdtype import mxfp4
from liblp.models.utils import encode_weight
from liblp.models.tinyllama_mxi4fp32 import (
    TinyLLamaMXI4FP32, TinyLLamaMXI4FP32Config, PROJ_SUFFIXES,
)
from liblp.torch import mxfp4 as torch_mxfp4
from liblp.triton.mxfp4 import mv_w4a16

TinyLLamaMXFP4Config = TinyLLamaMXI4FP32Config


def pack_llama_weights_mxfp4(
    hf_model: str, savefile: Optional[str] = None, device="cuda"
) -> Dict[str, torch.Tensor]:
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(hf_model, dtype=torch.float16)
    packed: Dict[str, torch.Tensor] = {}
    for key, t in model.state_dict().items():
        if any(key.endswith(s) for s in PROJ_SUFFIXES):
            u8, sc = encode_weight(t.to(device), mxfp4.scale_block_size, torch_mxfp4.encode)
            packed[key] = u8
            packed[key + "_scales"] = sc
            packed[key + "_shape"] = torch.tensor(tuple(t.shape))
        else:
            packed[key] = t.to(torch.float16).to(device)
    if savefile:
        torch.save(packed, savefile)
    return packed


class TinyLLamaMXFP4(TinyLLamaMXI4FP32):
    def _proj(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return mv_w4a16(self.w[key], self.w[key + "_scales"], x)