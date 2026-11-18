
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from liblp.lpdtype import mxi4fp32
from liblp.torch import mxi4fp32 as torch_mxi4fp32
from liblp.models.utils import encode_weight
from liblp.triton.mxi4fp32 import *


PROJ_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def pack_llama_weights_mxi4(
    hf_model: str, savefile: Optional[str] = None, device="cuda"
) -> Dict[str, torch.Tensor]:
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(hf_model, dtype=torch.float16)
    packed: Dict[str, torch.Tensor] = {}
    for key, t in model.state_dict().items():
        if any(key.endswith(s) for s in PROJ_SUFFIXES):
            u8, sc = encode_weight(t.to(device), mxi4fp32.scale_block_size, encode)
            packed[key] = u8
            packed[key + "_scales"] = sc
            packed[key + "_shape"] = torch.tensor(tuple(t.shape))
        else:  # embeddings, lm_head, norms, rotary buffers: keep fp16
            packed[key] = t.to(torch.float16).to(device)
    if savefile:
        torch.save(packed, savefile)
    return packed


@dataclass
class TinyLLamaMXI4FP32Config:
    hidden_size: int = 2048
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    intermediate_size: int = 5632
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_len: int = 2048

    @staticmethod
    def from_hf(c) -> "TinyLLamaMXI4FP32Config":
        theta = getattr(c, "rope_theta", None) or (
            getattr(c, "rope_parameters", None) or {}
        ).get("rope_theta", 10000.0)
        return TinyLLamaMXI4FP32Config(
            hidden_size=c.hidden_size,
            num_hidden_layers=c.num_hidden_layers,
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            intermediate_size=c.intermediate_size,
            rms_norm_eps=c.rms_norm_eps,
            rope_theta=float(theta),
            max_len=min(c.max_position_embeddings, 2048),
        )


class TinyLLamaMXI4FP32:
    def __init__(
        self, 
        cfg: TinyLLamaMXI4FP32Config, 
        packed: Dict[str, torch.Tensor], # packed weights
        device: torch.device = torch.device("cuda"),
    ):
        self.cfg, self.device = cfg, torch.device(device)
        self.w = {k: v.to(self.device) for k, v in packed.items()}
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.groups = cfg.num_attention_heads // cfg.num_key_value_heads

        inv = 1.0 / (
            cfg.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
                / self.head_dim
            )
        )
        pos = torch.arange(cfg.max_len, dtype=torch.float32, device=self.device)
        emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
        self.cos, self.sin = emb.cos(), emb.sin()

        shape = (cfg.max_len, cfg.num_key_value_heads, self.head_dim)
        self.k_cache = [
            torch.zeros(shape, dtype=torch.float16, device=self.device)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.v_cache = [torch.zeros_like(c) for c in self.k_cache]

    def reset(self):
        for c in self.k_cache + self.v_cache:
            c.zero_()

    def _proj(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return mv_w4a16(self.w[key], self.w[key + "_scales"], x)

    def _rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean() + self.cfg.rms_norm_eps) * w.float()

    def _rope(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        half = self.head_dim // 2
        rot = torch.cat([-x[:, half:], x[:, :half]], dim=-1)
        return x * self.cos[pos] + rot * self.sin[pos]

    def _attention(self, h: torch.Tensor, layer: int, pos: int) -> torch.Tensor:
        c, n = self.cfg, f"model.layers.{layer}.self_attn"
        q = self._proj(f"{n}.q_proj.weight", h).view(c.num_attention_heads, self.head_dim)
        k = self._proj(f"{n}.k_proj.weight", h).view(c.num_key_value_heads, self.head_dim)
        v = self._proj(f"{n}.v_proj.weight", h).view(c.num_key_value_heads, self.head_dim)

        q, k = self._rope(q, pos), self._rope(k, pos)
        self.k_cache[layer][pos] = k.to(torch.float16)
        self.v_cache[layer][pos] = v.to(torch.float16)

        K = self.k_cache[layer][: pos + 1].repeat_interleave(self.groups, dim=1).permute(1, 0, 2)
        V = self.v_cache[layer][: pos + 1].repeat_interleave(self.groups, dim=1).permute(1, 0, 2)
        out = F.scaled_dot_product_attention(q.to(torch.float16).unsqueeze(1), K, V)
        out = out.reshape(-1).float()
        return self._proj(f"{n}.o_proj.weight", out)

    def _mlp(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        n = f"model.layers.{layer}.mlp"
        gate = self._proj(f"{n}.gate_proj.weight", h)
        up = self._proj(f"{n}.up_proj.weight", h)
        return self._proj(f"{n}.down_proj.weight", F.silu(gate) * up)

    @torch.no_grad()
    def forward(self, token_id: int, pos: int) -> torch.Tensor:
        x = self.w["model.embed_tokens.weight"][token_id].float()
        for i in range(self.cfg.num_hidden_layers):
            p = f"model.layers.{i}"
            x = x + self._attention(
                self._rmsnorm(x, self.w[f"{p}.input_layernorm.weight"]), i, pos
            )
            x = x + self._mlp(
                self._rmsnorm(x, self.w[f"{p}.post_attention_layernorm.weight"]), i
            )
        x = self._rmsnorm(x, self.w["model.norm.weight"])
        return F.linear(x.to(torch.float16), self.w["lm_head.weight"]).float()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 2,
    ) -> List[int]:
        self.reset()
        tokens = list(prompt_ids)
        for pos, t in enumerate(tokens):
            logits = self.forward(t, pos)
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                idx = torch.tensor(sorted(set(tokens[-64:])), device=logits.device)
                l = logits[idx]
                logits[idx] = torch.where(l > 0, l / repetition_penalty, l * repetition_penalty)
            if temperature == 0.0:
                nxt = int(logits.argmax())
            else:
                nxt = int(torch.multinomial(F.softmax(logits / temperature, -1), 1))
            tokens.append(nxt)
            if nxt == eos_token_id or len(tokens) >= self.cfg.max_len:
                break
            logits = self.forward(nxt, len(tokens) - 1)
        return tokens


# @torch.no_grad()
# def compare_with_hf(hf_name: str, prompt_ids: List[int], packed=None):
#     """Teacher-forced logit comparison against the fp16 HF model."""
#     from transformers import LlamaForCausalLM

#     hf = LlamaForCausalLM.from_pretrained(hf_name, dtype=torch.float16).cuda().eval()
#     ref = hf(torch.tensor([prompt_ids], device="cuda")).logits[0].float()
#     packed = packed or pack_llama_weights_mxi4(hf_name)
#     m = MXI4TinyLlama(MXI4Config.from_hf(hf.config), packed)
#     m.reset()
#     mine = torch.stack([m.forward(t, i) for i, t in enumerate(prompt_ids)])
#     cos = F.cosine_similarity(mine, ref, dim=-1)
#     top1 = (mine.argmax(-1) == ref.argmax(-1)).float().mean()
#     print(f"[model] cosine(logits) mean={cos.mean():.4f} min={cos.min():.4f} | top-1 agreement={top1:.2%}")
#     return mine, ref


# if __name__ == "__main__":
#     check_gemv(2048, 2048)  # attention-shaped
#     check_gemv(5632, 2048)  # gate/up
#     check_gemv(2048, 5632)  # down (K not a multiple of 64 tiles -> exercises masks)
#     check_gemv(256, 2048)  # k/v (GQA)

#     from transformers import AutoTokenizer

#     name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     tok = AutoTokenizer.from_pretrained(name)
#     ids = tok("The capital of France is Paris. The capital of Japan is", return_tensors="pt").input_ids[0].tolist()
#     compare_with_hf(name, ids)