import argparse
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from liblp.triton.mxi4fp32 import encode
from liblp.models.tinyllama_mxi4fp32 import *
from liblp.models.utils import encode_weight

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_wikitext2_tokens(tokenizer, max_tokens: int = 0):
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tokens > 0:
        ids = ids[:max_tokens]
    return ids.to("cuda")


@torch.no_grad()
def ppl_hf(model, ids, seq_len=2048, desc="hf"):
    model.eval()
    nll, count = torch.zeros((), device="cuda"), 0
    for i in tqdm(range(0, len(ids) - 1, seq_len), desc=desc):
        chunk = ids[i : i + seq_len]
        if len(chunk) < 2:
            break
        logits = model(chunk[None]).logits[0].float()
        nll += F.cross_entropy(logits[:-1], chunk[1:], reduction="sum")
        count += len(chunk) - 1
    return torch.exp(nll / count).item()


@torch.no_grad()
def ppl_kernel(model: TinyLLamaMXI4FP32, ids, seq_len=2048, desc="kernel"):
    nll, count = torch.zeros((), device="cuda"), 0
    for i in tqdm(range(0, len(ids) - 1, seq_len), desc=desc):
        chunk = ids[i : i + seq_len].tolist()
        if len(chunk) < 2:
            break
        model.reset()
        for pos in range(len(chunk) - 1):
            logits = model.forward(chunk[pos], pos)
            nll += F.cross_entropy(
                logits[None], torch.tensor([chunk[pos + 1]], device="cuda"), reduction="sum"
            )
        count += len(chunk) - 1
    return torch.exp(nll / count).item()


def is_proj(name: str) -> bool:
    return any(name.endswith(s) for s in PROJ_SUFFIXES)


def layer_idx(name: str):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


@torch.no_grad()
def backup_proj(model):
    return {n: p.data.clone() for n, p in model.named_parameters() if is_proj(n)}


@torch.no_grad()
def restore_proj(model, backup):
    for n, p in model.named_parameters():
        if n in backup:
            p.data.copy_(backup[n])


@torch.no_grad()
def fake_quantize_(model, keep_fp16):
    for n, p in model.named_parameters():
        if is_proj(n) and not keep_fp16(n):
            u8, sc = encode_weight(p.data, mxi4fp32.scale_block_size, encode)
            dq = torch_mxi4fp32.decode(u8.reshape(-1, mxi4fp32.scale_block_size // 2), sc.reshape(-1, 1))
            p.data.copy_(dq.reshape(p.shape).to(p.dtype))


def ablation_groups(num_layers: int):
    has = lambda *subs: (lambda k: any(s in k for s in subs))
    return {
        "all int4": lambda k: False,
        "keep down_proj fp16": has("down_proj"),
        "keep gate/up fp16": has("gate_proj", "up_proj"),
        "keep MLP fp16 (attn-only int4)": has("mlp"),
        "keep q/k fp16": has("q_proj", "k_proj"),
        "keep v/o fp16": has("v_proj", "o_proj"),
        "keep attn fp16 (MLP-only int4)": has("self_attn"),
        "keep layers 0-1 fp16": lambda k: layer_idx(k) in (0, 1),
        f"keep layers {num_layers-2}-{num_layers-1} fp16": lambda k: layer_idx(k)
        in (num_layers - 2, num_layers - 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tokens", type=int, default=0, help="0 = full wikitext-2 test")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--kernel_tokens", type=int, default=4096, help="0 = skip kernel check")
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = get_wikitext2_tokens(tok, args.max_tokens)
    print(f"evaluating on {len(ids)} tokens, seq_len={args.seq_len}")

    hf = LlamaForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
    backup = backup_proj(hf)
    nl = hf.config.num_hidden_layers

    rows = [("fp16 baseline", ppl_hf(hf, ids, args.seq_len, "fp16"))]
    groups = ablation_groups(nl)
    if not args.ablate:
        groups = {"all int4": groups["all int4"]}
    for name, keep in groups.items():
        restore_proj(hf, backup)
        fake_quantize_(hf, keep)
        rows.append((f"W4A16 | {name}", ppl_hf(hf, ids, args.seq_len, name)))
    restore_proj(hf, backup)

    base = rows[0][1]
    print("\n" + "=" * 64)
    for name, p in rows:
        print(f"{name:<44s} ppl={p:8.3f}  (+{p - base:6.3f})")
    print("=" * 64)

    if args.kernel_tokens > 0:
        sub = ids[: args.kernel_tokens]
        fp16_sub = ppl_hf(hf, sub, args.seq_len, "fp16 subset")
        fake_quantize_(hf, lambda k: False)
        fq_sub = ppl_hf(hf, sub, args.seq_len, "fakequant subset")
        restore_proj(hf, backup)

        packed = pack_llama_weights_mxi4(MODEL)
        model = TinyLLamaMXI4FP32(TinyLLamaMXI4FP32Config.from_hf(hf.config), packed)
        k_sub = ppl_kernel(model, sub, args.seq_len)
        print(f"subset of {len(sub)} tokens:")
        print(f"fp16 HF ppl={fp16_sub:.3f}")
        print(f"fakequant (HF) ppl={fq_sub:.3f}")
        print(f"Triton kernel ppl={k_sub:.3f}")


if __name__ == "__main__":
    main()