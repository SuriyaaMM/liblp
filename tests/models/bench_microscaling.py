import argparse, gc, time
import torch, triton
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from liblp.lpdtype import mxi4fp32
from liblp.models.utils import encode_weight
from liblp.models.tinyllama_mxi4fp32 import (
    TinyLLamaMXI4FP32, TinyLLamaMXI4FP32Config, PROJ_SUFFIXES,
)
from liblp.models.tinyllama_kernel import TinyLlamaKernel, pack_state_dict
from liblp.torch import mxi4fp32 as T_I4, mxfp4 as T_F4
from liblp.triton import mxi4fp32 as K_I4, mxfp4 as K_F4

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SHAPES = {"q/o 2048x2048": (2048, 2048), "k/v 256x2048": (256, 2048),
          "gate/up 5632x2048": (5632, 2048), "down 2048x5632": (2048, 5632)}

FORMATS = {  
    "mxi4":  (K_I4.encode, T_I4.decode, K_I4.mv_w4a16, mxi4fp32.scale_block_size),
    "mxfp4": (T_F4.encode, T_F4.decode, K_F4.mv_w4a16, 32),
}
VARIANTS = ["fp32", "fp16", "mxi4", "mxfp4"]
L2_FLUSH_BYTES = 512 << 20   


def is_proj(k):
    return any(k.endswith(s) for s in PROJ_SUFFIXES)


def peak_bw_gbs():
    a = torch.empty(1 << 29, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    ms = triton.testing.do_bench(lambda: b.copy_(a))
    return 2 * a.numel() / ms / 1e6


def make_graph(fn):
    for _ in range(2):        
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def time_calls(calls):
    """Mean GPU ms per call; calls touch distinct weights so L2 can't help."""
    def run():
        for c in calls:
            c()
    g = make_graph(run)
    return triton.testing.do_bench(g.replay, warmup=25, rep=100) / len(calls)


def n_copies(nbytes):
    return max(1, min(1024, L2_FLUSH_BYTES // nbytes + 1))


class TinyLlamaDense(TinyLLamaMXI4FP32):
    def _proj(self, key, x):
        w = self.w[key]
        return F.linear(x.to(w.dtype), w).float()


@torch.no_grad()
def bench_shapes(peak):
    print(f"{'shape':<20}{'impl':<8}{'us':>8}{'GB/s':>9}{'%peak':>8}{'vs fp16':>9}{'rel err':>10}")
    for name, (n, k) in SHAPES.items():
        torch.manual_seed(0)
        w = (torch.randn(n, k, device="cuda") * 0.02).half()
        x32 = torch.randn(k, device="cuda")
        x16 = x32.half()
        exact = w.float() @ x32
        res = []   # (label, ms, nbytes, out)

        for label, dt, x in (("fp32", torch.float32, x32), ("fp16", torch.float16, x16)):
            wd = w.to(dt)
            nb = wd.numel() * wd.element_size()
            ws = [wd.clone() for _ in range(n_copies(nb))]
            ms = time_calls([(lambda wi=wi: F.linear(x[None], wi)) for wi in ws])
            res.append((label, ms, nb, F.linear(x[None], wd)[0].float()))
            del ws

        for fmt, (enc, dec, mv, bs) in FORMATS.items():
            u8, sc = encode_weight(w, bs, enc)
            wq = dec(u8.reshape(-1, bs // 2), sc.reshape(-1, 1)).reshape(n, k)
            ref = wq @ x32
            out = mv(u8, sc, x32)
            kerr = ((out - ref).norm() / ref.norm()).item()
            assert kerr < 1e-3, f"{fmt} {name}: kernel mismatch (rel err {kerr})"
            nb = u8.numel() * u8.element_size() + sc.numel() * sc.element_size()
            pairs = [(u8.clone(), sc.clone()) for _ in range(n_copies(nb))]
            ms = time_calls([(lambda a=a, b=b: mv(a, b, x32)) for a, b in pairs])
            res.append((fmt, ms, nb, out))
            del pairs

        ms16 = next(r[1] for r in res if r[0] == "fp16")
        for label, ms, nb, out in res:
            gbs = nb / ms / 1e6
            err = ((out - exact).norm() / exact.norm()).item()
            print(f"{name if label == 'fp32' else '':<20}{label:<8}{ms*1e3:8.1f}{gbs:9.0f}"
                  f"{gbs/peak*100:7.0f}%{ms16/ms:9.2f}{err:10.4f}")
        print()


def build(variant, cfg, sd):
    if variant in ("fp32", "fp16"):
        dt = torch.float32 if variant == "fp32" else torch.float16
        packed = {k: v.to("cuda", dt if is_proj(k) else torch.float16) for k, v in sd.items()}
        return TinyLlamaDense(cfg, packed)
    enc, _, mv, bs = FORMATS[variant]
    return TinyLlamaKernel(cfg, pack_state_dict(sd, enc, bs), mv)


def proj_plan(m):
    c, plan = m.cfg, []
    for i in range(c.num_hidden_layers):
        a, f = f"model.layers.{i}.self_attn", f"model.layers.{i}.mlp"
        for key, kin in ((f"{a}.q_proj.weight", c.hidden_size), (f"{a}.k_proj.weight", c.hidden_size),
                         (f"{a}.v_proj.weight", c.hidden_size), (f"{a}.o_proj.weight", c.hidden_size),
                         (f"{f}.gate_proj.weight", c.hidden_size), (f"{f}.up_proj.weight", c.hidden_size),
                         (f"{f}.down_proj.weight", c.intermediate_size)):
            plan.append((key, torch.randn(kin, device="cuda")))
    return plan


def step_bytes(m, plan):
    keys = [k for k, _ in plan] + ["lm_head.weight"]
    return sum(m.w[kk].numel() * m.w[kk].element_size()
               for k in keys for kk in (k, k + "_scales") if kk in m.w)


@torch.no_grad()
def gpu_step_ms(m, plan):
    lm_x = torch.randn(m.cfg.hidden_size, device="cuda").half()
    def step():
        for key, x in plan:
            m._proj(key, x)
        F.linear(lm_x, m.w["lm_head.weight"])
    return triton.testing.do_bench(make_graph(step).replay, warmup=25, rep=100)


@torch.no_grad()
def eager_ms_per_token(m, prompt, n_new):
    m.reset()
    for pos, t in enumerate(prompt):
        logits = m.forward(t, pos)
    tok, pos = int(logits.argmax()), len(prompt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_new):
        logits = m.forward(tok, pos)
        tok = int(logits.argmax())
        pos += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_new * 1e3


def bench_models(args, peak):
    hf = LlamaForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    cfg = TinyLLamaMXI4FP32Config.from_hf(hf.config)
    sd = hf.state_dict()
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, 32000, (args.prompt_len,), generator=g).tolist()

    rows = {}
    for v in args.variants:
        print(f"[{v}] building + benchmarking ...")
        m = build(v, cfg, sd)
        plan = proj_plan(m)
        nb = step_bytes(m, plan)
        gms = gpu_step_ms(m, plan)
        ems = None
        if not args.skip_e2e:
            eager_ms_per_token(m, prompt, 8)  # warmup / autotune
            ems = min(eager_ms_per_token(m, prompt, args.new_tokens) for _ in range(3))
        rows[v] = (nb, gms, ems)
        del m, plan
        gc.collect(); torch.cuda.empty_cache()

    g16 = rows["fp16"][1] if "fp16" in rows else None
    print("\n" + "=" * 112)
    print(f"{'variant':<8}{'weights MB':>12}{'GPU ms/tok':>12}{'GB/s':>8}{'%peak':>8}"
          f"{'vs fp16':>9}{'GPU tok/s':>11}{'eager ms/tok':>14}{'eager tok/s':>13}{'GPU-bound %':>13}")
    for v, (nb, gms, ems) in rows.items():
        gbs = nb / gms / 1e6
        sp = f"{g16/gms:9.2f}" if g16 else f"{'-':>9}"
        e1 = f"{ems:14.2f}{1e3/ems:13.1f}{gms/ems*100:12.0f}%" if ems else f"{'-':>14}{'-':>13}{'-':>13}"
        print(f"{v:<8}{nb/1e6:12.0f}{gms:12.3f}{gbs:8.0f}{gbs/peak*100:7.0f}%{sp}{1e3/gms:11.0f}{e1}")
    print("=" * 112)
    print("GPU ms/tok = CUDA-graph time for 154 projections + lm_head (attention/norm/rope excluded).")
    print("GPU-bound %  = GPU ms / eager ms. Low values mean the Python loop, not the kernels, is the limit.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--prompt_len", type=int, default=32)
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--scale_mode", default="e8m0", choices=["e8m0", "e8m0_ceil", "fp32"])
    ap.add_argument("--skip_e2e", action="store_true")
    ap.add_argument("--skip_shapes", action="store_true")
    args = ap.parse_args()

    peak = peak_bw_gbs()
    print(f"measured copy bandwidth: {peak:.0f} GB/s\n")
    if not args.skip_shapes:
        bench_shapes(peak)
    bench_models(args, peak)


if __name__ == "__main__":
    main()