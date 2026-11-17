
import torch
import torch.nn.functional as F
import triton

from liblp.lptensor import lptensor
from liblp.lpdtype import mxi4fp32

shapes = [(1, 4096, 4096), (1, 8192, 8192), (1, 16384, 8192), (1, 16384, 16384)]

print("Shape (M, N, K)    |  FP32 (us) |  FP16 (us) |  MXI4 (us) |   GB/s | vs FP32 | vs FP16 |  Err vs decoded")
for m, n, k in shapes:
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    x16, w16 = x.half(), w.half()

    x_lp = lptensor(x, mxi4fp32)
    w_lp = lptensor(w, mxi4fp32)

    t_fp32 = triton.testing.do_bench(lambda: F.linear(x, w)) * 1e3
    t_fp16 = triton.testing.do_bench(lambda: F.linear(x16, w16)) * 1e3
    t_mxi4 = triton.testing.do_bench(lambda: x_lp @ w_lp) * 1e3

    ref = x_lp.get().reshape(m, k) @ w_lp.get().reshape(n, k).T
    out = x_lp @ w_lp
    err = ((out - ref).norm() / ref.norm()).item()

    bytes_moved = w_lp.enc_data.numel() + w_lp.enc_scale.numel() * 4
    gbps = bytes_moved / (t_mxi4 * 1e-6) / 1e9

    print(f"{m}x{n}x{k:<10} | {t_fp32:10.1f} | {t_fp16:10.1f} | {t_mxi4:10.1f} | {gbps:6.1f} | {t_fp32 / t_mxi4:6.2f}x | {t_fp16 / t_mxi4:6.2f}x | {err:.2e}")