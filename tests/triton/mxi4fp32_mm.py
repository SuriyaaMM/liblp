import torch
import triton

from liblp.lptensor import lptensor
from liblp.lpdtype import mxi4fp32
from liblp.triton.mxi4fp32 import matmul as triton_mxi4_matmul

device = torch.device("cuda")

# do_bench takes warmup and rep in milliseconds, not iterations
warmup_ms = 25
rep_ms = 100

shapes = [
    (128, 128, 800),     # MNIST Layer 1
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

header = (
    f"{'Shape (M, N, K)':<18} | "
    f"{'FP32 (ms)':>10} | "
    f"{'FP16 (ms)':>10} | "
    f"{'MXI4 (ms)':>10} | "
    f"{'MXI4 TFLOPS':>11} | "
    f"{'vs FP32':>8} | "
    f"{'vs FP16':>8} | "
    f"{'Rel Err':>8}"
)
print(header)
print("-" * len(header))

for M, N, K in shapes:
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32)

    # Pre-encode outside timed region
    a_lp = lptensor(A, mxi4fp32)
    b_lp = lptensor(B, mxi4fp32)

    B_T = B.T.contiguous()
    A_h = A.half()
    B_h_T = B_T.half()

    tile_y = 64 if (M >= 512 and N >= 512) else 32

    mxi4_fn = lambda: triton_mxi4_matmul(
        a_lp.enc_data, a_lp.enc_scale, a_lp.shape,
        b_lp.enc_data, b_lp.enc_scale, b_lp.shape,
        tile_y=tile_y,
    )

    # Numerical sanity check (relative L2 error vs FP32)
    ref = A @ B_T
    out = mxi4_fn()  # also triggers compilation before timing
    rel_err = (torch.norm(out - ref) / torch.norm(ref)).item()

    # Pure GEMM timings (ms), do_bench flushes L2 between runs by default
    t_fp32 = triton.testing.do_bench(lambda: A @ B_T, warmup=warmup_ms, rep=rep_ms, return_mode="median")
    t_fp16 = triton.testing.do_bench(lambda: A_h @ B_h_T, warmup=warmup_ms, rep=rep_ms, return_mode="median")
    t_mxi4 = triton.testing.do_bench(mxi4_fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")

    flops = 2.0 * M * N * K
    mxi4_tflops = (flops / (t_mxi4 * 1e-3)) / 1e12

    print(
        f"{f'{M}x{N}x{K}':<18} | "
        f"{t_fp32:>10.4f} | "
        f"{t_fp16:>10.4f} | "
        f"{t_mxi4:>10.4f} | "
        f"{mxi4_tflops:>11.2f} | "
        f"{t_fp32 / t_mxi4:>7.2f}x | "
        f"{t_fp16 / t_mxi4:>7.2f}x | "
        f"{rel_err:>8.4f}"
    )
