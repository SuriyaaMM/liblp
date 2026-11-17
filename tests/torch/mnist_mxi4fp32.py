import torch
import torch.nn as nn
import torchvision
from liblp.lptensor import lptensor
from liblp.lpdtype import mxi4fp32

device = torch.device("cuda")

# ---- 1. A standard MNIST MLP ----
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))

base = MNISTNet().to(device)
base.load_state_dict(torch.load("mnist_mlp.pt", map_location=device))
base.eval()

# ---- 2. Zero-pad to satisfy encode's %32 guard (on GPU) ----
W1 = torch.zeros(128, 800, device=device); W1[:, :784] = base.fc1.weight.data
b1 = base.fc1.bias.data.to(device)
W2 = torch.zeros(32, 128, device=device);  W2[:10, :]  = base.fc2.weight.data
b2 = torch.zeros(32, device=device);       b2[:10]     = base.fc2.bias.data

# Pre-encode static weights on GPU once
w1_lp = lptensor(W1, mxi4fp32)
w2_lp = lptensor(W2, mxi4fp32)

# Pre-allocate input buffer so zero-padding allocation isn't measured as compute
xp_buf = torch.zeros(128, 800, device=device)

def forward_fp32(xp):                                               # xp: (bs, 800) on cuda
    h = torch.relu(xp @ W1.T + b1)                                  # (bs, 128)
    return (h @ W2.T + b2)[:, :10]                                  # (bs, 10)

# ---- 3. Triton mxi4fp32 inference ----
def forward_mxi4(xp):
    x_lp = lptensor(xp, mxi4fp32)
    h    = torch.relu((x_lp @ w1_lp) + b1)     # x_lp @ w1_lp is already (bs, 128) fp32
    h_lp = lptensor(h, mxi4fp32)
    out  = (h_lp @ w2_lp) + b2                 # h_lp @ w2_lp is already (bs, 32) fp32
    return out[:, :10]

# ---- 4. Accuracy Check ----
loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    ),
    batch_size=128,
    drop_last=True,
    pin_memory=True,
)

correct_fp = correct_q = total = 0
with torch.no_grad():
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        xp_buf[:, :784] = x.view(-1, 784)

        total += y.size(0)
        correct_fp += (forward_fp32(xp_buf).argmax(1) == y).sum().item()
        correct_q  += (forward_mxi4(xp_buf).argmax(1) == y).sum().item()

print(f"fp32   acc: {correct_fp / total:.4f}")
print(f"mxi4   acc: {correct_q  / total:.4f}")
print("-" * 52)

# ---- 5. GPU Benchmarking Helper ----
def benchmark_fn(fn, *args, warmup: int = 25, iters: int = 200):
    with torch.no_grad():
        # Warmup: triggers Triton JIT compilation & GPU clock boost
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        # Flush L2 cache between iterations for realistic cold-cache timings
        l2_flush = torch.empty(int(16e6), dtype=torch.int8, device=device)

        for i in range(iters):
            l2_flush.zero_()
            start_events[i].record()
            fn(*args)
            end_events[i].record()

        torch.cuda.synchronize()

    times_ms = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    return {
        "mean":   times_ms.mean().item(),
        "median": times_ms.median().item(),
        "min":    times_ms.min().item(),
        "p95":    torch.quantile(times_ms, 0.95).item(),
    }

# ---- 6. Benchmark End-to-End Forward & Isolated GEMM ----
sample_x, _ = next(iter(loader))
xp_bench = torch.zeros(128, 800, device=device)
xp_bench[:, :784] = sample_x.to(device).view(-1, 784)
x_lp_bench = lptensor(xp_bench, mxi4fp32)
W1_T = W1.T.contiguous()

fp32_e2e  = benchmark_fn(forward_fp32, xp_bench)
mxi4_e2e  = benchmark_fn(forward_mxi4, xp_bench)

# Isolated Layer-1 GEMM only: (128, 800) @ (800, 128) — excludes encode/decode overhead
fp32_gemm = benchmark_fn(lambda a, b: a @ b, xp_bench, W1_T)
mxi4_gemm = benchmark_fn(lambda a, b: a @ b, x_lp_bench, w1_lp)

print(f"{'Benchmark (bs=128)':<26} | {'Mean (ms)':>9} | {'Median (ms)':>11} | {'Min (ms)':>8} | {'P95 (ms)':>8}")
print("-" * 74)
for name, stats in [
    ("FP32 E2E Forward", fp32_e2e),
    ("MXI4 E2E Forward", mxi4_e2e),
    ("FP32 Layer1 GEMM", fp32_gemm),
    ("MXI4 Layer1 GEMM", mxi4_gemm),
]:
    print(f"{name:<26} | {stats['mean']:>9.4f} | {stats['median']:>11.4f} | {stats['min']:>8.4f} | {stats['p95']:>8.4f}")