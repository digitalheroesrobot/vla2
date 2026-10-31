# =============================================================================
# phase0/matmul_benchmark.py
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. How to BENCHMARK Python code correctly (warmup, repeats, min-of-N).
#   2. How naive Python matmul compares to NumPy (BLAS) — orders of magnitude.
#   3. How NumPy (CPU) compares to PyTorch (CPU) vs PyTorch (GPU, if any).
#   4. FLOP counting: turning wall-clock time into hardware utilization %.
#   5. Why matmul is the *bottleneck* of every VLM, and what the real
#      numbers look like at LLaVA-like shapes.
#
# WHY IT MATTERS TECHNICALLY
#   - Naive Python matmul is ~100,000x slower than a BLAS kernel. That gap
#     is cache locality + SIMD + multithreading. GPUs widen it further.
#   - Every VLM forward pass is dominated by a handful of matmuls. Knowing
#     their FLOP cost tells you WHY a bigger image = slower inference,
#     WHY 8-bit quantization gives ~2x speedup, and WHY your laptop can't
#     train LLaVA but can *run* a small one.
#   - FLOPS utilization (% of theoretical peak) is how ML infra engineers
#     judge whether a training run is "efficient" or "wasting GPUs".
#
# WHERE IT SHIPS IN THE REAL WORLD
#   - Meta/OpenAI/Anthropic training clusters: engineers compute exactly
#     this FLOP-per-step number to predict run cost before spending $10M.
#   - llama.cpp / vLLM / TensorRT-LLM: matmul kernel choice (FP16 vs INT8
#     vs INT4) is chosen from micro-benchmarks like this one.
#   - Robot VLA inference: if your matmul takes 60ms on the robot's Jetson,
#     your policy runs at 16Hz max. Benchmark first, deploy later.
#   - Quantization papers (e.g., GPTQ, AWQ): every speedup number they
#     quote comes from the same kind of timing loop you'll write here.
# =============================================================================

import time
import numpy as np

# -----------------------------------------------------------------------------
# 1. Benchmark helper — the *correct* way to time code
# -----------------------------------------------------------------------------
def benchmark(fn, *args, warmup=2, repeats=5, **kwargs):
    """Return min time (seconds) over `repeats` runs after `warmup` runs.

    WHY min and not mean:
      Timing is one-sided noise — interruptions by the OS scheduler, other
      processes, cache misses. The MIN captures the clean run. This is the
      standard method used by `timeit` and every ML systems paper.
    WHY warmup:
      First call includes import/first-touch costs, JIT, cuBLAS init (on GPU),
      and cold caches. Warmup makes the timed calls steady-state.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times)


# -----------------------------------------------------------------------------
# 2. FLOP math — how many multiplies for one matmul
# -----------------------------------------------------------------------------
def matmul_flops(n, m, p):
    """Floating-point ops for (n,m) @ (m,p).
    Each output element needs m multiplies + (m-1) adds ≈ 2m flops.
    Total = n * p * 2 * m."""
    return 2 * n * m * p


# -----------------------------------------------------------------------------
# 3. The three implementations we'll compare
# -----------------------------------------------------------------------------
def matmul_naive(A, B):
    """Pure Python triple loop — for intuition only. Never use in production."""
    n, m = A.shape
    _, p = B.shape
    C = [[0.0] * p for _ in range(n)]
    # Convert to lists to avoid numpy scalar overhead per element.
    A_list = A.tolist()
    B_list = B.tolist()
    for i in range(n):
        Ci = C[i]
        Ai = A_list[i]
        for k in range(m):
            aik = Ai[k]
            Bk = B_list[k]
            for j in range(p):
                Ci[j] += aik * Bk[j]
    return C


def matmul_numpy(A, B):
    """NumPy uses BLAS: cache-tiled, SIMD, multi-threaded. This is the CPU
    baseline every ML framework builds on top of."""
    return A @ B


def matmul_torch(A_t, B_t):
    """PyTorch dispatches to its own BLAS (MKL on CPU, cuBLAS on GPU). Same
    math, different kernel — with the advantage that it can also run on GPU."""
    return A_t @ B_t


# -----------------------------------------------------------------------------
# 4. Report a matmul comparison at a given shape
# -----------------------------------------------------------------------------
def compare_shape(n, m, p, run_naive=True):
    print(f"\n=== shape ({n}, {m}) @ ({m}, {p}) ===")
    print(f"  output size = {n*p:,} elements")
    print(f"  FLOPs       = {matmul_flops(n,m,p):,} (~{matmul_flops(n,m,p)/1e6:.1f} MFLOP)")

    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, m)).astype(np.float32)
    B = rng.standard_normal((m, p)).astype(np.float32)

    # ---- NumPy ----
    t_np = benchmark(matmul_numpy, A, B)
    gflops_np = matmul_flops(n, m, p) / t_np / 1e9
    print(f"  numpy  : {t_np*1e3:9.3f} ms  → {gflops_np:7.2f} GFLOP/s")

    # ---- PyTorch CPU ----
    try:
        import torch
        A_t = torch.from_numpy(A)
        B_t = torch.from_numpy(B)
        t_cpu = benchmark(matmul_torch, A_t, B_t)
        gflops_cpu = matmul_flops(n, m, p) / t_cpu / 1e9
        print(f"  torch CPU: {t_cpu*1e3:9.3f} ms → {gflops_cpu:7.2f} GFLOP/s")

        # ---- PyTorch GPU (if available) ----
        if torch.cuda.is_available():
            A_g = A_t.cuda()
            B_g = B_t.cuda()
            torch.cuda.synchronize()
            # GPU timing needs a sync because launches are async — without
            # sync, you'd be measuring how fast CUDA *enqueues*, not computes.
            def run_gpu():
                torch.matmul(A_g, B_g)
                torch.cuda.synchronize()
            t_gpu = benchmark(run_gpu, warmup=3, repeats=10)
            gflops_gpu = matmul_flops(n, m, p) / t_gpu / 1e9
            print(f"  torch GPU: {t_gpu*1e3:9.3f} ms → {gflops_gpu:7.2f} GFLOP/s")
    except ImportError:
        print("  torch not installed; skipping PyTorch comparison")

    # ---- Naive Python (small shapes only!) ----
    if run_naive:
        t_py = benchmark(matmul_naive, A, B, warmup=0, repeats=1)
        gflops_py = matmul_flops(n, m, p) / t_py / 1e9
        speedup = t_py / t_np
        print(f"  python : {t_py*1e3:9.3f} ms  → {gflops_py:7.4f} GFLOP/s"
              f"  ({speedup:,.0f}x slower than numpy)")


# -----------------------------------------------------------------------------
# 5. main — the shapes that MATTER for a VLM
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Small sanity-check shape — safe to include naive Python here.
    compare_shape(16, 16, 16, run_naive=True)

    # One Linear layer of LLaVA's LLM. Single image, one token, hidden=4096.
    # This is the exact shape inside `self.q_proj(hidden)` in every attention.
    compare_shape(1, 4096, 4096, run_naive=False)

    # The image side of LLaVA-1.5:
    #   - 576 visual tokens (24x24 patches) × hidden 4096
    #   - this matmul runs several times per forward pass
    compare_shape(576, 4096, 4096, run_naive=False)

    # One attention score matmul, batch=1, heads=32, seq=576, head_dim=128.
    # This one gets reshaped as (32, 576, 128) @ (32, 128, 576) — one launch
    # per head (or batched), producing (32, 576, 576). Notice the 576² memory
    # — that's the O(N²) cost that FlashAttention tames.
    compare_shape(32 * 576, 128, 576, run_naive=False)

    # Bigger shape — a whole 8k-token context matmul.
    # This is the type that pushes training clusters.
    compare_shape(8192, 4096, 4096, run_naive=False)

    print("\nNote: GFLOP/s utilization vs peak tells you if the kernel")
    print("is compute-bound (good) or memory-bound (wasting hardware).")