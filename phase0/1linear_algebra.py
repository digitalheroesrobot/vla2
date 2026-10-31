# =============================================================================
# phase0/linear_algebra.py
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. Matrix multiplication written by hand — so it's never a black box again.
#      Every Linear layer, every attention score, every embedding lookup in a
#      VLM is one of these.
#   2. Numerical gradient — how you *verify* a gradient you derived by hand.
#      Used by torch.autograd.gradcheck in every serious ML repo.
#   3. Numerical JACOBIAN — the vector-output generalization of the gradient.
#      This is what's actually flowing backward through a linear layer.
#   4. Softmax + cross-entropy + its gradient — the single most-used loss in
#      ALL of deep learning. GPT, LLaVA, RT-2 all use exactly this.
#   5. Chain rule on a scalar — the core of backprop, in its smallest form.
#
# WHY IT MATTERS TECHNICALLY
#   - Matmul shapes decide GPU memory. (B,H,T,T) attention scores blow up
#     quadratically with visual token count — this is why LLaVA uses 576
#     tokens per image, not 4096.
#   - CE gradient is beautifully simple: softmax(logits) - onehot(y). Once
#     you see it, every "cross-entropy loss" line in every repo makes sense.
#   - Numeric-vs-analytic gradient check is the debugging tool that catches
#     99% of "my model won't train" bugs. Learn it now, save weeks later.
#   - The Jacobian demo teaches you WHY vector-valued functions need a
#     matrix of derivatives — not a vector. This is exactly the shape that
#     autograd internally works with (before VJPs collapse it).
#
# WHERE IT SHIPS IN THE REAL WORLD
#   - PyTorch's F.cross_entropy = softmax + NLL + this exact gradient, fused
#     into one CUDA kernel. Runs on every training step of every LLM/VLM.
#   - RT-2's action decoder uses cross-entropy over discretized action bins —
#     literally the same loss you're writing here, just with 256 bins per joint.
#   - torch.autograd.gradcheck is what PyTorch engineers run before merging
#     any new op (FlashAttention, quantization, custom kernels) into mainline.
#   - Attention kernels (FlashAttention-2, PagedAttention in vLLM) spend 90%
#     of their effort optimizing the two matmuls you write below.
# =============================================================================

import numpy as np


# -----------------------------------------------------------------------------
# 1. Matrix multiplication from scratch
# -----------------------------------------------------------------------------
def matmul_naive(A, B):
    """A: (n, m), B: (m, p) -> (n, p). Pure Python loops, no numpy @."""
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "shape mismatch"
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C


# -----------------------------------------------------------------------------
# 2. Numerical gradient (finite differences) — SCALAR output only
# -----------------------------------------------------------------------------
def numerical_grad(f, x, eps=1e-6):
    """Central-difference gradient for a scalar function f, vector input x.

    grad[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)

    Precision ~O(eps^2). Small enough eps gives accuracy ~1e-7 for float64.

    IMPORTANT: f must return a SCALAR (single number). If f returns a
    vector, use numerical_jacobian() below instead.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = float(f(xp))
        fm = float(f(xm))
        grad[i] = (fp - fm) / (2 * eps)
    return grad


# -----------------------------------------------------------------------------
# 3. Numerical JACOBIAN — for VECTOR-output functions f: R^n -> R^m
# -----------------------------------------------------------------------------
def numerical_jacobian(f, x, eps=1e-6):
    """Jacobian of f: R^n -> R^m at point x.

    Returns J with shape (m, n):  J[i, j] = df_i / dx_j.

    This is what autograd actually builds internally. For a Linear layer
    y = Wx, the Jacobian d y / d x is exactly W (m x n). Every backward
    pass through a Linear layer is a matmul by this Jacobian (VJP).
    """
    x = np.asarray(x, dtype=np.float64)
    f0 = np.asarray(f(x), dtype=np.float64).ravel()
    m = f0.size
    n = x.size
    J = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        fp = np.asarray(f(xp), dtype=np.float64).ravel()
        fm = np.asarray(f(xm), dtype=np.float64).ravel()
        J[:, j] = (fp - fm) / (2 * eps)
    return J


# -----------------------------------------------------------------------------
# 4. Softmax + cross-entropy + its analytic gradient
# -----------------------------------------------------------------------------
def softmax(x):
    """Stable softmax over last axis (subtract max to avoid exp overflow)."""
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def cross_entropy(logits, y):
    """Mean cross-entropy loss. logits: (B, C). y: (B,) integer labels."""
    p = softmax(logits)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()


def cross_entropy_grad(logits, y):
    """Analytic gradient of cross_entropy w.r.t. logits.

    Result: (softmax(logits) - onehot(y)) / B. Beautifully simple.
    That single formula is the gradient at the heart of every LM training
    step on Earth.
    """
    p = softmax(logits)
    p[np.arange(len(y)), y] -= 1.0
    return p / len(y)


# -----------------------------------------------------------------------------
# 5a. Chain rule on a SCALAR function (the common case)
# -----------------------------------------------------------------------------
def chain_rule_demo_scalar():
    """f(x) = sum((3x + 1)^2), so df/dx_i = 2*(3x_i + 1)*3.

    Summing makes the output a scalar — matching numerical_grad's contract.
    Sums over independent inputs are the standard trick to reduce a
    per-example vector loss into one training scalar.
    """
    def f(x): return float(((3 * x + 1) ** 2).sum())
    def df_dx(x): return 2 * (3 * x + 1) * 3
    x = np.array([0.5, -1.0, 2.0])
    print("analytic:", df_dx(x))
    print("numeric :", numerical_grad(f, x))


# -----------------------------------------------------------------------------
# 5b. Chain rule on a VECTOR function (the Jacobian case)
# -----------------------------------------------------------------------------
def chain_rule_demo_jacobian():
    """f(x) = (3x + 1)^2 elementwise, so f: R^3 -> R^3.

    The Jacobian is diagonal: df_i/dx_j = 0 when i != j, and
    df_i/dx_i = 2*(3*x_i + 1)*3. Diagonal because the inputs are independent.
    """
    def f(x): return (3 * x + 1) ** 2
    def jacobian_analytic(x): return np.diag(2 * (3 * x + 1) * 3)
    x = np.array([0.5, -1.0, 2.0])
    print("analytic Jacobian:\n", jacobian_analytic(x))
    print("numeric  Jacobian:\n", numerical_jacobian(f, x))


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # (1) matmul matches numpy
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 5))
    print("matmul match:", np.allclose(matmul_naive(A, B), A @ B))

    # (2) analytic CE gradient matches numeric CE gradient
    logits = rng.standard_normal((4, 10))
    y = np.array([1, 3, 5, 7])
    ana = cross_entropy_grad(logits, y)
    num = numerical_grad(
        lambda z: cross_entropy(z.reshape(4, 10), y),
        logits.flatten(),
    ).reshape(4, 10)
    print("max gradient error:", np.abs(ana - num).max())  # expect < 1e-6

    # (3a) chain rule — scalar case
    print("\n--- chain rule (scalar) ---")
    chain_rule_demo_scalar()

    # (3b) chain rule — Jacobian case
    print("\n--- chain rule (Jacobian) ---")
    chain_rule_demo_jacobian()