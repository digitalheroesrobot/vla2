# =============================================================================
# phase0/broadcasting_and_norms.py
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. NumPy/PyTorch BROADCASTING rules — how shapes (B, T, D) + (D,) just work,
#      and why (B, T, D) + (B, 1) also works but (B, T, D) + (B, T) does not.
#   2. The four norms you'll meet in EVERY VLM: L2 norm, LayerNorm, RMSNorm,
#      and BatchNorm. Why LayerNorm is in transformers, why BatchNorm isn't.
#   3. Why attention uses scaled dot-product = QK^T / sqrt(d_k): the math
#      behind the scale factor. The scale factor exists because of variance,
#      and variance is what LayerNorm normalizes.
#   4. The "silent bugs": shape mismatches that broadcast to the WRONG shape
#      and quietly compute garbage. You'll learn to test for them.
#
# WHY IT MATTERS TECHNICALLY
#   - Broadcasting is a memory-saving trick: a (1, D) bias "copies" to (B, T, D)
#     without actually allocating B*T copies. That's why nn.Linear adds bias
#     in O(D) memory, not O(B*T*D).
#   - LayerNorm is what makes deep transformers trainable. Without it,
#     activations explode or collapse by layer 40. Every VLM has 24-80 of them.
#   - RMSNorm (LLaMA/Qwen) is LayerNorm minus the mean-centering. Slightly
#     cheaper, works just as well. Modern VLMs (LLaVA, Qwen-VL, π0) use it.
#   - The `/ sqrt(d_k)` scale keeps softmax from saturating → keeps gradients
#     alive → keeps the model trainable. If d_k = 128, Q·K can be ~128.
#     Divided by sqrt(128) ≈ 11.3, it's back to ~1. Softmax behaves.
#
# WHERE IT SHIPS IN THE REAL WORLD
#   - EVERY LayerNorm in LLaVA, Qwen-VL, LLaMA, GPT: the exact formula below,
#     scaled up. PyTorch has an optimized kernel.
#   - RMSNorm is what LLaMA-3, Qwen2, Mistral, and π0 use. It's the modern
#     default. It exists in the wild *because* of the experiment you'll run.
#   - The "silent bug" test is exactly what code reviewers do at labs: they
#     assert every tensor's shape right after the layer that produced it.
#     Google/DeepMind call this "shape-check discipline".
#   - The `/sqrt(d_k)` scale appears in every attention implementation from
#     the original Transformer paper to FlashAttention-3's CUDA kernel.
# =============================================================================

import numpy as np


# -----------------------------------------------------------------------------
# 1. Broadcasting — the rules, demonstrated
# -----------------------------------------------------------------------------
def broadcasting_demo():
    """Broadcasting aligns two shapes from the RIGHT and pads the shorter one
    with 1s. Then dims must match or be 1. If any dim is 1 it 'stretches'."""
    print("--- broadcasting ---")
    A = np.random.randn(2, 3, 4)        # (B, T, D)

    # (D,) → stretched to (B, T, D). This is how a Linear layer's bias works.
    b = np.random.randn(4)
    print("A + b shape:", (A + b).shape)     # (2, 3, 4)

    # (1, 1, D) also works.
    b2 = np.random.randn(1, 1, 4)
    print("A + b2 shape:", (A + b2).shape)   # (2, 3, 4)

    # (1, T, 1) also works — a per-timestep scalar multiply.
    scale = np.random.randn(1, 3, 1)
    print("A * scale shape:", (A * scale).shape)  # (2, 3, 4)

    # (T,) does NOT broadcast to (B, T, D). It aligns from the right → (T,)
    # matches the LAST dim, which is D=4, so if T != D it fails. If T == D it
    # silently broadcasts to the WRONG thing. This is the classic silent bug.
    print("\nSilent bug example:")
    A2 = np.random.randn(2, 4, 4)   # T=4, D=4 — makes (4,) ambiguous!
    x = np.random.randn(4)
    # Did the user mean (1, 1, D) or (1, T, 1)? Shape math can't tell.
    # If they meant per-token scaling but T == D, THIS CODE WILL RUN and
    # silently compute the wrong thing.
    print("A2 + x shape:", (A2 + x).shape)   # (2,4,4) — silently wrong if intent was per-token

    # Fix: always name your dims with explicit dim=1 where you mean broadcast.
    x_per_token = x.reshape(1, 4, 1)        # (1, T, 1) — intent is now explicit
    print("A2 + x_per_token shape:", (A2 + x_per_token).shape)
    print("Lesson: if T == D, use explicit reshape(1, T, 1).")


# -----------------------------------------------------------------------------
# 2. L2 norm and normalized embeddings
# -----------------------------------------------------------------------------
def l2_normalize(x, eps=1e-12):
    """x / ||x|| along last axis. Used by CLIP to put image and text embeddings
    on the unit hypersphere, so cosine similarity = dot product."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def l2_demo():
    print("\n--- L2 normalize ---")
    v = np.array([[3.0, 4.0], [1.0, 0.0]])
    n = l2_normalize(v)
    print("norms after:", np.linalg.norm(n, axis=-1))    # [1. 1.]
    # Real-world: CLIP keeps IMAGE and TEXT embeddings both L2-normalized.
    # Then cos_sim(i, t) = i · t, and CLIP's logits are just i @ t.T * scale.


# -----------------------------------------------------------------------------
# 3. LayerNorm — the transformer default
# -----------------------------------------------------------------------------
def layernorm(x, gamma, beta, eps=1e-5):
    """x: (..., D). Normalize over D. gamma/beta: (D,) learnable scale/shift."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


def rmsnorm(x, gamma, eps=1e-6):
    """LLaMA/Qwen/π0 norm: normalize by RMS only, no mean-centering, no beta.
    Cheaper (one fewer pass) and empirically as good. Now the default in
    every new LLM after LLaMA-2."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)


def norm_demo():
    print("\n--- LayerNorm vs RMSNorm ---")
    D = 8
    x = np.random.randn(2, 5, D) * 3.0 + 1.0  # deliberately non-zero mean & big scale
    gamma_ln = np.ones(D)
    beta_ln  = np.zeros(D)
    gamma_rms = np.ones(D)

    y_ln  = layernorm(x, gamma_ln, beta_ln)
    y_rms = rmsnorm(x, gamma_rms)

    # After LN: mean ~0, std ~1 per row (if gamma=1,beta=0)
    print("LN  per-row mean:", y_ln.mean(axis=-1).round(3))
    print("LN  per-row std :", y_ln.std(axis=-1).round(3))
    # After RMS: std ~1 *only if mean was already ~0*; here mean is nonzero,
    # so RMS = sqrt(mean^2 + var) and y_rms.std() won't be exactly 1.
    print("RMS per-row mean:", y_rms.mean(axis=-1).round(3))
    print("RMS per-row std :", y_rms.std(axis=-1).round(3))
    # Lesson: RMSNorm DOES NOT center. If your activations have a mean offset
    # (as after a bias-heavy layer), RMSNorm "sees" that as signal, LayerNorm
    # removes it. In practice LLMs learn to live without the centering, and
    # save the FLOPs. That's why LLaMA-2+ moved to RMSNorm.


# -----------------------------------------------------------------------------
# 4. Why `Q @ K.T / sqrt(d_k)` — the variance math
# -----------------------------------------------------------------------------
def attention_scale_demo():
    """If Q, K are iid N(0, 1) of dim d, then Q·K has variance d.
    Dividing by sqrt(d) gives variance 1. Softmax saturates if variance
    is high → vanishing gradients. This is why the scale exists."""
    print("\n--- why / sqrt(d_k) ---")
    for d in [4, 16, 64, 256]:
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((2000, d))
        K = rng.standard_normal((2000, d))
        raw = (Q * K).sum(-1)                    # 2000 dot products
        print(f"d={d:4d}  raw var={raw.var():.2f}  scaled var={(raw/np.sqrt(d)).var():.2f}")


# -----------------------------------------------------------------------------
# 5. BatchNorm vs LayerNorm — why transformers use LN, not BN
# -----------------------------------------------------------------------------
def bn_vs_ln():
    """BatchNorm normalizes over the BATCH dim. LayerNorm normalizes over
    the FEATURE dim. For transformers with variable-length sequences and
    small batch sizes, BN stats are noisy / batch-dependent. LN is
    per-sample and therefore deterministic at inference — critical for
    robots where the same image must give the same action every time."""
    print("\n--- BN vs LN in one line ---")
    print("BN: x - mean(x, over batch) / std(x, over batch)  ← depends on batch")
    print("LN: x - mean(x, over features) / std(x, over features) ← per-sample")
    print("VLA policy: same input MUST give same action → LN wins by design.")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    broadcasting_demo()
    l2_demo()
    norm_demo()
    attention_scale_demo()
    bn_vs_ln()

    # The "silent bug" check that saves your future self:
    print("\n--- shape check discipline ---")
    x = np.random.randn(2, 5, 8)          # pretend: (B=2, T=5, D=8)
    normed = layernorm(x, np.ones(8), np.zeros(8))
    assert normed.shape == x.shape, f"LayerNorm should preserve shape, got {normed.shape}"
    # If you're EVER unsure, add these asserts right after every layer.
    # Code reviewers at labs do exactly this before merge.
    print("LayerNorm preserves shape OK:", normed.shape)