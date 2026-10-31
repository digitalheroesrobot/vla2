# =============================================================================
# phase2/attention_from_scratch.py
# =============================================================================
"""
AFTER THIS STEP YOU WILL...
--------------------------
  1. UNDERSTAND Q, K, V — the three matrices inside every transformer.
     Q = "what am I looking for?", K = "what can I offer?", V = "what do I
     actually give if selected?". Every attention call is softmax(QKᵀ/√d)·V.
  2. SEE the attention matrix as an image. Each row = a query token; each
     column = a key token; brightness = how much that token attended. You'll
     look at a real 8-token sentence and see which words attend to which.
  3. UNDERSTAND the mask. Causal mask (decoder) hides the future. No mask
     (encoder, ViT) lets every token see every other. The mask IS the
     multimodal architecture in a VLM (see phase 4/5).
  4. KNOW WHY /√d exists. Without it, QKᵀ grows with d, softmax saturates,
     gradients vanish. With it, logits stay O(1) and training is stable.
  5. HAVE a reusable MultiHeadAttention class you'll drop into the GPT
     you build next (Phase 2 Step 2).

WHAT YOU ARE DOING (one line)
-----------------------------
  Implementing scaled dot-product attention + multi-head attention from
  scratch on a toy 8-token "sentence", then drawing the attention matrix
  for single-head and multi-head variants.

WHAT YOU WILL GET ON DISK
-------------------------
  outputs/attn_01_matrix.png       — single-head attention matrix (8×8)
  outputs/attn_02_causal.png       — same, but causal-masked
  outputs/attn_03_multihead.png    — 4 heads side by side, each 8×8
  outputs/attn_04_variance.png     — logits variance vs d (with/without scale)

SANITY CHECK (what "success" means here)
----------------------------------------
  - The 8×8 attention matrix rows sum to 1.0 (softmax output) → engine correct
  - The causal matrix has zeros strictly above the diagonal → mask works
  - Multi-head matrix shows FOUR DIFFERENT patterns → heads specialize
  - The variance plot shows raw variance ∝ d, scaled variance ≈ 1 → the
    /√d justification is empirically visible
  - If any of these fail, look at "WHAT FAILURE LOOKS LIKE" below

WHAT FAILURE LOOKS LIKE
-----------------------
  - rows don't sum to 1.0         → softmax applied on wrong axis
  - causal matrix has bright upper triangle → mask not applied / wrong sign
  - all 4 heads look identical    → weights collapsed (LR too high, or
                                    one-shot init not random enough)
  - variance plot: scaled var ≠ 1 → forgot to divide by sqrt(d)

WHERE IT SHIPS IN THE REAL WORLD
--------------------------------
  - Every transformer block you'll ever build (GPT, ViT, LLaVA's LLM,
    RT-2's action decoder, π0's action expert) uses this exact function.
  - The attention matrix is the debugging tool for "why did my model
    predict X?". When LLaVA describes an image wrong, engineers look at the
    visual-token attention map to see WHICH PATCHES the model looked at.
  - FlashAttention (used by every production LLM) is a rewrite of this
    function that computes the SAME answer with O(N) memory instead of O(N²).
    Once you understand this file, FlashAttention is just an optimization.
  - Flamingo's Perceiver Resampler (DeepMind) uses cross-attention — Q from
    text, K/V from image patches — to compress a video into 64 fixed tokens.
    This exact mechanism is how modern VLMs handle video.
"""
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. Scaled dot-product attention:
#         attn(Q, K, V) = softmax(Q Kᵀ / √d_k) V
#      Shape analysis:
#         Q: (B, H, Tq, d_k)   "queries"
#         K: (B, H, Tk, d_k)   "keys"
#         V: (B, H, Tk, d_v)   "values"
#         scores = Q @ Kᵀ / √d_k         (B, H, Tq, Tk)
#         weights = softmax(scores)       (B, H, Tq, Tk)  rows sum to 1
#         output = weights @ V            (B, H, Tq, d_v)
#   2. Multi-head: split d_model into H heads, run H attentions in parallel,
#      concatenate. Each head has its own W_q, W_k, W_v so it can specialize.
#   3. Causal mask: before softmax, set scores at future positions to -inf.
#      Softmax then gives them probability 0. That's the whole trick.
#   4. Why √d_k: if Q, K ~ N(0, 1) each of dim d, then Q·K has variance d.
#      Dividing by √d normalizes variance to 1 → softmax stays sensitive.
# =============================================================================

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# PART 1 — SCALED DOT-PRODUCT ATTENTION (the core function)
# =============================================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, H, Tq, d_k)
    K: (B, H, Tk, d_k)
    V: (B, H, Tk, d_v)
    mask: (B, 1, Tq, Tk) boolean; True = keep, False = mask out (-inf)

    Returns:
      out:     (B, H, Tq, d_v)   weighted sum of values
      weights: (B, H, Tq, Tk)    attention probabilities (rows sum to 1)
    """
    d_k = Q.size(-1)
    # (1) Score every query against every key. This is the matmul that
    #     FlashAttention later rewrites to avoid materializing (Tq, Tk).
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # (2) Apply mask (add -inf before softmax so masked entries get prob 0)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # (3) Softmax over keys -> per-query probability distribution
    weights = F.softmax(scores, dim=-1)

    # (4) Weighted sum of values
    out = weights @ V
    return out, weights


# =============================================================================
# PART 2 — MULTI-HEAD ATTENTION (nn.Module)
# =============================================================================
class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention. Learnable projections W_q, W_k, W_v, W_o.
    d_model is split across H heads of size d_head = d_model // H.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, return_weights=False):
        """
        x: (B, T, d_model)
        mask: (B, 1, T, T) boolean, True = keep
        """
        B, T, _ = x.shape
        # Project, then split into heads: (B, T, H, d_head) -> (B, H, T, d_head)
        def split(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q = split(self.q_proj(x))
        K = split(self.k_proj(x))
        V = split(self.v_proj(x))

        out, weights = scaled_dot_product_attention(Q, K, V, mask)  # (B,H,T,d_head)
        # Merge heads: (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        if return_weights:
            return out, weights
        return out


# =============================================================================
# PART 3 — VISUALIZATION
# =============================================================================
def plot_attention_matrix(weights, tokens, title, path):
    """
    weights: (Tq, Tk) numpy array, rows sum to 1
    tokens:  list of token strings (length Tk)
    """
    fig, ax = plt.subplots(figsize=(max(5, len(tokens) * 0.6),
                                    max(4, len(tokens) * 0.6)))
    im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens)
    ax.set_xlabel("Key (what is attended TO)")
    ax.set_ylabel("Query (who is attending)")
    ax.set_title(title)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            v = weights[i, j]
            color = "white" if v < 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_multihead(weights_per_head, tokens, path):
    """
    weights_per_head: list of (H) matrices, each (Tq, Tk)
    """
    H = len(weights_per_head)
    fig, axes = plt.subplots(1, H, figsize=(H * 3.2, 3.4))
    if H == 1: axes = [axes]
    for h, (ax, W) in enumerate(zip(axes, weights_per_head)):
        ax.imshow(W, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=45,
                                                              ha="right", fontsize=7)
        ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"head {h}", fontsize=9)
    fig.suptitle("Multi-head attention: each head learns a different pattern", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_variance(path):
    """
    Show that Q·K variance grows with d unless you divide by √d.
    """
    rng = np.random.default_rng(0)
    dims = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    raw_vars, scaled_vars = [], []
    for d in dims:
        Q = rng.standard_normal((2000, d))
        K = rng.standard_normal((2000, d))
        dots = (Q * K).sum(-1)
        raw_vars.append(dots.var())
        scaled_vars.append((dots / math.sqrt(d)).var())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dims, raw_vars, marker="o", label="raw Q·K variance")
    ax.plot(dims, scaled_vars, marker="s", label="scaled by 1/√d")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("d_k"); ax.set_ylabel("variance")
    ax.set_title("Why / √d_k matters: variance grows with d unless scaled")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# =============================================================================
# PART 4 — MAIN
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)

    print("=" * 70)
    print("Attention from scratch — Q, K, V, mask, multi-head, why √d")
    print("=" * 70)

    # ---- 1. A toy 8-token "sentence" -------------------------------------
    # Real transformers use BPE over ~50k vocab. Here we use 8 words to
    # keep the attention matrix small enough to print every number.
    tokens = ["the", "cat", "sat", "on", "the", "warm", "red", "mat"]
    T = len(tokens)
    d_model = 32
    n_heads = 4

    # Fake embeddings: one random vector per token. Real transformers learn
    # these; here we just need 8 distinct vectors to demonstrate attention.
    x = torch.randn(1, T, d_model)                          # (1, 8, 32)

    # ---- 2. Single-head attention -----------------------------------------
    print("\n[1/4] Single-head attention")
    mha_1 = MultiHeadAttention(d_model=d_model, n_heads=1)
    _, W1 = mha_1(x, return_weights=True)                   # (1, 1, 8, 8)
    W1 = W1[0, 0].detach().numpy()
    print(f"  attention matrix shape: {W1.shape}")
    print(f"  rows sum to 1?  {np.allclose(W1.sum(axis=-1), 1.0)}")
    plot_attention_matrix(W1, tokens, "Single-head attention (no mask)",
                          f"{OUT_DIR}/attn_01_matrix.png")

    # ---- 3. Causal (decoder) mask ----------------------------------------
    print("\n[2/4] Causal mask")
    causal = torch.triu(torch.ones(T, T), diagonal=1).bool()  # True above diag
    mask = ~causal                                             # keep, except upper
    mask = mask.unsqueeze(0).unsqueeze(0)                      # (1,1,T,T)
    mha_c = MultiHeadAttention(d_model=d_model, n_heads=1)
    _, Wc = mha_c(x, mask=mask, return_weights=True)
    Wc = Wc[0, 0].detach().numpy()
    print(f"  upper triangle max: {Wc[np.triu_indices(T, k=1)].max():.6f}  (should be ~0)")
    plot_attention_matrix(Wc, tokens, "Causal (decoder) attention",
                          f"{OUT_DIR}/attn_02_causal.png")

    # ---- 4. Multi-head attention -----------------------------------------
    print("\n[3/4] Multi-head attention")
    mha_h = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    _, Wh = mha_h(x, return_weights=True)                   # (1, H, 8, 8)
    Wh = Wh[0].detach().numpy()                             # (H, 8, 8)
    plot_multihead([Wh[h] for h in range(n_heads)], tokens,
                   f"{OUT_DIR}/attn_03_multihead.png")

    # ---- 5. Why √d_k -----------------------------------------------------
    print("\n[4/4] Why 1/√d_k")
    plot_variance(f"{OUT_DIR}/attn_04_variance.png")

    # ---- Final sanity report ---------------------------------------------
    print("\nSanity report:")
    print(f"  rows sum to 1 in single-head:   {np.allclose(W1.sum(-1), 1.0)}")
    print(f"  causal upper triangle is 0:     {(Wc[np.triu_indices(T, k=1)] < 1e-5).all()}")
    print(f"  each head differs from head 0:  "
          f"{[float(np.abs(Wh[h] - Wh[0]).mean()) > 0.05 for h in range(n_heads)]}")

    print(f"\nAll plots saved to ./{OUT_DIR}/attn_*.png")