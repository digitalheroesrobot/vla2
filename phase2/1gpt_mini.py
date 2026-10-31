# =============================================================================
# phase2/gpt_mini.py
# =============================================================================
"""
# When you want real text later (optional)

Replace `data/tiny_text/corpus.txt` with any text file < 5 MB:
- Project Gutenberg plain-text books (~1 MB each)
- A few pages of Wikipedia you copy-paste
- Lyrics, poems, technical docs — anything you like

The script uses a **character-level tokenizer** (vocab ~70 chars). Real text
works fine but needs more steps

AFTER THIS STEP YOU WILL...
--------------------------
  1. UNDERSTAND a decoder-only language model end-to-end: embeddings,
     causal transformer blocks, layer norm, residual connections, LM head.
  2. SEE the training loss fall on a real (small) text corpus, and watch
     the model generate coherent sentences for the first time.
  3. KNOW why "GPT" is really just an MLP with attention, at scale, and
     why the same architecture appears inside every VLM.
  4. HAVE a working text generator you can point at any .txt file.
  5. BE ABLE to explain the difference between GPT (decoder) and ViT
     (encoder), and why LLaVA bolts a ViT-style vision encoder to a GPT.

WHAT YOU ARE DOING (one line)
-----------------------------
  Assembling embedding + N transformer blocks + LM head into a GPT-style
  model, training it on a bundled tiny text corpus, and generating text.

WHAT YOU WILL GET ON DISK
-------------------------
  data/tiny_text/corpus.txt        — auto-generated corpus (~30 KB)
  outputs/gpt_01_loss.png          — train loss curve
  outputs/gpt_02_generations.txt   — sample generations at 4 checkpoints
  outputs/gpt_03_attention.png     — attention matrix at a middle layer
                                     for one generated sample
  outputs/gpt_checkpoint.pt        — trained model weights

DATASET CONTRACT
----------------
  data/tiny_text/corpus.txt (auto-generated if missing, or yours).
  See phase2/DATA.md. No downloads.

SANITY CHECK (what "success" means here)
----------------------------------------
  - Loss drops from ~4.2 to < 1.0 over training (chars, ~70 vocab)
  - Generated text contains real words from the corpus vocabulary
  - Later generations are MORE coherent than earlier ones (checkpoint log)
  - Attention matrix shows a diagonal-ish causal band (not upper triangle)
  - If loss plateaus > 3.0, LR is too low or block_size too small

WHAT FAILURE LOOKS LIKE
-----------------------
  - loss flat at ~4.2           → weights not updating (check opt.step)
  - loss → nan                  → LR too high, or init std too big
  - generated text is garbage    → undertrained (more steps), or corpus too small
  - attention has upper-triangle brightness → causal mask bug
  - FileNotFoundError on save    → outputs/ missing (this file creates it)

WHERE IT SHIPS IN THE REAL WORLD
--------------------------------
  - This is the text core of LLaVA, Qwen-VL, PaliGemma: a small-to-medium
    GPT-like model (Vicuna, Qwen, Gemma) that receives concatenated
    [image_tokens | text_tokens] and predicts next text tokens.
  - Same shape for RT-2's action decoder: the "text" becomes action tokens.
  - Every "chat" product you've used — ChatGPT, Claude, Gemini, Copilot —
    is this file, at 100-1000x the scale, with better data and training.
  - The attention-matrix plot here is how interpretability teams look at
    WHICH tokens a model attends to before predicting the next one.
"""
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. A GPT block = LayerNorm + CausalMHA + residual + LayerNorm + MLP +
#      residual. Same block repeated N times.
#   2. Tokens from a character tokenizer. Real GPT uses BPE (tiktoken); the
#      math is identical — token IDs in, logits over the same vocab out.
#   3. Causal mask: each position may attend to positions <= itself. That
#      is what makes next-token prediction a valid training signal.
#   4. Autoregressive sampling: at each step, take logits at the last
#      position, softmax -> probs, sample a token, append, repeat.
#   5. Loss = cross-entropy on "predict token t+1 from tokens 0..t".
# =============================================================================

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR   = "outputs"
TEXT_DIR  = os.path.join("data", "tiny_text")
TEXT_FILE = os.path.join(TEXT_DIR, "corpus.txt")
CKPT      = os.path.join(OUT_DIR, "gpt_checkpoint.pt")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# PART 1 — TINY TEXT CORPUS (auto-generated if missing)
# =============================================================================
def ensure_corpus(path=TEXT_FILE):
    """
    Generates a ~200-line synthetic corpus from a small grammar.
    Structure is repetitive enough that a tiny GPT learns it quickly —
    perfect for verifying the training loop works before scaling up.
    """
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    print(f"Generating synthetic corpus at {path} ...")
    subjects = ["the quick brown fox", "the lazy dog", "a small red bird",
                "the old grey cat", "a young green frog"]
    verbs    = ["jumps over", "sleeps near", "sings to", "watches",
                "runs past"]
    objects  = ["the tall tree", "the warm sun", "the cold river",
                "a tiny stone", "the quiet house"]
    adverbs  = ["slowly", "quickly", "quietly", "happily", "carefully"]
    rng = random.Random(42)
    lines = []
    for _ in range(400):
        lines.append(
            f"{rng.choice(subjects)} {rng.choice(verbs)} "
            f"{rng.choice(objects)} {rng.choice(adverbs)}."
        )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(f"  wrote {len(text)/1024:.1f} KB")


# =============================================================================
# PART 2 — CHARACTER TOKENIZER (simplest possible)
# =============================================================================
class CharTokenizer:
    """
    Char-level tokenizer. Real GPT uses BPE (tiktoken, ~50k vocab). The
    math after this step is identical — token IDs in, logits over vocab out.
    """
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
    def encode(self, s): return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids): return "".join(self.itos[int(i)] for i in ids)
    @property
    def vocab_size(self): return len(self.vocab)


# =============================================================================
# PART 3 — TRANSFORMER BLOCK
# =============================================================================
class CausalSelfAttention(nn.Module):
    """Multi-head causal attention (from phase2/attention_from_scratch.py)."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, return_weights=False):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)              # (B, H, T, dh)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        out = weights @ v                                  # (B, H, T, dh)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.proj(out)
        if return_weights: return out, weights
        return out


class Block(nn.Module):
    """Pre-LN transformer block: LN -> attn -> residual -> LN -> MLP -> residual."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x, return_weights=False):
        if return_weights:
            h, w = self.attn(self.ln1(x), return_weights=True)
            x = x + h
            x = x + self.mlp(self.ln2(x))
            return x, w
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    Decoder-only language model. Default: 6 layers, 6 heads, 192 dim
    (~2.5M params). Trains to interesting output on the synthetic corpus
    in ~2 min on CPU.
    """
    def __init__(self, vocab_size, block_size=128, d_model=192,
                 n_heads=6, n_layers=6, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=0.8, top_k=20):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# =============================================================================
# PART 4 — DATASET
# =============================================================================
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size
    def __len__(self): return len(self.ids) - self.block_size
    def __getitem__(self, i):
        x = self.ids[i:i+self.block_size]
        y = self.ids[i+1:i+self.block_size+1]
        return x, y


# =============================================================================
# PART 5 — VISUALIZATION
# =============================================================================
def plot_loss(losses, path=f"{OUT_DIR}/gpt_01_loss.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(losses, color="C0")
    ax.set_xlabel("Step"); ax.set_ylabel("Cross-entropy (chars)")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def save_generations(text, path=f"{OUT_DIR}/gpt_02_generations.txt"):
    with open(path, "w") as f: f.write(text)
    print(f"  saved {path}")


def plot_attention(model, tokenizer, prompt, path=f"{OUT_DIR}/gpt_03_attention.png"):
    """Draw attention matrix from the MIDDLE block on a short prompt."""
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    with torch.no_grad():
        B, T = ids.shape
        pos = torch.arange(T, device=device)
        x = model.tok_emb(ids) + model.pos_emb(pos)[None]
        # find the middle block
        mid = len(model.blocks) // 2
        for i, b in enumerate(model.blocks):
            if i == mid:
                x, w = b(x, return_weights=True)      # w: (1, H, T, T)
            else:
                x = b(x)
    W = w[0, 0].cpu().numpy()                          # first head
    tokens = list(prompt[:T])
    fig, ax = plt.subplots(figsize=(max(5, T*0.35), max(4, T*0.35)))
    im = ax.imshow(W, cmap="viridis", vmin=0, vmax=W.max() + 1e-9)
    ax.set_xticks(range(T)); ax.set_xticklabels(tokens, fontsize=7)
    ax.set_yticks(range(T)); ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xlabel("Key (attended TO)"); ax.set_ylabel("Query (attending)")
    ax.set_title(f"Causal attention — block {mid+1}/{len(model.blocks)}, head 0",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# =============================================================================
# PART 6 — MAIN
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    print(f"Using device: {device}")
    print("=" * 70)
    print("GPT-mini — assembling attention into a language model")
    print("=" * 70)

    # ---- 1. Data ---------------------------------------------------------
    ensure_corpus()
    text = open(TEXT_FILE, "r").read()
    tokenizer = CharTokenizer(text)
    print(f"\nCorpus: {len(text):,} chars, vocab: {tokenizer.vocab_size}")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n_train], ids[n_train:]

    BLOCK = 96
    train_ds = CharDataset(train_ids, BLOCK)
    val_ds   = CharDataset(val_ids,   BLOCK)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=32, shuffle=False)

    # ---- 2. Model --------------------------------------------------------
    model = MiniGPT(tokenizer.vocab_size, block_size=BLOCK).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, {len(model.blocks)} blocks")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # ---- 3. Train --------------------------------------------------------
    print("\n[1/5] Training")
    N_STEPS = 600
    losses = []
    ckpt_steps = [0, 150, 300, 599]
    generations = {}
    for step in range(N_STEPS):
        try: xb, yb = next(it)
        except (NameError, StopIteration):
            it = iter(train_dl); xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step in ckpt_steps:
            prompt = "the quick brown fox "
            gen_ids = model.generate(
                torch.tensor([tokenizer.encode(prompt)], device=device),
                max_new_tokens=60, temperature=0.7, top_k=15
            )
            gen_text = tokenizer.decode(gen_ids[0].cpu().tolist())
            generations[step] = gen_text
            print(f"  step {step:3d}  loss={loss.item():.4f}")

    # ---- 4. Evaluate -----------------------------------------------------
    print("\n[2/5] Evaluating on val split")
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_dl:
            _, l = model(xb.to(device), yb.to(device))
            val_losses.append(l.item())
    print(f"  val loss = {np.mean(val_losses):.4f}")

    # ---- 5. Save outputs -------------------------------------------------
    print("\n[3/5] Saving loss curve")
    plot_loss(losses)

    print("\n[4/5] Saving generations")
    out = "Sample generations at checkpoints (same prompt: "
    out += "'the quick brown fox ')\n\n"
    for step in ckpt_steps:
        out += f"--- step {step} ---\n{generations[step]}\n\n"
    save_generations(out)

    print("\n[5/5] Plotting attention")
    plot_attention(model, tokenizer, "the quick brown fox ")

    torch.save(model.state_dict(), CKPT)
    print(f"  saved {CKPT}")

    # ---- Summary ---------------------------------------------------------
    print("\nSummary")
    print(f"  params: {n_params:,}")
    print(f"  final train loss: {losses[-1]:.4f}")
    print(f"  final val loss:   {np.mean(val_losses):.4f}")
    print(f"  all outputs saved to ./{OUT_DIR}/")