# =============================================================================
# phase4/clip_mini.py
# =============================================================================
"""
AFTER THIS STEP YOU WILL...
--------------------------
  1. UNDERSTAND contrastive learning — how CLIP trains two completely
     different networks (image encoder, text encoder) to output vectors in
     the SAME space where matched pairs are close and mismatched ones are far.
  2. SEE the InfoNCE loss matrix as an image — every cell = one (image,
     caption) pair. Diagonal = correct matches (should be bright), off-diagonal
     = "wrong pair" (should be dark). This is the visualization every CLIP
     paper shows.
  3. KNOW zero-shot classification: embed the text of "a photo of a circle",
     "a photo of a square", etc., and pick the closest one to a new image's
     embedding. CLIP achieves *classification without ever seeing labeled
     training images* this way.
  4. SEE the shared embedding space via PCA — after training, image vectors
     and text vectors of the same concept sit in nearly the same 2D spot.
  5. HAVE a mini-CLIP you can drop into LLaVA in Phase 5 — the frozen
     vision encoder is exactly this file's image side.

WHAT YOU ARE DOING (one line)
-----------------------------
  Training a small image encoder + small text encoder with contrastive
  loss on synthetic (image, caption) pairs, and visualizing the alignment.

WHAT YOU WILL GET ON DISK
-------------------------
  outputs/clip_01_loss.png            — training loss curve
  outputs/clip_02_similarity.png      — 16×16 image-text similarity matrix
                                         after training (diagonal should be bright)
  outputs/clip_03_tsne_like.png       — 2D PCA of image vs text embeddings,
                                         showing clusters by concept
  outputs/clip_04_retrieval.png       — 8 val images with their top-3
                                         retrieved captions (ranked)
  outputs/clip_checkpoint.pt          — both encoders' weights

SANITY CHECK (what "success" means here)
----------------------------------------
  - Training loss drops from ~log(B) (random) to near 0
  - Diagonal of similarity matrix is bright, off-diagonal dark
  - Retrieval: for each val image, the top-1 caption correctly names it
  - PCA plot: image dots and text dots of same concept are close
  - Zero-shot accuracy on held-out data > 0.9

WHAT FAILURE LOOKS LIKE
-----------------------
  - loss flat at log(B) (≈2.77 for B=16)  → encoders not learning
                                            (LR too low, or shape bug)
  - loss goes to 0 but retrieval wrong     → encoders collapsed:
                                            all vectors became identical
                                            (usually LR too high)
  - retrieval: image retrieves caption of
    a DIFFERENT concept, reliably           → label bug in caption generation
  - similarity matrix all one color         → features collapsed (normalize
                                            by L2 before loss!)
  - FileNotFoundError                       → outputs/ missing (auto-created)

WHERE IT SHIPS IN THE REAL WORLD
--------------------------------
  - CLIP (OpenAI) → image search, content moderation, Stable Diffusion's
    text encoder, and the vision backbone in every VLM listed below.
  - SigLIP (Google) → successor to CLIP, used in PaliGemma and OpenVLA.
  - LLaVA, Qwen-VL, InternVL all use a CLIP or SigLIP ViT.
  - Zero-shot classification is how modern vision models ship without
    fine-tuning — CLIP's trick means one model handles 30,000 categories.
  - The similarity matrix IS the retrieval system behind "search your
    photos by text" in Apple Photos, Google Photos, etc.
"""
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. Two encoders: E_img(image) → ℝ^d, E_txt(caption) → ℝ^d. Separate
#      networks, SAME output dimension d (e.g., 128).
#   2. L2-normalize both embeddings before the loss. This makes cosine
#      similarity = dot product, and prevents the "make everything big"
#      cheat that a naive loss would allow.
#   3. InfoNCE loss:
#         sim = img_emb @ txt_emb.T * logit_scale     # (B, B)
#         loss = 0.5 * (CE(sim, arange(B)) + CE(sim.T, arange(B)))
#      The label is the identity permutation: image i matches caption i.
#      This is contrastive: increase diagonal, decrease off-diagonal.
#   4. Temperature (logit_scale) is LEARNED in real CLIP. It controls how
#      sharp the softmax is. Start at 1/0.07 ≈ 14.
#   5. Symmetric loss: image→text and text→image. Half the batches would
#      be one-sided if we didn't do both; symmetry makes training stable.
# =============================================================================

import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Install pillow:  pip install pillow")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OUT_DIR        = "outputs"
PAIRS_DIR      = "data/tiny_pairs"
IMG_SIZE       = 32
D_EMBED        = 128
BATCH          = 32
N_STEPS        = 1200
LR             = 1e-3
SYNTHETIC_N    = 600
INIT_TEMP      = 1.0 / 0.07              # ≈ 14.3, CLIP's default

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# PART 1 — SYNTHETIC PAIRED DATASET (or real folder if present)
# =============================================================================
SHAPES   = ["circle", "square", "triangle"]
COLORS   = ["red", "green", "blue", "yellow", "purple"]
COLOR_RGB = {
    "red":    (0.9, 0.2, 0.2),
    "green":  (0.2, 0.8, 0.3),
    "blue":   (0.2, 0.3, 0.9),
    "yellow": (0.95, 0.85, 0.2),
    "purple": (0.6, 0.2, 0.8),
}


def _render(shape, color, rng, size=IMG_SIZE):
    img = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy = rng.integers(10, size - 10, size=2)
    r = rng.integers(6, 11)
    rgb = np.array(COLOR_RGB[color], dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    if shape == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    elif shape == "square":
        mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    else:
        mask = ((yy - (cy - r)) >= 0) & (np.abs(xx - cx) <= (yy - (cy - r)) * 0.9)
    img[mask] = rgb
    img += rng.normal(0, 0.04, img.shape)
    return np.clip(img, 0, 1)


def _caption(shape, color):
    # slight phrasing variety — CLIP learns to ignore "a photo of ..."
    templates = [
        f"a {color} {shape}",
        f"a photo of a {color} {shape}",
        f"the {color} {shape}",
    ]
    return templates[np.random.randint(len(templates))]


class SyntheticPairs(torch.utils.data.Dataset):
    """
    (image, caption) pairs generated on the fly, deterministic by seed so
    train/val splits are stable.
    """
    def __init__(self, n=SYNTHETIC_N, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.items = []
        for i in range(n):
            shape = SHAPES[i % len(SHAPES)]
            color = COLORS[(i * 7 + 3) % len(COLORS)]
            self.items.append((shape, color))

    def __len__(self): return self.n

    def __getitem__(self, i):
        shape, color = self.items[i]
        img = _render(shape, color, self.rng)
        cap = _caption(shape, color)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = (img - 0.5) / 0.5
        return img, cap


class FolderPairs(Dataset):
    """Loads data/tiny_pairs/images/*.png + captions.txt."""
    def __init__(self, root, size=IMG_SIZE):
        self.size = size
        self.items = []
        cap_path = os.path.join(root, "captions.txt")
        with open(cap_path) as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line: continue
                fn, cap = line.split("\t", 1)
                self.items.append((os.path.join(root, "images", fn), cap))
        if not self.items:
            raise RuntimeError(f"No pairs found in {root}.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, cap = self.items[i]
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).float(), cap


# =============================================================================
# PART 2 — SIMPLE CHAR-LEVEL TOKENIZER FOR CAPTIONS
# =============================================================================
# Real CLIP uses BPE over ~50k vocab. Char-level is fine here — the encoder
# just has to map captions to vectors; the loss doesn't care about tokenizer.
class TextTokenizer:
    def __init__(self, texts, max_len=24):
        vocab = sorted(set("".join(texts)))
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.pad_id = len(vocab)
        self.vocab_size = len(vocab) + 1
        self.max_len = max_len
    def encode(self, s):
        ids = [self.stoi.get(c, self.pad_id - 1) for c in s.lower()[:self.max_len]]
        return ids + [self.pad_id] * (self.max_len - len(ids))


# =============================================================================
# PART 3 — TWO ENCODERS
# =============================================================================
class ImageEncoder(nn.Module):
    """Small CNN. Output is L2-normalized so its length is 1."""
    def __init__(self, d_out=D_EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, d_out),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TextEncoder(nn.Module):
    """Small transformer-ish stack. Mean-pool over tokens then project."""
    def __init__(self, vocab_size, d_out=D_EMBED, d_model=64, max_len=24):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab_size - 1)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True,
                                           dim_feedforward=128)
        self.tr = nn.TransformerEncoder(layer, num_layers=2)
        self.proj = nn.Linear(d_model, d_out)
    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.emb(ids) + self.pos(pos)[None]
        # pad mask: True = ignore (transformer convention)
        pad = (ids == self.emb.padding_idx)
        x = self.tr(x, src_key_padding_mask=pad)
        # masked mean pool
        valid = (~pad).unsqueeze(-1).float()
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return F.normalize(self.proj(pooled), dim=-1)


# =============================================================================
# PART 4 — CONTRASTIVE LOSS
# =============================================================================
def clip_loss(img_emb, txt_emb, logit_scale):
    """
    img_emb, txt_emb: (B, d), already L2-normalized
    logit_scale: scalar, exp'd (see below) — CLIP learns this parameter.

    The target is the identity: image i matches caption i.
    Symmetric loss = image→text retrieval + text→image retrieval.
    """
    scale = logit_scale.exp().clamp(max=100.0)
    sim = scale * img_emb @ txt_emb.T           # (B, B)
    targets = torch.arange(len(img_emb), device=img_emb.device)
    return 0.5 * (F.cross_entropy(sim, targets) +
                  F.cross_entropy(sim.T, targets))


# =============================================================================
# PART 5 — VISUALIZATION
# =============================================================================
def plot_loss(losses, path=f"{OUT_DIR}/clip_01_loss.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(losses, color="C0")
    ax.set_xlabel("Step"); ax.set_ylabel("Contrastive loss")
    ax.set_title("InfoNCE training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_similarity(img_emb, txt_emb, path=f"{OUT_DIR}/clip_02_similarity.png"):
    """The (B, B) similarity matrix. Diagonal = matches = should be bright."""
    sim = (img_emb @ txt_emb.T).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xlabel("Caption index"); ax.set_ylabel("Image index")
    ax.set_title("Image-text similarity matrix\n(diagonal = correct pairs)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_pca(img_embs, txt_embs, img_labels, txt_labels,
             path=f"{OUT_DIR}/clip_03_tsne_like.png"):
    """PCA to 2D of combined embeddings. Same-concept image and text dots
    should cluster together after training."""
    from numpy.linalg import svd
    all_emb = np.vstack([img_embs, txt_embs])
    mean = all_emb.mean(0, keepdims=True)
    X = all_emb - mean
    U, S, Vt = svd(X, full_matrices=False)
    pc = X @ Vt[:2].T                            # (N, 2)

    img_2d = pc[:len(img_embs)]
    txt_2d = pc[len(img_embs):]
    concepts = sorted(set(img_labels + txt_labels))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))
    for ci, c in enumerate(concepts):
        col = cmap(ci % 10)
        m_i = [i for i, l in enumerate(img_labels) if l == c]
        m_t = [i for i, l in enumerate(txt_labels) if l == c]
        ax.scatter(img_2d[m_i, 0], img_2d[m_i, 1], s=60, marker="o",
                   color=col, edgecolor="black", linewidth=0.3,
                   label=f"{c} (image)")
        ax.scatter(txt_2d[m_t, 0], txt_2d[m_t, 1], s=60, marker="X",
                   color=col, edgecolor="black", linewidth=0.3,
                   label=f"{c} (text)")
    ax.set_title("Shared embedding space (PCA 2D)\nO = image, X = text")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_retrieval(images, caps, img_emb, txt_emb, n=8,
                   path=f"{OUT_DIR}/clip_04_retrieval.png"):
    """For each val image, list top-3 captions by similarity. Top should match."""
    sim = (img_emb @ txt_emb.T).cpu().numpy()
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 1, figsize=(6, 1.6 * n))
    if n == 1: axes = [axes]
    for i, ax in enumerate(axes):
        img = (images[i] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        top3 = np.argsort(-sim[i])[:3]
        caption_lines = [f"#{r+1}: {caps[j][:50]}" for r, j in enumerate(top3)]
        correct = (top3[0] == i)
        ax.imshow(np.tile(img, (1, 6, 1)))         # stretch to make text room
        ax.axis("off")
        for r, line in enumerate(caption_lines):
            ax.text(0.02 + img.shape[1] * 6 + 0.1, r + 0.7, line,
                    fontsize=8, color="green" if (r == 0 and correct) else "black")
    fig.suptitle("Image → text retrieval (top-3 captions per image)")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# =============================================================================
# PART 6 — MAIN
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)

    # ---- Data ------------------------------------------------------------
    if os.path.isdir(PAIRS_DIR) and os.path.exists(os.path.join(PAIRS_DIR, "captions.txt")):
        print(f"Using real pairs from {PAIRS_DIR}/")
        full = FolderPairs(PAIRS_DIR)
        texts = [cap for _, cap in full.items]
    else:
        print(f"No real pairs at {PAIRS_DIR}/ — using synthetic (image, caption)")
        full = SyntheticPairs(SYNTHETIC_N, seed=0)
        texts = [cap for (_, cap) in [full[i] for i in range(len(full))]]

    # split 80/20 for eval
    n = len(full); k = int(0.8 * n)
    tr_idx = list(range(k)); va_idx = list(range(k, n))
    train_ds = torch.utils.data.Subset(full, tr_idx)
    val_ds   = torch.utils.data.Subset(full, va_idx)

    tokenizer = TextTokenizer(texts)
    print(f"Dataset: {n} pairs, vocab (chars) = {tokenizer.vocab_size}")

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        caps = [b[1] for b in batch]
        ids = torch.tensor([tokenizer.encode(c) for c in caps])
        return imgs, ids, caps

    train_dl = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate)

    # ---- Encoders --------------------------------------------------------
    img_enc = ImageEncoder().to(device)
    txt_enc = TextEncoder(tokenizer.vocab_size, max_len=tokenizer.max_len).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(INIT_TEMP)))
    n_params = sum(p.numel() for p in img_enc.parameters()) + \
               sum(p.numel() for p in txt_enc.parameters())
    print(f"Total encoder params: {n_params:,}")

    opt = torch.optim.AdamW(
        list(img_enc.parameters()) + list(txt_enc.parameters()) + [logit_scale],
        lr=LR, weight_decay=0.01,
    )

    # ---- Train -----------------------------------------------------------
    print(f"\n[1/4] Training for {N_STEPS} steps...")
    losses = []
    step = 0; it = iter(train_dl)
    while step < N_STEPS:
        try: imgs, ids, _ = next(it)
        except StopIteration: it = iter(train_dl); imgs, ids, _ = next(it)
        imgs, ids = imgs.to(device), ids.to(device)
        ie = img_enc(imgs)
        te = txt_enc(ids)
        loss = clip_loss(ie, te, logit_scale)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"scale={logit_scale.exp().item():.1f}")
        step += 1

    plot_loss(losses)

    # ---- Evaluate --------------------------------------------------------
    print("\n[2/4] Evaluating retrieval on val split")
    img_enc.eval(); txt_enc.eval()
    all_imgs, all_ids, all_caps = [], [], []
    with torch.no_grad():
        for imgs, ids, caps in val_dl:
            all_imgs.append(imgs); all_ids.append(ids); all_caps.extend(caps)
    imgs_v = torch.cat(all_imgs).to(device)
    ids_v  = torch.cat(all_ids).to(device)
    with torch.no_grad():
        ie_v = img_enc(imgs_v)
        te_v = txt_enc(ids_v)
    sim = (ie_v @ te_v.T).cpu().numpy()
    top1 = (sim.argmax(axis=1) == np.arange(len(imgs_v))).mean()
    print(f"  top-1 retrieval accuracy: {top1:.3f}")

    # zero-shot: build a caption per class, pick argmax
    if isinstance(full, SyntheticPairs):
        zs_caps = [f"a {c} {s}" for s in SHAPES for c in COLORS]
        zs_labels = [f"{c} {s}" for s in SHAPES for c in COLORS]
        with torch.no_grad():
            zs_ids = torch.tensor([tokenizer.encode(c) for c in zs_caps]).to(device)
            zs_emb = txt_enc(zs_ids)                     # (n_zs, d)
        scores = ie_v @ zs_emb.T                          # (n_val, n_zs)
        # true label for each val image
        true_labels = [zs_labels[0]] * len(imgs_v)  # placeholder
        # derive from cap text
        def parse(cap):
            words = cap.lower().replace("a photo of ", "").replace("the ", "").split()
            shape = next((s for s in SHAPES if s in words), "?")
            color = next((c for c in COLORS if c in words), "?")
            return f"{color} {shape}"
        true_labels = [parse(c) for c in all_caps]
        pred_idx = scores.argmax(1).cpu().numpy()
        pred_labels = [zs_labels[i] for i in pred_idx]
        zs_acc = np.mean([p == t for p, t in zip(pred_labels, true_labels)])
        print(f"  zero-shot classification accuracy: {zs_acc:.3f}")
    else:
        zs_acc = None
        true_labels = None

    # ---- Visualizations --------------------------------------------------
    print("\n[3/4] Similarity matrix + PCA + retrieval")
    plot_similarity(ie_v, te_v)
    img_labels = true_labels if true_labels is not None else ["?"] * len(imgs_v)
    txt_labels = true_labels if true_labels is not None else ["?"] * len(all_caps)
    plot_pca(ie_v.cpu().numpy(), te_v.cpu().numpy(), img_labels, txt_labels)
    plot_retrieval(imgs_v.cpu(), all_caps, ie_v.cpu(), te_v.cpu())

    torch.save({"img": img_enc.state_dict(), "txt": txt_enc.state_dict(),
                "logit_scale": logit_scale.item()}, f"{OUT_DIR}/clip_checkpoint.pt")
    print(f"\n[4/4] Saved {OUT_DIR}/clip_checkpoint.pt")
    print(f"\nFinal top-1 retrieval: {top1:.3f}"
          + (f", zero-shot acc: {zs_acc:.3f}" if zs_acc is not None else ""))