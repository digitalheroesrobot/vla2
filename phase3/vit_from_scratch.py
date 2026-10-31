# =============================================================================
# phase3/vit_from_scratch.py
# =============================================================================
"""
AFTER THIS STEP YOU WILL...
--------------------------
  1. UNDERSTAND how a ViT turns an image into a sequence of tokens
     (patchify: cut into 4×4 or 8×8 patches, project each to a vector),
     and why this lets you reuse the SAME transformer block from Phase 2.
  2. SEE attention maps drawn ON TOP of the image — bright = "the model
     looked here to make its decision". This is what interpretability teams
     stare at when debugging real vision models.
  3. COMPARE ViT vs CNN on your photo folder. The ViT has more parameters
     but no conv inductive bias — you'll see if it wins or needs more data.
     (Spoiler: on 24–120 images the CNN usually wins. That's a *lesson*.)
  4. HAVE a working ViT feature extractor — this becomes the vision encoder
     inside LLaVA in Phase 5, unfrozen only for the last stage.
  5. EXPLAIN positional embeddings, patch embeddings, CLS token, and why
     ViT needs more data than a CNN to match it.

WHAT YOU ARE DOING (one line)
-----------------------------
  Building a minimal ViT, training it on your photo folder (or the bundled
  sample), and rendering attention maps to see where it looks.

WHAT YOU WILL GET ON DISK
-------------------------
  outputs/vit_01_patches.png       — the input image + its 4×4 patches
  outputs/vit_02_attention.png     — attention map from the CLS token to
                                     every patch, overlayed on the image
  outputs/vit_03_predictions.png   — 16 val images, green=correct red=wrong
  outputs/vit_04_curve.png         — loss + val accuracy
  outputs/vit_checkpoint.pt        — trained weights

DATASET CONTRACT
----------------
  Uses data/tiny_classify/ if present, else the auto-generated sample
  (data/tiny_classify_sample/). No downloads. See phase1/DATA.md.

SANITY CHECK (what "success" means here)
----------------------------------------
  - Attention map has at least one bright patch per class → model is
    actually looking at the object, not the background.
  - Val accuracy on the sample (shapes) is > 0.8 with a small ViT, but
    likely < 0.9 — ViT needs more data than CNN to reach high accuracy.
  - Loss decreases smoothly. If flat, LR too low. If NaN, LR too high.
  - Patches plot shows the input cleanly cut into a grid (no overlap).

WHAT FAILURE LOOKS LIKE
-----------------------
  - attention map uniform gray           → model isn't using attention yet
                                            (undertrained, or LR too low)
  - attention map looks like noise       → probably too few steps
  - val_acc < 0.5 on shapes               → patch size or embed dim mismatch
  - loss NaN                              → LR too high (try 1e-4)
  - FileNotFoundError on save             → outputs/ missing (this file makes it)

WHERE IT SHIPS IN THE REAL WORLD
--------------------------------
  - LLaVA-1.5's vision encoder is CLIP ViT-L/14-336. Same architecture
    (patchify + transformer blocks + LN), 24 blocks, 1024 dim, FROZEN.
  - Qwen-VL uses a custom ViT. PaliGemma uses SigLIP (ViT variant).
  - OpenVLA feeds a SigLIP ViT's patch tokens straight into the LLM.
  - The attention-map overlay is the standard debugging tool when a VLM
    describes an image wrong — "did the encoder see the object at all?"
"""
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. Patchify: a (3, 64, 64) image with patch=8 → 64 patches of shape
#      (3, 8, 8) → flattened to 192-dim → projected to d_model (e.g. 128).
#      Result: (N_patches, d_model) = (64, 128). Same shape as text tokens!
#   2. CLS token: a learnable vector prepended to the patch sequence.
#      After transformer blocks, its output summarizes the whole image.
#      Classification head sits on it. (ViTs often use mean-pool of patch
#      tokens instead — we use CLS here because attention maps are cleaner.)
#   3. Positional embeddings: patches lose their spatial layout if you
#      just flatten them. Add a learnable (N_patches + 1, d_model) table.
#   4. No causal mask in ViT. Every patch can attend to every other patch.
#      This is why ViT is O(N²) — but N=196 (patch=14 on 224) is small.
#   5. Why ViT needs more data than CNN: convs *assume* locality and
#      translation-equivariance. ViT *learns* those from data. With 120
#      images, CNN's assumption wins. With 120M images (JFT-300M in the
#      original ViT paper), ViT wins.
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
DATA_DIR   = "data/tiny_classify"
SAMPLE_DIR = "data/tiny_classify_sample"
OUT_DIR    = "outputs"
IMG_SIZE   = 64
PATCH      = 8                          # 8x8 → 64 patches per image
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 4
BATCH      = 16
N_EPOCHS   = 25
LR         = 3e-4
VAL_SPLIT  = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# PART 1 — DATA (same folder contract as Phase 1)
# =============================================================================
def ensure_sample_dataset(path=SAMPLE_DIR):
    if os.path.isdir(path) and any(glob.glob(os.path.join(path, "*", "*.png"))):
        return
    print(f"Generating sample reference dataset at {path}/ ...")
    os.makedirs(path, exist_ok=True)
    for shape in ["circle", "square", "triangle"]:
        cls_dir = os.path.join(path, shape)
        os.makedirs(cls_dir, exist_ok=True)
        rng = np.random.default_rng(hash(shape) % (2**32))
        for i in range(8):
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            cx, cy = rng.integers(18, IMG_SIZE - 18, size=2)
            r = rng.integers(10, 20)
            color = rng.uniform(0.15, 1.0, size=3)
            yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
            if shape == "circle":
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            elif shape == "square":
                mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
            else:
                mask = ((yy - (cy - r)) >= 0) & (np.abs(xx - cx) <= (yy - (cy - r)) * 0.9)
            img[mask] = color
            img += rng.normal(0, 0.05, img.shape)
            arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(cls_dir, f"{i:04d}.png"))
    print(f"  wrote 24 sample images to {path}/")


def count_images(d):
    if not os.path.isdir(d): return 0
    n = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        n += len(glob.glob(os.path.join(d, "*", ext)))
    return n


class PhotoFolderDataset(Dataset):
    def __init__(self, root, size=IMG_SIZE, augment=False):
        self.size = size; self.augment = augment
        self.samples = []; self.class_names = []
        for idx, cls in enumerate(sorted(os.listdir(root))):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                files.extend(glob.glob(os.path.join(cls_dir, ext)))
            files = sorted(files)
            if not files: continue
            self.class_names.append(cls)
            for f in files: self.samples.append((f, idx))
            print(f"  class '{cls}': {len(files)} images")
        if not self.samples: raise RuntimeError(f"No images under {root}.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment:
            if np.random.rand() < 0.5: arr = arr[:, ::-1, :]
            arr = arr * np.random.uniform(0.8, 1.2)
            s = np.random.randint(0, 4)
            if s:
                arr = arr[s:-s, s:-s, :]
                im2 = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                im2 = im2.resize((self.size, self.size))
                arr = np.asarray(im2, dtype=np.float32) / 255.0
            arr = np.clip(arr, 0, 1)
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).float(), label


# =============================================================================
# PART 2 — VISION TRANSFORMER
# =============================================================================
class PatchEmbed(nn.Module):
    """
    (B, 3, H, W) → (B, N_patches, d_model).
    Implemented as a strided Conv2d with kernel=stride=patch. This is the
    canonical ViT patchify — mathematically identical to cutting patches
    and linearly projecting each one, but faster on GPU.
    """
    def __init__(self, img=IMG_SIZE, patch=PATCH, in_ch=3, d=D_MODEL):
        super().__init__()
        assert img % patch == 0
        self.n_patches = (img // patch) ** 2
        self.proj = nn.Conv2d(in_ch, d, kernel_size=patch, stride=patch)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)    # (B, N, d)


class ViTBlock(nn.Module):
    """Pre-LN transformer block — same shape as GPT block, but NO causal mask."""
    def __init__(self, d, h, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d, d), nn.Dropout(dropout),
        )
    def forward(self, x, return_attn=False):
        h = self.ln1(x)
        if return_attn:
            h, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        else:
            h, _ = self.attn(h, h, h, need_weights=False)
            w = None
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return (x, w) if return_attn else x


class ViT(nn.Module):
    """
    Minimal Vision Transformer. Returns class logits on the CLS token.
    """
    def __init__(self, img=IMG_SIZE, patch=PATCH, d=D_MODEL, h=N_HEADS,
                 n_layers=N_LAYERS, n_classes=3):
        super().__init__()
        self.patch = PatchEmbed(img, patch, 3, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch.n_patches, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(d, h) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, return_attn=False):
        x = self.patch(x)
        B = x.size(0)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos
        mid_attn = None
        mid = len(self.blocks) // 2
        for i, b in enumerate(self.blocks):
            if return_attn and i == mid:
                x, w = b(x, return_attn=True)
                mid_attn = w                    # (B, N+1, N+1)
            else:
                x = b(x)
        x = self.ln(x)
        logits = self.head(x[:, 0])            # classify on CLS
        return (logits, mid_attn) if return_attn else logits


# =============================================================================
# PART 3 — TRAIN / EVAL
# =============================================================================
def train_epoch(model, loader, opt, loss_fn):
    model.train(); running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, return_preds=False):
    model.eval(); correct = 0; total = 0
    ap, at = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        correct += (p == y).sum().item(); total += y.size(0)
        if return_preds: ap.append(p.cpu()); at.append(y.cpu())
    acc = correct / total
    if return_preds: return acc, torch.cat(ap), torch.cat(at)
    return acc


# =============================================================================
# PART 4 — VISUALIZATION
# =============================================================================
def _un(t): return (t * 0.5 + 0.5).clamp(0, 1)


def plot_patches(image_tensor, patch=PATCH, path=f"{OUT_DIR}/vit_01_patches.png"):
    """Show the input image and its patch grid. Visualizes the ViT's input."""
    img = _un(image_tensor).permute(1, 2, 0).numpy()
    H, W = img.shape[:2]
    n = H // patch
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("input image"); axes[0].axis("off")
    axes[1].imshow(img)
    for i in range(n + 1):
        axes[1].axhline(i * patch - 0.5, color="red", lw=0.7)
        axes[1].axvline(i * patch - 0.5, color="red", lw=0.7)
    axes[1].set_title(f"patchified: {n}×{n} = {n*n} patches of {patch}×{patch}")
    axes[1].axis("off")
    fig.suptitle("ViT input — image → tokens")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_attention_overlay(model, image_tensor, patch=PATCH,
                           path=f"{OUT_DIR}/vit_02_attention.png"):
    """
    CLS-token attention to each patch, upscaled and overlaid on the image.
    Bright = the CLS token looked there. This is THE vision debugging plot.
    """
    model.eval()
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)
        _, w = model(x, return_attn=True)      # (1, N+1, N+1)
    # CLS row, patch columns (skip CLS col)
    cls_attn = w[0, 0, 1:].cpu().numpy()       # (N,)
    n_side = int(round(math.sqrt(len(cls_attn))))
    heat = cls_attn.reshape(n_side, n_side)
    # upscale to image size
    heat_up = np.kron(heat, np.ones((patch, patch)))

    img = _un(image_tensor).permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img); axes[0].set_title("input"); axes[0].axis("off")
    im = axes[1].imshow(heat, cmap="viridis")
    axes[1].set_title("CLS attention per patch"); axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(img)
    axes[2].imshow(heat_up, cmap="hot", alpha=0.55)
    axes[2].set_title("overlay"); axes[2].axis("off")
    fig.suptitle("Where the ViT looks (middle block, CLS token)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_predictions(dataset, model, class_names, n=16,
                     path=f"{OUT_DIR}/vit_03_predictions.png"):
    model.eval()
    idx = np.random.RandomState(1).choice(len(dataset), min(n, len(dataset)), replace=False)
    imgs = torch.stack([dataset[i][0] for i in idx]).to(device)
    with torch.no_grad(): preds = model(imgs).argmax(1).cpu().tolist()
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, i, p in zip(axes.flat, idx, preds):
        img, label = dataset[i]
        ax.imshow(_un(img).permute(1, 2, 0).numpy())
        color = "green" if p == label else "red"
        ax.set_title(f"true: {class_names[label]}\npred: {class_names[p]}",
                     fontsize=8, color=color); ax.axis("off")
    fig.suptitle("ViT predictions  (green=correct, red=wrong)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_curve(tl, va, path=f"{OUT_DIR}/vit_04_curve.png"):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss", color="C0")
    ax1.plot(tl, color="C0", marker="o"); ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Val accuracy", color="C1")
    ax2.plot(va, color="C1", marker="s"); ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(0, 1)
    fig.suptitle("ViT training progress")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# =============================================================================
# PART 5 — MAIN
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)

    # ---- Data ------------------------------------------------------------
    if count_images(DATA_DIR) == 0:
        ensure_sample_dataset(SAMPLE_DIR)
        print(f"\nNo images in {DATA_DIR}/ — using sample {SAMPLE_DIR}/")
        active = SAMPLE_DIR
    else:
        print(f"\nFound images in {DATA_DIR}/ — using them")
        active = DATA_DIR

    train_full = PhotoFolderDataset(active, augment=True)
    val_full   = PhotoFolderDataset(active, augment=False)
    class_names = train_full.class_names
    n_classes = len(class_names)
    print(f"\nTotal images: {len(train_full)} | Classes: {class_names}")

    rng = np.random.default_rng(0)
    all_idx = np.arange(len(train_full)); rng.shuffle(all_idx)
    n_val = max(1, int(len(all_idx) * VAL_SPLIT))
    val_idx = all_idx[:n_val].tolist()
    train_idx = all_idx[n_val:].tolist()
    train_ds = torch.utils.data.Subset(train_full, train_idx)
    val_ds   = torch.utils.data.Subset(val_full,   val_idx)
    train_dl = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=2)

    # ---- Model -----------------------------------------------------------
    model = ViT(img=IMG_SIZE, patch=PATCH, d=D_MODEL, h=N_HEADS,
                n_layers=N_LAYERS, n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT params: {n_params:,} | patches/image: {model.patch.n_patches}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # ---- Train -----------------------------------------------------------
    print("\n[1/4] Training...")
    tl_hist, va_hist = [], []
    for epoch in range(N_EPOCHS):
        tl = train_epoch(model, train_dl, opt, loss_fn)
        va = evaluate(model, val_dl)
        tl_hist.append(tl); va_hist.append(va)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:2d}  loss={tl:.4f}  val_acc={va:.4f}")

    # ---- Visuals ---------------------------------------------------------
    print("\n[2/4] Visualizing patches and attention...")
    sample_img, _ = val_ds[0]
    plot_patches(sample_img)
    plot_attention_overlay(model, sample_img)

    print("\n[3/4] Predictions...")
    plot_predictions(val_full, model, class_names)

    print("\n[4/4] Curve...")
    plot_curve(tl_hist, va_hist)

    torch.save(model.state_dict(), f"{OUT_DIR}/vit_checkpoint.pt")
    print(f"\nSaved checkpoint: {OUT_DIR}/vit_checkpoint.pt")
    print(f"Final val_acc: {va_hist[-1]:.4f}")