# =============================================================================
# phase1/mlp_on_your_photos.py  —  REAL image folder, with bundled sample
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. Full real-world image pipeline: scan → load → split → augment → train.
#   2. Why augmentation is mandatory on real photos (40 imgs is tiny).
#   3. Why we split by FILE, not by random index (prevents leakage).
#   4. Failure-mode reading: predictions / confusion / per-class accuracy.
#   5. Graceful degradation: if user has no photos yet, fall back to a
#      bundled reference dataset — so the file is never "broken on day 1".
#
# THE FALLBACK CONTRACT
#   - If data/tiny_classify/  has images → use them (real photos).
#   - Else → auto-create data/tiny_classify_sample/ from a rendered
#     template (24 PNGs, ~24 KB), use that, and print instructions.
#   This makes the file ALWAYS runnable — the sample shows users exactly
#   what their real folder should look like.
#
# WHERE IT SHIPS IN THE REAL WORLD
#   - This "graceful fallback" pattern is used in every ML library:
#     HuggingFace falls back to small models, scikit-learn has toy datasets,
#     PyTorch has random tensors. It lets new users SEE the API work before
#     committing to data collection.
#   - The two-source pattern (real vs sample) is how production teams keep
#     CI fast: unit tests hit the sample, end-to-end runs hit real data.
# =============================================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
DATA_DIR     = "data/tiny_classify"          # your real photos
SAMPLE_DIR   = "data/tiny_classify_sample"   # auto-generated reference
OUT_DIR      = "outputs"
IMG_SIZE     = 64
N_EPOCHS     = 20
BATCH        = 32
LR           = 1e-3
VAL_SPLIT    = 0.2
FORCE_SAMPLE = False   # set True to ignore your real photos and use the sample

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Auto-generate the sample dataset if it doesn't exist yet
# -----------------------------------------------------------------------------
def ensure_sample_dataset(path=SAMPLE_DIR):
    """Create 24 PNGs (8 per class) if not already present."""
    if os.path.isdir(path) and any(glob.glob(os.path.join(path, "*", "*.png"))):
        return
    print(f"Generating sample reference dataset at {path}/ ...")
    os.makedirs(path, exist_ok=True)
    classes = ["circle", "square", "triangle"]
    for shape in classes:
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
    print(f"  ✓ wrote 24 sample images to {path}/")
    print(f"  → inspect them with:  xdg-open {path}/")


def count_images(d):
    if not os.path.isdir(d): return 0
    return len(glob.glob(os.path.join(d, "*", "*.png")) +
               glob.glob(os.path.join(d, "*", "*.jpg")) +
               glob.glob(os.path.join(d, "*", "*.jpeg")))


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class PhotoFolderDataset(Dataset):
    def __init__(self, root, size=IMG_SIZE, augment=False):
        self.size = size
        self.augment = augment
        self.samples = []
        self.class_names = []

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Folder '{root}' not found.")

        for idx, cls in enumerate(sorted(os.listdir(root))):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
                files.extend(glob.glob(os.path.join(cls_dir, ext)))
            files = sorted(files)
            if not files:
                continue
            self.class_names.append(cls)
            for f in files:
                self.samples.append((f, idx))
            print(f"  class '{cls}': {len(files)} images")

        if not self.samples:
            raise RuntimeError(f"No images found under {root}.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment:
            if np.random.rand() < 0.5: arr = arr[:, ::-1, :]
            arr = arr * np.random.uniform(0.8, 1.2)
            s = np.random.randint(0, 6)
            if s:
                arr = arr[s:-s, s:-s, :]
                im2 = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                im2 = im2.resize((self.size, self.size))
                arr = np.asarray(im2, dtype=np.float32) / 255.0
            arr = np.clip(arr, 0, 1)
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).float(), label


# -----------------------------------------------------------------------------
# Model / loop / plots — same as before
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=3 * IMG_SIZE * IMG_SIZE, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x.flatten(1))


def train_epoch(model, loader, opt, loss_fn):
    model.train(); running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); loss = loss_fn(model(x), y)
        loss.backward(); opt.step()
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


def _un(t): return (t * 0.5 + 0.5).clamp(0, 1)


def plot_samples(dataset, class_names, n=16, path=f"{OUT_DIR}/real_01_samples.png"):
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    idxs = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else list(range(len(dataset)))
    idx = np.random.RandomState(0).choice(idxs, min(n, len(idxs)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, i in zip(axes.flat, idx):
        img, label = base[i]
        ax.imshow(_un(img).permute(1, 2, 0).numpy())
        ax.set_title(class_names[label], fontsize=9); ax.axis("off")
    fig.suptitle("Dataset samples (ground truth)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_predictions(dataset, model, class_names, n=16,
                     path=f"{OUT_DIR}/real_02_predictions.png"):
    model.eval()
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    idxs = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else list(range(len(dataset)))
    idx = np.random.RandomState(1).choice(idxs, min(n, len(idxs)), replace=False)
    imgs = torch.stack([base[i][0] for i in idx]).to(device)
    with torch.no_grad(): preds = model(imgs).argmax(1).cpu().tolist()
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, i, p in zip(axes.flat, idx, preds):
        img, label = base[i]
        ax.imshow(_un(img).permute(1, 2, 0).numpy())
        color = "green" if p == label else "red"
        ax.set_title(f"true: {class_names[label]}\npred: {class_names[p]}",
                     fontsize=8, color=color); ax.axis("off")
    fig.suptitle("Predictions (green=correct, red=wrong)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_confusion(targets, preds, class_names, path=f"{OUT_DIR}/real_03_confusion.png"):
    n = len(class_names); cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(targets.tolist(), preds.tolist()): cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (val)")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                c = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", color=c)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_per_class(targets, preds, class_names,
                   path=f"{OUT_DIR}/real_04_per_class_acc.png"):
    n = len(class_names); accs = []
    for c in range(n):
        m = targets == c
        accs.append(0.0 if m.sum() == 0 else (preds[m] == c).float().mean().item())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(n), accs, color="steelblue")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy on class")
    ax.set_title("Per-class accuracy (val)")
    for i, a in enumerate(accs): ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_curves(tl, va, path=f"{OUT_DIR}/real_05_loss_curve.png"):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss (CE)", color="C0")
    ax1.plot(tl, color="C0", marker="o"); ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Val accuracy", color="C1")
    ax2.plot(va, color="C1", marker="s"); ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(0, 1)
    fig.suptitle("Training progress"); fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(0)

    real_count = count_images(DATA_DIR)

    if FORCE_SAMPLE or real_count == 0:
        ensure_sample_dataset(SAMPLE_DIR)
        if not FORCE_SAMPLE:
            print("\n" + "=" * 70)
            print(f"No real images found in '{DATA_DIR}/'.")
            print("Falling back to the bundled SAMPLE dataset so you can run it now.")
            print(f"\nLook at the sample first:   xdg-open {SAMPLE_DIR}/")
            print("Then collect real photos:  see phase1/DATA.md")
            print("=" * 70 + "\n")
        active_dir = SAMPLE_DIR
    else:
        print(f"Found {real_count} real images in '{DATA_DIR}/' — using them.")
        active_dir = DATA_DIR

    print(f"\nLoading images from: {active_dir}")
    train_full_aug = PhotoFolderDataset(active_dir, augment=True)
    val_full_clean = PhotoFolderDataset(active_dir, augment=False)
    class_names = train_full_aug.class_names
    n_classes = len(class_names)
    print(f"\nTotal images: {len(train_full_aug)} | Classes: {class_names}")

    rng = np.random.default_rng(0)
    all_idx = np.arange(len(train_full_aug)); rng.shuffle(all_idx)
    n_val = max(1, int(len(all_idx) * VAL_SPLIT))
    val_idx = all_idx[:n_val].tolist()
    train_idx = all_idx[n_val:].tolist()
    train_ds = torch.utils.data.Subset(train_full_aug, train_idx)
    val_ds   = torch.utils.data.Subset(val_full_clean, val_idx)

    train_dl = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=2)

    model = MLP(out_dim=n_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("\n[1/5] Visualizing samples...")
    plot_samples(train_ds, class_names)

    print("\n[2/5] Training...")
    train_losses, val_accs = [], []
    for epoch in range(N_EPOCHS):
        tl = train_epoch(model, train_dl, opt, loss_fn)
        va = evaluate(model, val_dl)
        train_losses.append(tl); val_accs.append(va)
        print(f"  epoch {epoch:2d}  train_loss={tl:.4f}  val_acc={va:.4f}")

    print("\n[3/5] Final evaluation...")
    acc, preds, targets = evaluate(model, val_dl, return_preds=True)
    print(f"  final val_acc = {acc:.4f}")

    print("\n[4/5] Diagnostics...")
    plot_predictions(val_ds, model, class_names)
    plot_confusion(targets, preds, class_names)
    plot_per_class(targets, preds, class_names)

    print("\n[5/5] Curves...")
    plot_curves(train_losses, val_accs)

    print(f"\nAll plots saved to ./{OUT_DIR}/  (prefix real_*)")