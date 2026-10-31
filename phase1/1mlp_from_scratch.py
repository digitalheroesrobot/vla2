# =============================================================================
# phase1/mlp_from_scratch.py  —  OFFLINE version
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. Complete training loop: forward → loss → backward → step.
#   2. CrossEntropyLoss = softmax + NLL (raw logits in, class idx out).
#   3. Train/eval mode: dropout ON in train, OFF in eval — deterministic
#      inference is mandatory for robot policies.
#   4. Why zero_grad() is mandatory (PyTorch accumulates grads by default).
#   5. Overfitting: train loss falls, val accuracy plateaus.
#   6. HOW TO VISUALLY DEBUG A MODEL: 4-plot pattern you'll reuse
#      in Phase 4 (CLIP) and Phase 5 (LLaVA).
#
# WHY SYNTHETIC DATA
#   Phase 1 is about the LOOP, not the images. Synthetic 64×64 shapes let
#   you see the loop work in <30 sec with ZERO downloads, ZERO storage,
#   ZERO setup. Real images come at Phase 3 (ViT) and Phase 4 (CLIP).
#
# WHERE IT SHIPS IN THE REAL WORLD
#   - Every VLA behavior-cloning script is this file with images swapped
#     for camera frames and labels swapped for 7-dim joint actions.
#   - The 4-plot pattern (samples / pred-vs-truth / confusion / loss) is the
#     standard first-pass visualization used at every ML lab. If you don't
#     produce these 4 plots in the first hour of training, you're flying blind.
#   - The DATA_MODE switch is how production teams A/B the pipeline: use
#     synthetic in CI, real data in production — same code path.
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")              # headless — works over SSH
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_MODE   = "synthetic"          # "synthetic" or "folder"
DATA_DIR    = "data/tiny_classify" # only used if DATA_MODE == "folder"
CLASS_NAMES = ["circle", "square", "triangle"]
IMG_SIZE    = 64
N_CLASSES   = len(CLASS_NAMES)

# -----------------------------------------------------------------------------
# Dataset 1 — synthetic shapes (default, offline, zero storage)
# -----------------------------------------------------------------------------
def _draw_shape(img, shape_id, rng):
    """Paint one random shape into a 64x64 float image in [0,1]."""
    h, w = img.shape[:2]
    cx = rng.integers(16, w - 16)
    cy = rng.integers(16, h - 16)
    r  = rng.integers(8, 16)
    color = rng.uniform(0.2, 1.0, size=3)      # random RGB

    yy, xx = np.mgrid[0:h, 0:w]
    if shape_id == 0:                          # circle
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    elif shape_id == 1:                        # square
        mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    else:                                      # triangle
        # barycentric-style: point is inside if below the three edges
        mask = ((yy - (cy - r)) >= 0) & \
               (np.abs(xx - cx) <= (yy - (cy - r)) * 0.9)
    img[mask] = color
    # Per-image noise: forces the model to learn shape, not a fixed filter.
    img += rng.normal(0, 0.05, size=img.shape)
    return img


class SyntheticShapeDataset(Dataset):
    """3-class synthetic shape images. Deterministic per index so that
    train and val sets are reproducible across runs.

    FIXED seed per split means: same image every epoch. The model can
    still overfit (which we want to detect), but we can compare runs fairly.
    """
    def __init__(self, n=600, seed=0):
        self.n = n
        rng = np.random.default_rng(seed)
        self.imgs   = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        self.labels = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            label = i % N_CLASSES
            self.labels[i] = label
            self.imgs[i] = _draw_shape(np.zeros_like(self.imgs[i]), label, rng)

    def __len__(self): return self.n

    def __getitem__(self, i):
        # (H, W, C) -> (C, H, W), float32
        img = torch.from_numpy(self.imgs[i]).permute(2, 0, 1).float()
        return img, int(self.labels[i])


# -----------------------------------------------------------------------------
# Dataset 2 — folder loader (for later, when you have real images)
# -----------------------------------------------------------------------------
class FolderImageDataset(Dataset):
    """Loads images from data/tiny_classify/<class>/*.png.
    Uses PIL — no torchvision needed. To use, download PIL (`pip install pillow`)
    and set DATA_MODE = "folder"."""
    def __init__(self, root, size=IMG_SIZE):
        from PIL import Image
        self.samples = []
        self.class_to_idx = {}
        for idx, cls in enumerate(sorted(os.listdir(root))):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue
            self.class_to_idx[cls] = idx
            for fn in sorted(os.listdir(cls_dir)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fn), idx))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}. See DATA.md.")
        self.size = size

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1), label


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """Three linear layers + dropout. Same shape as a VLA action head."""
    def __init__(self, in_dim=3 * IMG_SIZE * IMG_SIZE, hidden=256, out_dim=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x.flatten(1))


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()                # MUST be before backward
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, return_preds=False):
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        if return_preds:
            all_preds.append(preds.cpu()); all_targets.append(y.cpu())
    acc = correct / total
    if return_preds:
        return acc, torch.cat(all_preds), torch.cat(all_targets)
    return acc


# -----------------------------------------------------------------------------
# Visualizations — the 4-plot pattern
# -----------------------------------------------------------------------------
def plot_samples(dataset, n=16, path=f"{OUT_DIR}/01_samples.png"):
    """'Look at your data before your model.' First thing to do on any
    dataset. In production, this catches label errors, corruption, and
    class imbalance before you spend GPU hours training."""
    idx = np.random.RandomState(0).choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, i in zip(axes.flat, idx):
        img, label = dataset[i]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(CLASS_NAMES[label], fontsize=9); ax.axis("off")
    fig.suptitle("Dataset samples (ground truth)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_predictions(dataset, model, n=16, path=f"{OUT_DIR}/02_predictions.png"):
    """Green=correct, Red=wrong. Fastest way to spot failure modes.
    If all errors are 'circle→square', that's a shape-geometry problem.
    If all errors are 'square→triangle', maybe rotation augment would help.
    In Phase 5 you'll make the same plot with 'true: mug / pred: bottle'."""
    model.eval()
    idx = np.random.RandomState(1).choice(len(dataset), n, replace=False)
    imgs = torch.stack([dataset[i][0] for i in idx]).to(device)
    with torch.no_grad():
        preds = model(imgs).argmax(1).cpu().tolist()
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, i, p in zip(axes.flat, idx, preds):
        img, label = dataset[i]
        ax.imshow(img.permute(1, 2, 0).numpy())
        color = "green" if p == label else "red"
        ax.set_title(f"true: {CLASS_NAMES[label]}\npred: {CLASS_NAMES[p]}",
                     fontsize=8, color=color); ax.axis("off")
    fig.suptitle("Predictions (green=correct, red=wrong)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_confusion_matrix(targets, preds, path=f"{OUT_DIR}/03_confusion_matrix.png"):
    """Row=true, col=predicted. Diagonal=correct. Off-diagonal=structured errors.
    THIS is how you decide what data to collect next. In a VLA, a row of
    dark off-diagonal cells = 'the robot confuses these two objects'."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(targets.tolist(), preds.tolist()): cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (val)")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j]:
                c = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", color=c)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_curves(train_losses, val_accs, path=f"{OUT_DIR}/04_loss_curve.png"):
    """Reading this plot:
       ✅ both trend right → healthy
       ⚠️ loss↓, acc flat → overfitting
       ❌ loss flat → LR too low or loss bug
       ❌ loss NaN → LR too high (try lr=1.0 to see)"""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss (CE)", color="C0")
    ax1.plot(train_losses, color="C0", marker="o"); ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Val accuracy", color="C1")
    ax2.plot(val_accs, color="C1", marker="s"); ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(0, 1)
    fig.suptitle("Training progress"); fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(0)

    # Load the dataset according to DATA_MODE
    if DATA_MODE == "synthetic":
        print("Mode: synthetic (no files needed)")
        train_ds = SyntheticShapeDataset(n=900, seed=0)
        val_ds   = SyntheticShapeDataset(n=300, seed=1)
    elif DATA_MODE == "folder":
        print(f"Mode: folder ({DATA_DIR})")
        full = FolderImageDataset(DATA_DIR)
        n = len(full); k = int(0.8 * n)
        train_ds, val_ds = torch.utils.data.random_split(full, [k, n - k])
    else:
        raise ValueError(f"Unknown DATA_MODE: {DATA_MODE}")

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    model = MLP().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nParam count: {sum(p.numel() for p in model.parameters()):,}")

    # 1) look at data first
    print("\n[1/4] Visualizing samples...")
    plot_samples(train_ds if DATA_MODE == "synthetic" else train_ds.dataset,
                 n=16)

    # 2) train
    print("\n[2/4] Training...")
    epochs = 10
    train_losses, val_accs = [], []
    for epoch in range(epochs):
        tl = train_epoch(model, train_dl, opt, loss_fn)
        va = evaluate(model, val_dl)
        train_losses.append(tl); val_accs.append(va)
        print(f"  epoch {epoch:2d}  train_loss={tl:.4f}  val_acc={va:.4f}")

    # 3) final eval
    print("\n[3/4] Final evaluation...")
    acc, preds, targets = evaluate(model, val_dl, return_preds=True)
    print(f"  final val_acc = {acc:.4f}")

    # 4) visualize results
    print("\n[4/4] Rendering outputs...")
    plot_predictions(val_ds if DATA_MODE == "synthetic" else val_ds.dataset,
                     model, n=16)
    plot_confusion_matrix(targets, preds)
    plot_curves(train_losses, val_accs)

    print(f"\nAll plots saved to ./{OUT_DIR}/")
    print("Open them:  xdg-open outputs/   (Linux)  |  open outputs/   (Mac)")