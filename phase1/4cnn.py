# =============================================================================
# phase1/cnn_from_scratch.py
# =============================================================================
"""
AFTER THIS STEP YOU WILL...
--------------------------
  1. UNDERSTAND why convolution exists — and why it beats an MLP on images.
     An MLP flattens the image and loses spatial structure; a conv layer
     slides small filters over the image and PRESERVES locality.
  2. SEE feature maps as images. Each conv filter produces an activation
     map: a picture of "where in the image did this filter fire?". After
     training, filters detect edges, colors, corners — the visual alphabet
     of every vision model (ViT, CLIP, LLaVA, OpenVLA all start here).
  3. COMPARE an MLP vs a CNN on the SAME dataset. The CNN should hit higher
     val accuracy with FAR fewer parameters. That gap is why the entire
     field uses convolutions (or ViT patches, which are basically the same).
  4. EXPLAIN pooling, stride, receptive field, and why CNNs are
     translation-equivariant, in your own words.
  5. HAVE a working image classifier you can point at your own photo folder.

WHAT YOU ARE DOING (one line)
-----------------------------
  Training a small CNN on your local image folder (or the bundled sample),
  then saving four visuals: filter kernels, feature maps, predictions,
  and the MLP-vs-CNN accuracy comparison.

WHAT YOU WILL GET ON DISK
-------------------------
  outputs/cnn_01_filters.png        — the 32 learned 3x3 filters (layer 1)
  outputs/cnn_02_featuremaps.png    — feature maps for one input image,
                                       at every conv layer
  outputs/cnn_03_predictions.png    — 16 val images, green=correct red=wrong
  outputs/cnn_04_mlp_vs_cnn.png     — bar chart comparing the two models

DATASET CONTRACT
----------------
  Uses data/tiny_classify/ (your real photos) if present.
  Falls back to data/tiny_classify_sample/ (24 auto-generated PNGs).
  No downloads. See phase1/DATA.md.

SANITY CHECK (what "success" means here)
----------------------------------------
  - The CNN's val_acc should be >= the MLP's val_acc on the SAME data.
    On the sample dataset (3 shapes, very separable), expect both near 1.0.
    On real photos, the CNN should clearly win, often by 10-20%.
  - Feature maps in cnn_02 show STRUCTURE: some maps highlight edges,
    some highlight colors, some are mostly black (dead filters — this is
    normal at small scale, but you should see at least a few working).
  - Training loss decreases monotonically; val_acc climbs to a plateau.

WHAT FAILURE LOOKS LIKE
-----------------------
  - CNN val_acc < MLP val_acc    → training loop bug or wrong reshape
  - Feature maps all identical   → weights didn't update (check opt.step)
  - Feature maps all uniform gray → ReLU killed everything (LR too high)
  - FileNotFoundError on save    → outputs/ missing (this file creates it)
"""
# =============================================================================
# WHAT THIS FILE TEACHES (technical)
#   1. nn.Conv2d: 32 filters of shape (3, 3, 3) sliding over a (3, 64, 64)
#      image with stride 1 + padding 1 → output (32, 64, 64). Each output
#      channel is "where did this filter fire".
#   2. Pooling: (2,2) max-pool halves H and W. Purpose: (a) shrink compute,
#      (b) give small shifts the same answer → translation robustness.
#   3. Receptive field: after two conv+pool blocks, a single neuron sees
#      ~14x14 input pixels. After four, ~28x28 — half the image. This is
#      how CNNs grow "context" without ever seeing the full image at once.
#   4. Why CNN > MLP on images: an MLP has to relearn "how to detect an
#      edge" for every (row, col) position independently. A conv layer
#      learns ONE edge detector and applies it everywhere. Same idea as
#      a transformer applying one attention pattern to every token.
#
# WHERE IT SHIPS IN THE REAL WORLD
#   - CNN backbones (ResNet, EfficientNet) still serve in production for
#     edge devices, phones, and real-time robotics where ViT latency is
#     too high.
#   - Every VLM's vision encoder is either a CNN (older VLMs like Flamingo's
#     vision stack) or a ViT (newer: CLIP, SigLIP, LLaVA). Both are built
#     from the same conv/attention primitives you'll see here.
#   - Feature maps are the standard debugging tool in vision teams. When a
#     VLM "misses" an object, engineers visualize encoder feature maps to
#     see if the object ever produced strong activations.
#   - Robotics: feature maps from the vision encoder feed the policy — a
#     small CNN at 30 Hz is often preferred over ViT for real-time control.
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
DATA_DIR     = "data/tiny_classify"
SAMPLE_DIR   = "data/tiny_classify_sample"
OUT_DIR      = "outputs"
IMG_SIZE     = 64
BATCH        = 32
N_EPOCHS     = 15
LR           = 1e-3
VAL_SPLIT    = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Sample dataset generator (same as phase1/mlp_on_your_photos.py)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class PhotoFolderDataset(Dataset):
    def __init__(self, root, size=IMG_SIZE, augment=False):
        self.size = size
        self.augment = augment
        self.samples = []
        self.class_names = []
        for idx, cls in enumerate(sorted(os.listdir(root))):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                files.extend(glob.glob(os.path.join(cls_dir, ext)))
            files = sorted(files)
            if not files: continue
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
# Models
# -----------------------------------------------------------------------------
class MLPBaseline(nn.Module):
    """Same structure as Step 1. Used ONLY as a comparison baseline."""
    def __init__(self, in_dim=3 * IMG_SIZE * IMG_SIZE, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x.flatten(1))


class SmallCNN(nn.Module):
    """
    Two conv blocks + a classifier head.

    Block 1: Conv(3→32, 3x3) → ReLU → Conv(32→32, 3x3) → ReLU → MaxPool(2)
             (64x64 → 32x32)
    Block 2: Conv(32→64, 3x3) → ReLU → Conv(64→64, 3x3) → ReLU → MaxPool(2)
             (32x32 → 16x16)
    Head   : GlobalAvgPool → Linear(64→n_classes)

    Total params: ~50k (vs ~270k for the MLP). Smaller model, higher accuracy
    on images — that's the CNN advantage.
    """
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                # 32 -> 16
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x, return_features=False):
        feats = self.features(x)                            # (B, 64, 16, 16)
        pooled = feats.mean(dim=(2, 3))                     # global avg pool
        logits = self.head(pooled)
        if return_features:
            return logits, feats
        return logits


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    running = 0.0
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
    model.eval()
    correct, total = 0, 0
    ap, at = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        if return_preds: ap.append(p.cpu()); at.append(y.cpu())
    acc = correct / total
    if return_preds: return acc, torch.cat(ap), torch.cat(at)
    return acc


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def _un(t): return (t * 0.5 + 0.5).clamp(0, 1)


def plot_filters(model, path=f"{OUT_DIR}/cnn_01_filters.png"):
    """
    Draw every filter (kernel) of the first conv layer as a small RGB image.
    Layer-1 filters of a trained CNN look like:
      - edge detectors (vertical, horizontal, diagonal)
      - color blobs
      - simple gradients
    This is the visual alphabet every vision model learns first.
    """
    W = model.features[0].weight.detach().cpu()             # (32, 3, 3, 3)
    n = W.shape[0]; cols = 8; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n: continue
        # normalize to [0,1] for display
        k = W[i]
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)
        ax.imshow(k.permute(1, 2, 0).numpy())
        ax.set_title(f"f{i}", fontsize=7)
    fig.suptitle("First-layer filters (each is a 3×3 RGB kernel)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_feature_maps(model, image_tensor, path=f"{OUT_DIR}/cnn_02_featuremaps.png"):
    """
    Feed one image through the conv stack, and draw the activation maps.

    Row 1: the input image, upscaled.
    Row 2: 8 of the 32 output channels after conv block 1.
    Row 3: 8 of the 64 output channels after conv block 2.

    Bright pixel = 'this filter fired strongly here'.
    Dark pixel   = 'this filter found nothing here'.
    """
    model.eval()
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)
        # manual forward with two snapshots
        h = x
        snapshots = []
        for layer in model.features:
            h = layer(h)
            snapshots.append(h)
        # snapshots[1]  = after conv1 (32, 64, 64)
        # snapshots[-1] = after conv2 block + pool (64, 16, 16)
        fm1 = snapshots[1][0].cpu()          # (32, 64, 64)
        fm2 = snapshots[-1][0].cpu()         # (64, 16, 16)

    def top_k_channels(fm, k=8):
        # pick the k channels with highest mean activation — these are the
        # filters that "found something" in this image. Others may be dead.
        means = fm.mean(dim=(1, 2))
        idx = means.argsort(descending=True)[:k].tolist()
        return [(i, fm[i]) for i in idx]

    fig, axes = plt.subplots(3, 8, figsize=(14, 6))
    # Row 1: input (repeat across 8 cols)
    inp = _un(image_tensor).permute(1, 2, 0).numpy()
    for j in range(8):
        axes[0, j].imshow(inp); axes[0, j].axis("off")
    axes[0, 0].set_title("input", fontsize=8)
    # Row 2: top-8 channels of layer 1
    for j, (ch, m) in enumerate(top_k_channels(fm1)):
        axes[1, j].imshow(m.numpy(), cmap="viridis")
        axes[1, j].axis("off")
        axes[1, j].set_title(f"L1 c{ch}", fontsize=7)
    # Row 3: top-8 channels of layer 2
    for j, (ch, m) in enumerate(top_k_channels(fm2)):
        axes[2, j].imshow(m.numpy(), cmap="viridis")
        axes[2, j].axis("off")
        axes[2, j].set_title(f"L2 c{ch}", fontsize=7)
    fig.suptitle("Feature maps: how the CNN 'sees' one image", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_predictions(dataset, model, class_names, n=16,
                     path=f"{OUT_DIR}/cnn_03_predictions.png"):
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
    fig.suptitle("CNN predictions  (green=correct, red=wrong)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def plot_mlp_vs_cnn(mlp_acc, cnn_acc, mlp_params, cnn_params,
                    path=f"{OUT_DIR}/cnn_04_mlp_vs_cnn.png"):
    """
    Bar chart: accuracy (left axis) and parameter count (right axis).
    The pattern to look for:
      - CNN has FEWER params AND HIGHER accuracy → CNN wins on images
      - MLP has MORE params AND LOWER accuracy    → flattens spatial structure
    This is the empirical argument for convolution (and, later, ViT).
    """
    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(2); w = 0.35
    ax1.bar(x - w/2, [mlp_acc, cnn_acc], w, label="val accuracy",
            color=["#d97777", "#77d997"])
    ax1.set_ylabel("Validation accuracy")
    ax1.set_xticks(x); ax1.set_xticklabels(["MLP", "CNN"])
    ax1.set_ylim(0, 1.05)
    for i, a in enumerate([mlp_acc, cnn_acc]):
        ax1.text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.bar(x + w/2, [mlp_params, cnn_params], w, label="param count",
            color=["#b0b0b0", "#707070"], alpha=0.7)
    ax2.set_ylabel("Parameter count")
    for i, p in enumerate([mlp_params, cnn_params]):
        ax2.text(i + w/2, p + max(mlp_params, cnn_params) * 0.02,
                 f"{p/1000:.0f}k", ha="center", fontsize=8)

    fig.suptitle("MLP vs CNN on the same dataset")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(0)

    # Data source selection
    real_count = count_images(DATA_DIR)
    if real_count == 0:
        ensure_sample_dataset(SAMPLE_DIR)
        print(f"\nNo images in {DATA_DIR}/ — using sample dataset {SAMPLE_DIR}/")
        active_dir = SAMPLE_DIR
    else:
        print(f"\nFound {real_count} images in {DATA_DIR}/ — using them.")
        active_dir = DATA_DIR

    # Two datasets from the same folder: one augmented for training, one clean
    # for evaluation. SAME split indices → no leakage.
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
    val_ds_clean = torch.utils.data.Subset(val_full_clean, val_idx)

    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds_clean, BATCH, shuffle=False, num_workers=2)

    # ---- 1. Train MLP baseline -------------------------------------------
    print("\n[1/5] Training MLP baseline (for comparison)")
    mlp = MLPBaseline(out_dim=n_classes).to(device)
    mlp_opt = optim.AdamW(mlp.parameters(), lr=LR, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(N_EPOCHS):
        tl = train_epoch(mlp, train_dl, mlp_opt, loss_fn)
        if (epoch + 1) % 5 == 0:
            va = evaluate(mlp, val_dl)
            print(f"  MLP  epoch {epoch+1:2d}  loss={tl:.4f}  val_acc={va:.4f}")
    mlp_acc = evaluate(mlp, val_dl)
    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"  MLP final: val_acc={mlp_acc:.4f}, params={mlp_params:,}")

    # ---- 2. Train CNN ----------------------------------------------------
    print("\n[2/5] Training CNN")
    cnn = SmallCNN(n_classes=n_classes).to(device)
    cnn_opt = optim.AdamW(cnn.parameters(), lr=LR, weight_decay=1e-3)
    for epoch in range(N_EPOCHS):
        tl = train_epoch(cnn, train_dl, cnn_opt, loss_fn)
        if (epoch + 1) % 5 == 0:
            va = evaluate(cnn, val_dl)
            print(f"  CNN  epoch {epoch+1:2d}  loss={tl:.4f}  val_acc={va:.4f}")
    cnn_acc = evaluate(cnn, val_dl)
    cnn_params = sum(p.numel() for p in cnn.parameters())
    print(f"  CNN final: val_acc={cnn_acc:.4f}, params={cnn_params:,}")

    # ---- 3. Feature maps on one validation image -------------------------
    print("\n[3/5] Visualizing features...")
    plot_filters(cnn)
    val_img, _ = val_ds_clean[0]
    plot_feature_maps(cnn, val_img)

    # ---- 4. Predictions + comparison plot --------------------------------
    print("\n[4/5] Predictions and comparison...")
    plot_predictions(val_full_clean, cnn, class_names)
    plot_mlp_vs_cnn(mlp_acc, cnn_acc, mlp_params, cnn_params)

    # ---- 5. Final summary ------------------------------------------------
    print("\n[5/5] Summary")
    print(f"  MLP: acc={mlp_acc:.4f}  params={mlp_params:,}")
    print(f"  CNN: acc={cnn_acc:.4f}  params={cnn_params:,}")
    print(f"  CNN advantage: +{(cnn_acc-mlp_acc)*100:+.1f}% accuracy, "
          f"{(1 - cnn_params/mlp_params)*100:+.1f}% fewer params")
    print(f"\nAll plots saved to ./{OUT_DIR}/cnn_*.png")