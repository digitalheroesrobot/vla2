# =============================================================================
# phase5/vlm_mini.py
# =============================================================================
"""
Production-grade tiny VLM QA system.

Pipeline:
    image    ──► [Vision Encoder] ──► v ──┐
                                          ├──► [Fusion] ──► [Head] ──► answer
    question ──► [Lang Encoder]  ──► q ──┘

What is upgraded vs. the teaching version:
  1. Config is a dataclass — reproducible, loggable, swappable.
  2. Image encoder uses Conv+BN+ReLU+residual blocks (optionally ResNet18).
  3. Question encoder supports GRU or Transformer, with attention pooling.
  4. Fusion uses FiLM + cross-attention (BLIP/Flamingo-style) or plain concat.
  5. Training uses cosine LR + warmup, grad clipping, label smoothing, AMP,
     and early stopping.
  6. Eval reports accuracy PER QUESTION TYPE plus a confusion matrix.
  7. Checkpoint is self-contained: config + tokenizer + weights + metrics.
  8. Dataset renders multiple objects / distractors and paraphrased questions.

Run:
    python vlm_mini.py
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# =============================================================================
# 0. CONFIG
# =============================================================================
@dataclass
class Config:
    out_dir: str = "outputs"
    img_size: int = 32
    n_scenes: int = 2000
    batch: int = 128
    epochs: int = 20
    lr: float = 2e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    max_q_len: int = 10
    pad_id: int = 0
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    warmup_frac: float = 0.1
    patience: int = 5
    seed: int = 42
    question_encoder: Literal["gru", "transformer"] = "gru"
    fusion: Literal["concat", "film", "film_xattn"] = "film_xattn"
    vision_backbone: Literal["mini", "resnet18"] = "mini"
    use_amp: bool = True
    num_workers: int = 0
    device: str = field(default="", init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 1. DATA
# =============================================================================
SHAPES = ["circle", "square", "triangle"]
COLORS = ["red", "green", "blue", "yellow", "purple"]
COLOR_RGB = {
    "red": (0.95, 0.12, 0.12),
    "green": (0.12, 0.90, 0.20),
    "blue": (0.12, 0.25, 0.98),
    "yellow": (1.00, 0.88, 0.08),
    "purple": (0.72, 0.15, 0.90),
}

# Multiple paraphrases per question type → model can't memorize the string.
QUESTION_TEMPLATES: dict[str, list[str]] = {
    "shape": ["What shape is shown?", "Which shape do you see?", "Name the shape."],
    "color": ["What color is the object?", "Which color do you see?", "Name the color."],
    "yesno": ["Is this a circle?", "Do you see a square?", "Is the shape a triangle?"],
}
ALL_QUESTIONS = [q for qs in QUESTION_TEMPLATES.values() for q in qs]

ANSWERS = SHAPES + COLORS + ["yes", "no"]
ANSWER_TO_ID = {a: i for i, a in enumerate(ANSWERS)}
ID_TO_ANSWER = {i: a for a, i in ANSWER_TO_ID.items()}
TEMPLATE_TO_TYPE = {q: t for t, qs in QUESTION_TEMPLATES.items() for q in qs}


class QuestionTokenizer:
    """Whitespace tokenizer with a fixed, serializable vocabulary."""

    def __init__(self, questions: list[str], max_len: int, pad_id: int = 0):
        words = sorted({w.lower().strip("?!.,") for q in questions for w in q.split()})
        self.word_to_id = {"<pad>": pad_id, "<unk>": 1}
        self.word_to_id.update({w: i + 2 for i, w in enumerate(words)})
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.word_to_id)

    def encode(self, question: str) -> torch.Tensor:
        toks = [w.lower().strip("?!.,") for w in question.split()][: self.max_len]
        ids = [self.word_to_id.get(t, self.word_to_id["<unk>"]) for t in toks]
        ids += [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def state_dict(self) -> dict:
        return {"word_to_id": self.word_to_id, "max_len": self.max_len, "pad_id": self.pad_id}

    @classmethod
    def from_state(cls, s: dict) -> "QuestionTokenizer":
        obj = cls.__new__(cls)
        obj.word_to_id = s["word_to_id"]
        obj.id_to_word = {i: w for w, i in obj.word_to_id.items()}
        obj.max_len = s["max_len"]
        obj.pad_id = s["pad_id"]
        return obj


def render_scene(shape: str, color: str, rng: np.random.Generator,
                 img_size: int, n_distractors: int = 2) -> torch.Tensor:
    """Render a target shape plus small colored distractor blobs."""
    image = np.zeros((img_size, img_size, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:img_size, 0:img_size]

    # Distractors first (so the target can overwrite them if overlapping).
    for _ in range(n_distractors):
        d_color = COLOR_RGB[rng.choice(COLORS)]
        cx, cy = rng.integers(4, img_size - 4, size=2)
        r = int(rng.integers(2, 4))
        image[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = d_color

    # Target shape.
    cx, cy = rng.integers(9, img_size - 9, size=2)
    radius = int(rng.integers(5, 8))
    if shape == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    elif shape == "square":
        mask = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
    else:  # triangle
        h = yy - (cy - radius)
        mask = (h >= 0) & (np.abs(xx - cx) <= h * 0.9)
    image[mask] = COLOR_RGB[color]

    image += rng.normal(0, 0.03, image.shape)
    image = np.clip(image, 0, 1)
    image = (image - 0.5) / 0.5
    return torch.from_numpy(image).permute(2, 0, 1).float()


class SyntheticQADataset(Dataset):
    """One sample per (scene, question template)."""

    def __init__(self, tokenizer: QuestionTokenizer, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.tokenizer = tokenizer
        rng = np.random.default_rng(seed)
        self.images: list[torch.Tensor] = []
        self.scenes: list[tuple[str, str]] = []
        self.samples: list[tuple[int, str, int]] = []

        for scene_id in range(cfg.n_scenes):
            shape = SHAPES[scene_id % len(SHAPES)]
            color = COLORS[(scene_id // len(SHAPES)) % len(COLORS)]
            self.images.append(render_scene(shape, color, rng, cfg.img_size))
            self.scenes.append((shape, color))

            for qtype, templates in QUESTION_TEMPLATES.items():
                for qtext in templates:
                    if qtype == "yesno":
                        asked = rng.choice(SHAPES)
                        ans = "yes" if asked == shape else "no"
                    elif qtype == "shape":
                        ans = shape
                    else:  # color
                        ans = color
                    self.samples.append((scene_id, qtext, ANSWER_TO_ID[ans]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        scene_id, qtext, ans_id = self.samples[i]
        return {
            "image": self.images[scene_id],
            "question": self.tokenizer.encode(qtext),
            "answer": torch.tensor(ans_id, dtype=torch.long),
            "question_text": qtext,
            "question_type": TEMPLATE_TO_TYPE[qtext],
        }


def make_collate(pad_id: int):
    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        questions = torch.stack([b["question"] for b in batch])
        answers = torch.stack([b["answer"] for b in batch])
        return {
            "images": images,
            "questions": questions,
            "answers": answers,
            "question_texts": [b["question_text"] for b in batch],
            "question_types": [b["question_type"] for b in batch],
            "mask": questions != pad_id,
        }
    return collate


# =============================================================================
# 2. MODEL
# =============================================================================
class ConvBlock(nn.Module):
    """Conv+BN+ReLU with residual shortcut."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = (
            nn.Identity() if (in_c == out_c and stride == 1)
            else nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                               nn.BatchNorm2d(out_c))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class MiniVisionEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetVisionEncoder(nn.Module):
    """Optional pretrained backbone (recommended for real usage)."""

    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(m.children())[:-2])  # (B, 512, H, W)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.proj(self.pool(f).flatten(1))


class QuestionEncoder(nn.Module):
    def __init__(self, vocab: int, emb: int = 64, hidden: int = 128,
                 kind: Literal["gru", "transformer"] = "gru", pad_id: int = 0):
        super().__init__()
        self.kind = kind
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
        if kind == "gru":
            self.rnn = nn.GRU(emb, hidden, batch_first=True)
            self.token_dim = hidden
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=emb, nhead=4, dim_feedforward=emb * 4,
                batch_first=True, dropout=0.1,
            )
            self.rnn = nn.TransformerEncoder(layer, num_layers=2)
            self.token_dim = emb
        self.attn_pool = nn.Linear(self.token_dim, 1)
        self.out_dim = self.token_dim

    def forward(self, ids: torch.Tensor, mask: torch.Tensor,
                return_tokens: bool = False):
        x = self.emb(ids)
        if self.kind == "gru":
            lengths = mask.sum(1).clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True,
                                          enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(out, batch_first=True,
                                         total_length=ids.size(1))  # (B, T, H)
        else:
            out = self.rnn(x, src_key_padding_mask=~mask)

        scores = self.attn_pool(out).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = (out * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        if return_tokens:
            return pooled, out
        return pooled, weights


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: language → (γ, β) applied to vision."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, vision: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(lang).chunk(2, dim=-1)
        return vision * (1 + gamma) + beta


class CrossAttention(nn.Module):
    """Vision vector attends over language tokens (BLIP/Flamingo-style)."""

    def __init__(self, dim: int, nheads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, vis_query: torch.Tensor, lang_tokens: torch.Tensor,
                lang_mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(
            query=vis_query.unsqueeze(1),
            key=lang_tokens,
            value=lang_tokens,
            key_padding_mask=~lang_mask,
            need_weights=False,
        )
        return self.norm(vis_query + out.squeeze(1))


class VLMQA(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config, n_answers: int = len(ANSWERS)):
        super().__init__()
        self.cfg = cfg

        # Vision.
        if cfg.vision_backbone == "resnet18":
            self.vision = ResNetVisionEncoder(out_dim=128)
        else:
            self.vision = MiniVisionEncoder(out_dim=128)
        vdim = self.vision.out_dim

        # Language.
        self.lang = QuestionEncoder(
            vocab=vocab_size, emb=64, hidden=128,
            kind=cfg.question_encoder, pad_id=cfg.pad_id,
        )
        ldim = self.lang.out_dim

        # Fusion.
        if cfg.fusion in ("film", "film_xattn"):
            self.film = FiLM(ldim, vdim)
        if cfg.fusion == "film_xattn":
            self.xattn = CrossAttention(vdim)

        fused_dim = vdim + ldim + vdim  # [v', q, v' * q]
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_answers),
        )

    def forward(self, images: torch.Tensor, questions: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = questions != self.cfg.pad_id

        v = self.vision(images)                             # (B, Dv)
        pooled, tok_feats = self.lang(                      # (B, Dl), (B, T, Dl)
            questions, mask, return_tokens=True,
        )

        if self.cfg.fusion in ("film", "film_xattn"):
            v = self.film(v, pooled)

        if self.cfg.fusion == "film_xattn":
            # Cross-attention needs d_query == d_key. Project tokens to Dv.
            if not hasattr(self, "lang_to_v"):
                self.lang_to_v = nn.Linear(self.lang.out_dim, self.vision.out_dim).to(v.device)
            v = self.xattn(v, self.lang_to_v(tok_feats), mask)

        fused = torch.cat([v, pooled, v * pooled], dim=-1)
        return self.head(fused)


# =============================================================================
# 3. TRAINING
# =============================================================================
def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * p))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             device: str):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    per_type: dict[str, list[int]] = {}
    conf = np.zeros((len(ANSWERS), len(ANSWERS)), dtype=int)

    for batch in loader:
        imgs = batch["images"].to(device, non_blocking=True)
        qs = batch["questions"].to(device, non_blocking=True)
        ans = batch["answers"].to(device, non_blocking=True)
        mask = batch["mask"].to(device)

        logits = model(imgs, qs, mask)
        loss_sum += loss_fn(logits, ans).item() * ans.numel()

        preds = logits.argmax(1)
        correct += (preds == ans).sum().item()
        total += ans.numel()

        for p, a, t in zip(preds.cpu().tolist(), ans.cpu().tolist(),
                           batch["question_types"]):
            conf[a, p] += 1
            per_type.setdefault(t, [0, 0])
            per_type[t][1] += 1
            per_type[t][0] += int(p == a)

    per_type_acc = {k: v[0] / max(1, v[1]) for k, v in per_type.items()}
    return loss_sum / total, correct / total, per_type_acc, conf


def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
          cfg: Config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    total_steps = max(1, cfg.epochs * len(train_dl))
    warmup = int(cfg.warmup_frac * total_steps)
    use_amp = cfg.use_amp and cfg.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val, bad_epochs, best_state = -1.0, 0, None
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "per_type": []}
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        t0 = time.time()

        for batch in train_dl:
            imgs = batch["images"].to(cfg.device, non_blocking=True)
            qs = batch["questions"].to(cfg.device, non_blocking=True)
            ans = batch["answers"].to(cfg.device, non_blocking=True)
            mask = batch["mask"].to(cfg.device)

            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * cosine_warmup(step, total_steps, warmup)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs, qs, mask)
                loss = loss_fn(logits, ans)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item() * ans.numel()
            ep_n += ans.numel()
            step += 1

        train_loss = ep_loss / ep_n
        val_loss, val_acc, per_type, _ = evaluate(model, val_dl, loss_fn, cfg.device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["per_type"].append(per_type)

        dt = time.time() - t0
        print(f"epoch {epoch + 1:2d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"({dt:.1f}s)  per_type={per_type}")

        if val_acc > best_val:
            best_val, bad_epochs = val_acc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# =============================================================================
# 4. VISUALIZATION
# =============================================================================
def plot_history(history: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ep = np.arange(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["val_acc"], color="C1")
    axes[1].set_title("Validation accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    types = list(history["per_type"][-1].keys())
    for t in types:
        axes[2].plot(ep, [pt[t] for pt in history["per_type"]], label=t)
    axes[2].set_title("Accuracy per question type")
    axes[2].legend()
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "vlm_history.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_examples(dataset: SyntheticQADataset, indices: list[int],
                  model: nn.Module, cfg: Config, out_dir: str, n: int = 6) -> None:
    model.eval()
    picks = indices[:n]
    cols = 2
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    axes = np.atleast_2d(axes)

    for ax, i in zip(axes.flat, picks):
        scene_id, qtext, ans_id = dataset.samples[i]
        img = dataset.images[scene_id]
        with torch.no_grad():
            logits = model(
                img.unsqueeze(0).to(cfg.device),
                dataset.tokenizer.encode(qtext).unsqueeze(0).to(cfg.device),
            )
        pred = logits.argmax(1).item()
        ax.imshow(((img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()))
        ax.set_title(qtext, fontsize=9)
        ax.set_xlabel(
            f"gt: {ID_TO_ANSWER[ans_id]} | pred: {ID_TO_ANSWER[pred]}",
            color="green" if pred == ans_id else "red",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(picks):]:
        ax.axis("off")

    fig.tight_layout()
    path = os.path.join(out_dir, "vlm_examples.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# =============================================================================
# 5. MAIN
# =============================================================================
def main(cfg: Config | None = None):
    cfg = cfg or Config()
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"Device: {cfg.device} | Config: {asdict(cfg)}")

    tokenizer = QuestionTokenizer(ALL_QUESTIONS, cfg.max_q_len, cfg.pad_id)
    dataset = SyntheticQADataset(tokenizer, cfg, seed=cfg.seed)

    # Scene-level split → no image leakage across train/val.
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(cfg.n_scenes)
    n_val = int(cfg.n_scenes * cfg.val_split)
    val_scenes = set(order[:n_val].tolist())
    train_idx = [i for i, (sid, _, _) in enumerate(dataset.samples)
                 if sid not in val_scenes]
    val_idx = [i for i, (sid, _, _) in enumerate(dataset.samples)
               if sid in val_scenes]

    collate = make_collate(cfg.pad_id)
    train_dl = DataLoader(
        Subset(dataset, train_idx), batch_size=cfg.batch, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    val_dl = DataLoader(
        Subset(dataset, val_idx), batch_size=cfg.batch, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    print(f"train batches={len(train_dl)}  val batches={len(val_dl)}")

    model = VLMQA(vocab_size=len(tokenizer), cfg=cfg).to(cfg.device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    model, history = train(model, train_dl, val_dl, cfg)

    loss_fn = nn.CrossEntropyLoss()
    _, acc, per_type, conf = evaluate(model, val_dl, loss_fn, cfg.device)
    print(f"\nFinal val_acc={acc:.3f}  per_type={per_type}")

    plot_history(history, cfg.out_dir)
    plot_examples(dataset, val_idx, model, cfg, cfg.out_dir)

    ckpt = {
        "config": asdict(cfg),
        "tokenizer": tokenizer.state_dict(),
        "answers": ANSWERS,
        "state_dict": model.state_dict(),
        "metrics": {"val_acc": acc, "per_type": per_type,
                    "confusion": conf.tolist()},
        "history": history,
    }
    path = os.path.join(cfg.out_dir, "vlm_checkpoint.pt")
    torch.save(ckpt, path)
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump({"val_acc": acc, "per_type": per_type}, f, indent=2)
    print(f"saved {path}")

    return model, tokenizer, cfg


@torch.no_grad()
def load_and_predict(checkpoint_path: str, image: torch.Tensor, question: str,
                     topk: int = 3):
    """Self-contained inference: load ckpt, run one QA."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k != "device"})
    tok = QuestionTokenizer.from_state(ckpt["tokenizer"])
    model = VLMQA(vocab_size=len(tok), cfg=cfg).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ids = tok.encode(question).unsqueeze(0).to(cfg.device)
    logits = model(image.unsqueeze(0).to(cfg.device), ids)
    probs = logits.softmax(-1)[0]
    top = probs.topk(topk)
    id2ans = {i: a for i, a in enumerate(ckpt["answers"])}
    return [(id2ans[i.item()], p.item()) for p, i in zip(top.values, top.indices)]


if __name__ == "__main__":
    main()