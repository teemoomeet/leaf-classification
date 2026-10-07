# -*- coding: utf-8 -*-
"""迁移学习：用 Flavia 学到的叶片特征，在 Folio 数据集上训练识别 32 种作物。

背景
----
Flavia 模型（32 种中国树木）对 Folio（32 种作物/热带植物）完全无法识别，
因为两者物种零重叠。本脚本用 MobileNetV3（ImageNet 预训练）做 backbone，
换一个 32 类的分类头，在 Folio 的 637 张白底叶片图上做迁移学习微调，
得到一个「认 Folio 作物」的新模型。

策略（与 leaves/finetune.py 一致）：
  * 冻结网络浅层（前 N 个 stage），只微调后几个 stage + 新分类头；
  * 白底对齐 + 随机旋转/抖色增强，小数据量下防止过拟合；
  * 分层划分 train/val/test，留出独立测试集评估泛化。

用法（在 D:\\leaves 下，用项目 venv）：
    .\\.venv\\Scripts\\python.exe leaves\\train_folio.py --folio "D:\\folio\\Folio Leaf Dataset\\Folio"

产物：models/flavia_ft_folio_mobilenetv3.pt（torch 模型）+ 划分文件 + 评估 CSV
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .dataset import IMAGE_SUFFIXES, Record, save_split
from .deep import IMAGENET_MEAN, IMAGENET_STD, _torch
from .segmentation import leaf_on_white, segment_leaf

DEFAULT_OUT = "flavia_ft_folio_mobilenetv3.pt"
DEFAULT_SIZE = 224

# Folio 32 物种（官方顺序），文件夹名 -> 标签
FOLIO_SPECIES: list[str] = [
    "Beaumier du perou", "Eggplant", "Fruitcitere", "Guava", "Hibiscus",
    "Betel", "Rose", "Chrysanthemum", "Ficus", "Duranta gold",
    "Ashanti blood", "Bitter Orange", "Coeur Demoiselle", "Jackfruit",
    "Mulberry Leaf", "Pimento", "Pomme Jacquot", "Star Apple",
    "Barbados Cherry", "Sweet Olive", "Croton", "Thevetia",
    "Vieux Garcon", "Chocolate tree", "Carricature plant", "Coffee",
    "Ketembilla", "Chinese guava", "Lychee", "Geranium",
    "Sweet potato", "Papaya",
]

_FOLDER_ALIASES: dict[str, str] = {
    "beaumier du perou": "Beaumier du perou", "eggplant": "Eggplant",
    "aubergine": "Eggplant", "fruitcitere": "Fruitcitere", "guava": "Guava",
    "hibiscus": "Hibiscus", "betel": "Betel", "rose": "Rose",
    "chrysanthemum": "Chrysanthemum", "ficus": "Ficus",
    "duranta gold": "Duranta gold", "ashanti blood": "Ashanti blood",
    "bitter orange": "Bitter Orange", "coeur demoiselle": "Coeur Demoiselle",
    "jackfruit": "Jackfruit", "mulberry leaf": "Mulberry Leaf",
    "mulberry": "Mulberry Leaf", "pimento": "Pimento",
    "pomme jacquot": "Pomme Jacquot", "star apple": "Star Apple",
    "barbados cherry": "Barbados Cherry", "sweet olive": "Sweet Olive",
    "croton": "Croton", "thevetia": "Thevetia",
    "vieux garcon": "Vieux Garcon", "chocolate tree": "Chocolate tree",
    "caricature plant": "Carricature plant",
    "carricature plant": "Carricature plant", "coffee": "Coffee",
    "ketembilla": "Ketembilla", "chinese guava": "Chinese guava",
    "lychee": "Lychee", "geranium": "Geranium",
    "sweet potato": "Sweet potato", "papaya": "Papaya",
}


def resolve_label(folder_name: str) -> int | None:
    key = folder_name.strip().lower()
    if key in _FOLDER_ALIASES:
        return FOLIO_SPECIES.index(_FOLDER_ALIASES[key])
    for i, name in enumerate(FOLIO_SPECIES):
        if name.lower() == key or name.lower() in key or key in name.lower():
            return i
    return None


def scan_folio(folio_root: Path) -> list[Record]:
    """扫描 Folio 目录，返回带标签的样本列表。"""
    if not folio_root.exists():
        raise FileNotFoundError(f"找不到 Folio 目录：{folio_root}")
    records: list[Record] = []
    for sub in sorted(folio_root.iterdir()):
        if not sub.is_dir():
            continue
        label = resolve_label(sub.name)
        if label is None:
            continue
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                records.append(Record(str(f), label, FOLIO_SPECIES[label]))
    if not records:
        raise RuntimeError(f"在 {folio_root} 中没有解析到任何 Folio 图片")
    return records


def stratified_split(records: list[Record], test_size: float = 0.2,
                     val_size: float = 0.1, seed: int = 42) -> dict[str, list[Record]]:
    """按类别分层划分（每类约 18~20 张，test 每类约 3~4 张）。"""
    import random
    rng = random.Random(seed)
    by_label: dict[int, list[Record]] = {}
    for rec in records:
        by_label.setdefault(rec.label, []).append(rec)

    train, val, test = [], [], []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round((n - n_test) * val_size)))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    for g in (train, val, test):
        g.sort(key=lambda r: r.path)
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------- #
# 增强与张量化（与 finetune.py 保持一致，仅做白底对齐 + 旋转 + 抖色）
# --------------------------------------------------------------------------- #
def _pad_square(image_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = (size * 0.96) / float(max(h, w))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    oy, ox = (size - new_h) // 2, (size - new_w) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def _color_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.75, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.7, 1.25), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)


def _augment(whitened: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """对「已白底对齐」的图做随机旋转 + 抖色（轻量，不做分割）。"""
    angle = rng.uniform(0, 360)
    h, w = whitened.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(w * cos + h * sin), int(w * sin + h * cos)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rotated = cv2.warpAffine(whitened, M, (nw, nh), borderValue=(255, 255, 255))
    view = _pad_square(rotated, size)
    return _color_jitter(view, rng)


def _preprocess_whiten(image_bgr: np.ndarray) -> np.ndarray:
    """白底对齐 + 抠叶贴白底，作为训练/验证的缓存输入。"""
    whitened = leaf_on_white(image_bgr)
    if whitened is None:
        whitened = image_bgr
    seg = segment_leaf(whitened)
    if seg.ok:
        return np.where(seg.crop_mask[..., None] > 0, seg.crop, 255)
    return whitened


def _tensorize(image_bgr: np.ndarray, size: int) -> np.ndarray:
    square = _pad_square(image_bgr, size)
    t = square.astype(np.float32) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(t, (2, 0, 1))


# --------------------------------------------------------------------------- #
# 训练主流程
# --------------------------------------------------------------------------- #
def run(
    folio_root: Path | str,
    model_path: Path | str | None = None,
    arch: str = "mobilenet_v3_small",
    size: int = DEFAULT_SIZE,
    epochs: int = 20,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_backbone: float = 2e-4,
    freeze_stages: int = 2,
    seed: int = 42,
    mixup_alpha: float = 0.2,
) -> tuple[Path, float]:
    """执行迁移学习，返回 (模型路径, 测试集准确率)。"""
    torch = _torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    folio_root = Path(folio_root)
    records = scan_folio(folio_root)
    splits = stratified_split(records)
    n_train, n_val, n_test = len(splits["train"]), len(splits["val"]), len(splits["test"])
    print(f"Folio 数据：{len(records)} 张 → train {n_train} / val {n_val} / test {n_test}")

    # 保存划分（供后续评估复用）
    project = Path(__file__).resolve().parents[1]
    save_split(splits, project / "data" / "splits" / "folio_split.json")

    # 一次性预计算白底对齐（分割最耗时），缓存到内存，训练时只做轻量增强
    print("预处理白底对齐（一次性，较慢）...")
    _cache: dict[str, np.ndarray] = {}
    for rec in records:
        data = np.fromfile(rec.path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None:
            image = np.full((size, size, 3), 255, np.uint8)
        _cache[rec.path] = _preprocess_whiten(image)
    print(f"  预处理完成，共 {len(_cache)} 张")

    class FolioDS(Dataset):
        def __init__(self, recs: list[Record], train: bool) -> None:
            self.recs = recs
            self.train = train

        def __len__(self) -> int:
            return len(self.recs)

        def __getitem__(self, idx: int):
            rec = self.recs[idx]
            image = _cache[rec.path]
            if self.train:
                r = np.random.default_rng(seed + idx * 7919 + epoch * 104729)
                view = _augment(image, r, size)
            else:
                view = _pad_square(image, size)
            return _tensorize(view, size), int(rec.label)

    from torchvision.models import MobileNet_V3_Small_Weights as W

    net = models.mobilenet_v3_small(weights=W.DEFAULT)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(FOLIO_SPECIES))

    # 冻结浅层，只微调后 freeze_stages 个 stage + 分类头
    for p in net.features[:-freeze_stages].parameters():
        p.requires_grad_(False)
    backbone_params = [p for n, p in net.named_parameters()
                       if n.startswith("features") and p.requires_grad]
    head_params = list(net.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_ds = FolioDS(splits["train"], train=True)
    val_ds = FolioDS(splits["val"], train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    out_path = Path(model_path) if model_path else project / "models" / DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"迁移学习开始：{len(train_ds)} 训练 / {len(val_ds)} 验证，"
          f"输入 {size}px，{epochs} 轮，冻结前 {freeze_stages} 个 stage，"
          f"MixUp α={mixup_alpha}")

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        net.train()
        t0, running, seen, correct = time.time(), 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            # MixUp：按比例混合两张图 + 标签（仅在训练时，小数据量下抗过拟合）
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha)) \
                    if np.random.rand() < 0.5 else 1.0
                if lam < 1.0:
                    perm = torch.randperm(images.size(0))
                    mixed = lam * images + (1 - lam) * images[perm]
                    labels_a, labels_b = labels, labels[perm]
                    logits = net(mixed)
                    loss = lam * criterion(logits, labels_a) \
                        + (1 - lam) * criterion(logits, labels_b)
                else:
                    logits = net(images)
                    loss = criterion(logits, labels)
            else:
                logits = net(images)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(labels)
            # MixUp 时的训练准确率仅作参考（用原始标签近似）
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        scheduler.step()
        train_acc = correct / max(1, seen)

        net.eval()
        vcorrect = 0
        with torch.no_grad():
            for images, labels in val_loader:
                vcorrect += int((net(images).argmax(1) == labels).sum())
        val_acc = vcorrect / max(1, len(val_ds))
        mark = ""
        if val_acc >= best_acc:
            best_acc, mark = val_acc, "  <- 保存最优"
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"  轮次 {epoch + 1:>2}/{epochs}  loss={running / max(1, seen):.4f}  "
              f"训练={train_acc:.3f}  验证={val_acc:.3f}  "
              f"({time.time() - t0:.0f}s){mark}", flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save({
        "state_dict": net.state_dict(),
        "arch": arch,
        "size": size,
        "num_classes": len(FOLIO_SPECIES),
        "class_names": FOLIO_SPECIES,
        "format": "leaves-finetune-v1",
        "val_acc": best_acc,
    }, out_path)
    print(f"\n已保存：{out_path}（验证准确率 {best_acc:.3f}）")

    # ------------------------------------------------------------------ #
    # 在留出的 test 集上评估
    # ------------------------------------------------------------------ #
    test_ds = FolioDS(splits["test"], train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    net.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            all_pred.append(net(images).argmax(1).cpu().numpy())
            all_true.append(labels.numpy())
    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    test_acc = float((preds == trues).mean())
    print(f"测试集准确率：{test_acc:.3f}（{int((preds == trues).sum())}/{len(trues)}）")

    return out_path, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="迁移学习训练 Folio 作物识别模型")
    parser.add_argument("--folio", required=True, help="Folio 数据根目录（含物种子文件夹）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--freeze-stages", type=int, default=2,
                        help="冻结前 N 个 stage（默认 2，只微调后几层 + 分类头）")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="MixUp 增强强度（0 关闭）")
    parser.add_argument("--model", default=None, help="输出模型路径")
    args = parser.parse_args()
    raise SystemExit(run(
        args.folio, model_path=args.model, epochs=args.epochs, size=args.size,
        freeze_stages=args.freeze_stages, mixup_alpha=args.mixup_alpha,
    )[0] and 0)
