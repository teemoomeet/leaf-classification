# -*- coding: utf-8 -*-
"""统一叶片识别入口：一张叶子自动判断「树 / 作物 / 未知」。

组合项目里已有的三个模型，做到：
  * Flavia 树模型（32 种中国树木） + Mahalanobis OOD 检测器；
  * Folio 作物模型（32 种作物/热带植物，迁移学习微调得到）。

对每张图：
  1. 提取与 Flavia 训练一致的手工特征，计算 OOD 分数；
  2. OOD 分数 ≤ 阈值  → 落在「树」分布内，用 Flavia 模型识别树种；
  3. OOD 分数 > 阈值  → 不在树分布内，改用 Folio 模型识别作物；
  4. Folio 模型置信度也偏低 → 判为「未知」。

用法（在 D:\\leaves 下，用项目 venv）：
    .\\.venv\\Scripts\\python.exe -m leaves.classify_leaf --input "D:\\某张叶子.jpg"
    .\\.venv\\Scripts\\python.exe -m leaves.classify_leaf --input "D:\\某文件夹"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .backends import extract_matrix
from .dataset import IMAGE_SUFFIXES
from .models import load_model, predict_with_confidence
from .ood import load_ood, ood_scores
from .species import CHINESE_NAMES, SCIENTIFIC_NAMES

# Folio 作物模型（torch checkpoint）
FOLIO_MODEL_DEFAULT = "flavia_ft_folio_mobilenetv3.pt"

# 作物模型置信度低于该值时判「未知」
CROP_LOW_CONF = 0.35


@dataclass
class ClassifyOptions:
    """统一识别参数。"""

    tree_model: str | None = None          # Flavia 树模型（.joblib）
    crop_model: str | None = None          # Folio 作物模型（.pt）
    ood_path: str | None = None            # OOD 检测器
    whiten: bool = True                    # 是否白底对齐
    top_k: int = 3
    crop_low_conf: float = CROP_LOW_CONF   # 作物置信度阈值


def _read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _load_crop_model(model_path: Path):
    """加载 Folio 作物迁移模型，返回 (net, size, class_names)。"""
    from .deep import _torch
    torch = _torch()
    import torch.nn as nn
    from torchvision import models
    from torchvision.models import MobileNet_V3_Small_Weights as W

    ckpt = torch.load(model_path, map_location="cpu")
    size = ckpt.get("size", 224)
    num_classes = ckpt.get("num_classes", 32)
    class_names = ckpt.get("class_names", [f"crop_{i}" for i in range(num_classes)])

    net = models.mobilenet_v3_small(weights=W.DEFAULT)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, size, class_names


def _preprocess_crop(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """与 train_folio 一致的作物模型预处理（白底对齐 + 填方 + 归一化）。"""
    from .segmentation import leaf_on_white, segment_leaf
    from .deep import IMAGENET_MEAN, IMAGENET_STD

    whitened = leaf_on_white(image_bgr)
    if whitened is None:
        whitened = image_bgr
    seg = segment_leaf(whitened)
    crop = np.where(seg.crop_mask[..., None] > 0, seg.crop, 255) if seg.ok else whitened

    h, w = crop.shape[:2]
    scale = (size * 0.96) / float(max(h, w, 1))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    oy, ox = (size - new_h) // 2, (size - new_w) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    t = canvas.astype(np.float32) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(t, (2, 0, 1))


def classify(
    image_paths: list[Path | str],
    options: ClassifyOptions | None = None,
    verbose: bool = True,
) -> list[dict]:
    """对一批图片做「树/作物/未知」统一识别，返回结果列表。"""
    from .config import MODEL_DIR

    options = options or ClassifyOptions()

    # --- 加载 Flavia 树模型 + OOD 检测器 ---
    tree_path = Path(options.tree_model) if options.tree_model else (
        MODEL_DIR / "flavia_svm.joblib")
    pipeline, meta = load_model(tree_path)
    backend = str(meta.get("backend", "features"))
    arch = str(meta.get("arch") or "mobilenet_v3_small")
    tree_max_side = int(meta.get("max_side", 384))

    ood_loaded = load_ood(options.ood_path)
    if ood_loaded is None:
        raise RuntimeError(
            "未找到 OOD 检测器，请先运行  python -m leaves.train_ood  生成"
        )
    ood_detector, ood_threshold = ood_loaded

    # --- 加载 Folio 作物模型 ---
    crop_path = Path(options.crop_model) if options.crop_model else (
        MODEL_DIR / FOLIO_MODEL_DEFAULT)
    if not crop_path.exists():
        raise FileNotFoundError(
            f"未找到作物模型 {crop_path}，请先运行  python -m leaves.train_folio"
        )
    crop_net, crop_size, crop_names = _load_crop_model(crop_path)
    torch = __import__("torch")

    results: list[dict] = []

    # 逐张处理（为保持简单清晰，这里逐张提取特征，量不大时可接受）
    for path in image_paths:
        path = Path(path)
        rec: dict = {"file": path.name, "path": str(path)}
        image = _read_image(path)
        if image is None:
            rec.update({"status": "读取失败", "category": "-", "pred_name": "-",
                        "confidence": 0.0})
            results.append(rec)
            continue

        work = _resize(image, tree_max_side)
        if options.whiten:
            from .segmentation import leaf_on_white
            whitened = leaf_on_white(work)
            if whitened is not None:
                work = whitened

        # 1) 手工特征 + OOD 分数
        X, _, _ = extract_matrix([work], backend=backend, arch=arch,
                                 max_side=tree_max_side, workers=1, verbose=False)
        ood = float(ood_scores(ood_detector, X)[0])

        if ood <= ood_threshold:
            # 2) 落在树分布内 → Flavia 认树
            pred, conf, top_labels, top_scores = predict_with_confidence(
                pipeline, X, top_k=options.top_k)
            label = int(pred[0])
            rec.update({
                "status": "ok", "category": "树", "pred_name": CHINESE_NAMES[label],
                "scientific_name": SCIENTIFIC_NAMES[label],
                "confidence": float(conf[0]), "ood_score": ood,
            })
            for rank in range(top_labels.shape[1]):
                rl = int(top_labels[0, rank])
                rec[f"top{rank + 1}_name"] = CHINESE_NAMES[rl]
                rec[f"top{rank + 1}_score"] = float(top_scores[0, rank])
        else:
            # 3) 不在树分布内 → Folio 认作物
            crop_input = _preprocess_crop(work, crop_size)
            with torch.no_grad():
                logits = crop_net(torch.from_numpy(crop_input[None]).float())
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = np.argsort(-probs)[:options.top_k]
            best = int(top_idx[0])
            best_conf = float(probs[best])
            if best_conf < options.crop_low_conf:
                rec.update({"status": "ok", "category": "未知", "pred_name": "未知",
                            "scientific_name": "-", "confidence": best_conf,
                            "ood_score": ood})
            else:
                rec.update({"status": "ok", "category": "作物",
                            "pred_name": crop_names[best], "scientific_name": "-",
                            "confidence": best_conf, "ood_score": ood})
            for rank, idx in enumerate(top_idx):
                rec[f"top{rank + 1}_name"] = crop_names[int(idx)]
                rec[f"top{rank + 1}_score"] = float(probs[int(idx)])

        if verbose:
            cat = rec.get("category", "-")
            print(f"  {rec['file']:<30} [{cat}] {rec['pred_name']}  "
                  f"置信度 {rec['confidence']:.3f}  (OOD {rec['ood_score']:.1f})",
                  flush=True)
        results.append(rec)

    return results


def _collect_images(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统一叶片识别：树 / 作物 / 未知")
    parser.add_argument("--input", required=True, help="图片文件或文件夹")
    parser.add_argument("--tree-model", default=None, help="Flavia 树模型路径")
    parser.add_argument("--crop-model", default=None, help="Folio 作物模型路径")
    parser.add_argument("--ood", default=None, help="OOD 检测器路径")
    parser.add_argument("--out", default=None, help="结果 CSV 输出路径")
    parser.add_argument("--no-whiten", action="store_true", help="关闭白底对齐")
    args = parser.parse_args(argv)

    images = _collect_images(Path(args.input))
    if not images:
        print("没有找到图片")
        return 1

    opts = ClassifyOptions(
        tree_model=args.tree_model, crop_model=args.crop_model,
        ood_path=args.ood, whiten=not args.no_whiten,
    )
    print(f"统一识别 {len(images)} 张图片")
    print("-" * 72)
    results = classify(images, opts)
    print("-" * 72)

    # 汇总
    from collections import Counter
    cats = Counter(r.get("category", "-") for r in results)
    print("分类汇总：", dict(cats))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if results:
                w.writerow(list(results[0].keys()))
                for r in results:
                    w.writerow([r.get(k, "") for k in results[0].keys()])
        print(f"结果已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
