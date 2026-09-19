"""批量叶片识别（本项目的核心功能）。

给一批树叶图片，逐张输出「属于哪一种树」的结论，并给出中文名、拉丁学名、
置信度和 Top-3 候选，结果导出为 CSV / JSON，可选导出可视化拼图。

模型自带的元信息里记录了使用哪条技术路线（手工特征 / 深度特征），
因此这里会自动选用与训练时一致的特征提取方式。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .backends import extract_matrix
from .config import OUTPUT_DIR, ensure_dirs
from .dataset import IMAGE_SUFFIXES
from .models import load_model, predict_with_confidence
from .segmentation import SegmentedLeaf, leaf_on_white, segment_leaf
from .species import CHINESE_NAMES, SCIENTIFIC_NAMES
from .viz import plot_samples


@dataclass
class PredictOptions:
    """批量识别参数。"""

    model_path: Path | str | None = None
    max_side: int = 384
    top_k: int = 3
    #: 置信度低于该值时标记为「不确定」
    low_confidence: float = 0.35
    save_visualization: bool = True
    #: 可视化最多输出多少张
    max_visualized: int = 60
    #: 白底对齐预处理：``auto`` = 先把叶片抠到纯白背景再识别（适合真实
    #: 场景照片，与 Flavia 白底扫描图的分布对齐）；``off`` = 关闭。
    whiten: str = "auto"
    #: 内部使用：结果输出目录
    output_dir: Path | None = field(default=None, repr=False)


def _read_image(path: Path) -> np.ndarray | None:
    """用 imdecode 读取，兼容中文路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def list_images(input_dir: Path | str, recursive: bool = True) -> list[Path]:
    """列出目录下的所有图片（或校验单张图片）。"""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"输入目录不存在：{root}")
    if root.is_file():
        if root.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不是支持的图片格式：{root}")
        return [root]

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"{root} 下没有找到图片文件")
    return files


def _peak_crop_views(image: np.ndarray, n: int = 2) -> list[np.ndarray]:
    """距离变换峰值裁窗：从（白底化后的）图里取最像「单片叶」的局部视角。

    多叶簇生时整簇形状与单叶训练数据差异大，围绕叶身最厚处裁出的
    局部窗往往更接近训练分布；对单叶图它只是无害的放大裁剪。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = (~((s < 35) & (v > 200))).astype(np.uint8) * 255
    if np.count_nonzero(mask) < 0.02 * mask.size:
        return []
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (15, 15), 0)
    H, W = image.shape[:2]
    order = np.argsort(-dist.ravel())
    picked: list[tuple[int, int]] = []
    crops: list[np.ndarray] = []
    min_gap = int(0.12 * max(H, W))
    for flat_idx in order[:4000]:
        y, x = divmod(int(flat_idx), W)
        if all((x - px) ** 2 + (y - py) ** 2 > min_gap**2 for px, py in picked):
            picked.append((x, y))
            r = int(max(50, dist[y, x] * 2.2))
            crops.append(image[max(0, y - r):min(H, y + r),
                               max(0, x - r):min(W, x + r)].copy())
            if len(crops) >= n:
                break
    return crops


def recognize(
    image_paths: list[Path | str],
    options: PredictOptions | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """对一批图片做识别，返回结果表。

    返回的 DataFrame 列包括：文件名、路径、预测类别、中文名、拉丁学名、
    置信度、是否低置信度、Top-1~Top-K 候选，以及叶片面积占比等尺度信息。
    """
    options = options or PredictOptions()
    # 两类模型：.joblib = 冻结特征 + 传统分类器；.pt = 端到端微调网络
    torch_clf = None
    if options.model_path and str(options.model_path).lower().endswith(".pt"):
        from .torch_model import TorchLeafClassifier

        torch_clf = TorchLeafClassifier.load(options.model_path)
        backend, arch = "torch", ""
        # 白底化等预处理仍用较大分辨率，模型输入尺寸在张量化时统一
        model_max_side = options.max_side
    else:
        pipeline, meta = load_model(options.model_path)
        backend = str(meta.get("backend", "features"))
        arch = str(meta.get("arch") or "mobilenet_v3_small")
        model_max_side = int(meta.get("max_side", options.max_side))
    top_k = max(1, int(options.top_k))

    rows: list[dict] = []
    loaded: list[np.ndarray] = []
    valid_indices: list[int] = []
    previews: list[tuple[np.ndarray, SegmentedLeaf]] = []

    for path in image_paths:
        path = Path(path)
        record: dict = {"file": path.name, "path": str(path)}
        image = _read_image(path)
        if image is None:
            record.update({"status": "读取失败", "pred_name": "-", "confidence": 0.0})
            rows.append(record)
            continue

        h, w = image.shape[:2]
        resized = _resize(image, model_max_side)
        # 白底对齐：真实场景照片先抠叶贴白底，再走与训练一致的分割/特征流程
        preprocessed = False
        work = resized
        if options.whiten != "off":
            whitened = leaf_on_white(resized)
            if whitened is not None:
                work = whitened
                preprocessed = True
        seg = segment_leaf(work)

        record.update(
            {
                "width": w,
                "height": h,
                "leaf_area_ratio": round(seg.area_ratio, 4),
                "segmented": bool(seg.ok),
                "preprocessed": preprocessed,
                "status": "ok",
            }
        )
        rows.append(record)
        loaded.append(work)
        valid_indices.append(len(rows) - 1)
        # 可视化与原图同尺寸无关：直接用缩放后的图，保证掩码与画布对齐
        previews.append((work, seg))

    if loaded:
        if torch_clf is not None:
            # 多视角融合：主视角（已按 --whiten 预处理）+ 峰值局部窗，
            # 逐类取各视角最大概率 —— 多叶簇生时显著提升鲁棒性。
            all_views: list[np.ndarray] = []
            view_owner: list[int] = []
            for j, im in enumerate(loaded):
                all_views.append(im)
                view_owner.append(j)
            if options.whiten != "off":
                for j, im in enumerate(loaded):
                    for crop in _peak_crop_views(im, n=2):
                        all_views.append(crop)
                        view_owner.append(j)
            view_probs = torch_clf.predict_proba(all_views)
            n_cls = view_probs.shape[1]
            fused = np.zeros((len(loaded), n_cls), np.float32)
            best_view = [0] * len(loaded)
            for vi, owner in enumerate(view_owner):
                probs_i = view_probs[vi]
                improved = probs_i > fused[owner]
                fused[owner] = np.where(improved, probs_i, fused[owner])
                if improved.any():
                    best_view[owner] = vi
            pred = fused.argmax(axis=1)
            confidence = fused.max(axis=1)
            k = min(top_k, n_cls)
            top_labels = np.argsort(-fused, axis=1)[:, :k]
            top_scores = np.take_along_axis(fused, top_labels, axis=1)
            for j, row_idx in enumerate(valid_indices):
                rows[row_idx]["best_view"] = (
                    "主视角" if best_view[j] < len(loaded) else f"局部窗{best_view[j] - len(loaded) + 1}"
                )
        else:
            X, _, _ = extract_matrix(
                loaded,
                backend=backend,
                arch=arch,
                max_side=model_max_side,
                workers=4,
                verbose=False,
            )
            pred, confidence, top_labels, top_scores = predict_with_confidence(
                pipeline, X, top_k=top_k
            )
        for j, row_idx in enumerate(valid_indices):
            label = int(pred[j])
            record = rows[row_idx]
            record["pred_label"] = label
            record["pred_name"] = CHINESE_NAMES[label]
            record["scientific_name"] = SCIENTIFIC_NAMES[label]
            record["confidence"] = float(confidence[j])
            record["low_confidence"] = bool(confidence[j] < options.low_confidence)
            for rank in range(top_labels.shape[1]):
                rank_label = int(top_labels[j, rank])
                record[f"top{rank + 1}_name"] = CHINESE_NAMES[rank_label]
                record[f"top{rank + 1}_score"] = float(top_scores[j, rank])
            if verbose:
                flag = "  ⚠ 低置信度" if record["low_confidence"] else ""
                print(
                    f"  {record['file']:<28} -> {record['pred_name']:<8}"
                    f"({record['scientific_name']})"
                    f"  置信度 {record['confidence']:.3f}{flag}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column, default in (("pred_name", "-"), ("confidence", 0.0)):
        if column not in frame.columns:
            frame[column] = default

    if options.save_visualization and previews:
        vis_dir = (options.output_dir or OUTPUT_DIR) / "vis"
        _save_visualizations(rows, valid_indices, previews, options, vis_dir)
    return frame


def _save_visualizations(
    rows: list[dict],
    valid_indices: list[int],
    previews: list[tuple[np.ndarray, SegmentedLeaf]],
    options: PredictOptions,
    output_dir: Path,
) -> None:
    """导出「原图 + 分割轮廓 + 预测标签」的拼图。"""
    step = max(1, len(previews) // max(1, options.max_visualized))
    picked = list(range(0, len(previews), step))[: options.max_visualized]
    if not picked:
        return

    images: list[np.ndarray] = []
    titles: list[str] = []
    for j in picked:
        image, seg = previews[j]
        record = rows[valid_indices[j]]

        canvas = image.copy()
        mask = seg.mask
        if mask.shape[:2] == canvas.shape[:2]:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, (0, 90, 220), 3)
            x, y, bw, bh = seg.bbox
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 160, 0), 2)

        images.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        titles.append(
            f"{record['file']}\n{record.get('pred_name', '-')} "
            f"{record.get('confidence', 0):.2f}"
        )

    plot_samples(images, titles, output_dir / "recognition_grid.png", cols=6)


def recognize_folder(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    recursive: bool = True,
    options: PredictOptions | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """批量识别一个目录下的所有树叶图片，并导出结果文件。

    :param input_dir: 待识别的图片目录（或单张图片）
    :param output_dir: 结果输出目录，默认 ``outputs/predict_<目录名>``
    :param recursive: 是否递归子目录
    :param options: 识别参数
    :return: 结果 DataFrame
    """
    ensure_dirs()
    options = options or PredictOptions()
    files = list_images(input_dir, recursive=recursive)

    if output_dir is None:
        stem = Path(input_dir).name or "result"
        output_dir = OUTPUT_DIR / f"predict_{stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    options.output_dir = output_dir

    print(f"待识别图片 {len(files)} 张，目录：{Path(input_dir)}")
    print("-" * 68)
    frame = recognize(files, options, verbose=verbose)

    csv_path = output_dir / "predictions.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = output_dir / "predictions.json"
    json_path.write_text(
        json.dumps(frame.to_dict(orient="records"), ensure_ascii=False, indent=2,
                   default=str),
        encoding="utf-8",
    )

    print("-" * 68)
    ok = frame[frame["status"] == "ok"] if "status" in frame else frame
    print(f"识别完成：成功 {len(ok)} / {len(frame)} 张")
    if len(ok):
        print("\n识别结果分布（Top-1）:")
        for name, count in ok["pred_name"].value_counts().items():
            print(f"  {name:<10} {count:>4d} 张")
        low = ok[ok.get("low_confidence", False)]
        if len(low):
            print(
                f"\n⚠ 有 {len(low)} 张置信度低于 {options.low_confidence:.2f}，"
                "可能不属于这 32 种树，建议人工复核（见 CSV 的 low_confidence 列）。"
            )
    print(f"\n结果已保存：\n  {csv_path}\n  {json_path}")
    if options.save_visualization:
        print(f"  {output_dir / 'vis' / 'recognition_grid.png'}")
    return frame


__all__ = ["PredictOptions", "recognize", "recognize_folder", "list_images"]
