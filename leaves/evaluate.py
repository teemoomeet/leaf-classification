"""模型评估：准确率、分类报告、混淆矩阵、各类别准确率。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

from .config import OUTPUT_DIR, ensure_dirs
from .dataset import load_split, scan_dataset
from .features import extract_features
from .models import load_model, predict_with_confidence
from .segmentation import segment_leaf
from .species import CHINESE_NAMES, SCIENTIFIC_NAMES
from .viz import plot_confusion_matrix, plot_per_class_accuracy


def _features_for(paths: list[str], max_side: int = 384) -> np.ndarray:
    import cv2

    rows = []
    for path in paths:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"无法读取图像：{path}")
        h, w = image.shape[:2]
        scale = max_side / float(max(h, w))
        if scale < 1.0:
            image = cv2.resize(
                image, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        seg = segment_leaf(image)
        vector, _, _ = extract_features(image, max_side=max_side, precomputed_seg=seg)
        rows.append(vector)
    return np.vstack(rows)


def run(
    model_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    split: str = "test",
    output_dir: Path | str | None = None,
    top_k: int = 3,
) -> dict:
    """在划分好的测试集（或验证集）上评估模型。

    :param model_path: 模型文件路径，None 使用默认路径
    :param data_dir: 数据目录，None 时使用训练时保存的划分文件
    :param split: ``train`` / ``val`` / ``test``
    :param output_dir: 结果输出目录
    :param top_k: Top-K 准确率的 K
    :return: 评估指标字典
    """
    ensure_dirs()
    out_dir = Path(output_dir) if output_dir else (OUTPUT_DIR / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline, meta = load_model(model_path)
    max_side = int(meta.get("max_side", 384))

    if data_dir is not None:
        from .dataset import stratified_split

        records = scan_dataset(data_dir)
        splits = stratified_split(records, seed=int(meta.get("seed", 42)))
        group = splits[split]
    else:
        splits = load_split()
        group = splits[split]

    paths = [rec.path for rec in group]
    y_true = np.asarray([rec.label for rec in group], dtype=np.int64)
    print(f"评估集合: {split}，共 {len(paths)} 张图像")

    print("提取特征 ...")
    X = _features_for(paths, max_side=max_side)

    y_pred, confidence, top_labels, top_scores = predict_with_confidence(
        pipeline, X, top_k=top_k
    )

    acc = float(accuracy_score(y_true, y_pred))
    try:
        proba = _proba(pipeline, X)
        model_labels = np.asarray(pipeline.classes_)
        topk = float(top_k_accuracy_score(y_true, proba, k=top_k, labels=model_labels))
    except Exception:  # noqa: BLE001 - 某些分类器没有概率输出
        topk = float("nan")

    print(f"\n总体准确率 : {acc:.4f}")
    if not np.isnan(topk):
        print(f"Top-{top_k} 准确率: {topk:.4f}")

    present = sorted(set(y_true.tolist()))
    target_names = [f"{CHINESE_NAMES[i]}({SCIENTIFIC_NAMES[i]})" for i in present]
    report = classification_report(
        y_true, y_pred, labels=present, target_names=target_names,
        digits=4, zero_division=0,
    )
    print("\n分类报告:")
    print(report)
    (out_dir / "classification_report.txt").write_text(
        f"split={split}\naccuracy={acc:.4f}\ntop{top_k}_accuracy={topk:.4f}\n\n{report}",
        encoding="utf-8",
    )

    cm = confusion_matrix(y_true, y_pred, labels=present)
    plot_confusion_matrix(cm, present, title=f"Flavia 混淆矩阵（{split}）",
                          output=out_dir / "confusion_matrix.png")

    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    plot_per_class_accuracy(present, per_class_acc.tolist(),
                            out_dir / "per_class_accuracy.png")

    frame = pd.DataFrame(
        {
            "path": paths,
            "true_label": y_true,
            "true_name": [CHINESE_NAMES[i] for i in y_true],
            "pred_label": y_pred,
            "pred_name": [CHINESE_NAMES[i] for i in y_pred],
            "confidence": confidence,
            "correct": y_true == y_pred,
        }
    )
    for rank in range(top_k):
        frame[f"top{rank + 1}_name"] = [CHINESE_NAMES[i] for i in top_labels[:, rank]]
        frame[f"top{rank + 1}_score"] = top_scores[:, rank]
    frame.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"label": present, "name": [CHINESE_NAMES[i] for i in present],
         "accuracy": per_class_acc}
    ).to_csv(out_dir / "per_class_accuracy.csv", index=False, encoding="utf-8-sig")

    summary = {
        "split": split,
        "n_samples": len(paths),
        "accuracy": acc,
        f"top{top_k}_accuracy": topk,
        "per_class_accuracy": {
            CHINESE_NAMES[i]: float(a) for i, a in zip(present, per_class_acc)
        },
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    errors = frame[~frame["correct"]]
    print(f"\n错分样本 {len(errors)} 张，结果已写入 {out_dir}")
    if len(errors):
        print("错分最多的类别:")
        top_errors = errors["true_name"].value_counts().head(8)
        for name, count in top_errors.items():
            print(f"  {name:<10} {count} 张")
    return summary


def _proba(pipeline, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    scores = pipeline.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


__all__ = ["run"]
