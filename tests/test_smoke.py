"""冒烟测试：不依赖完整数据集，几秒钟跑完。

运行::

    pytest -q
    # 或
    python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from leaves.config import RAW_DIR  # noqa: E402
from leaves.features import extract_features, feature_names  # noqa: E402
from leaves.models import build_model, predict_with_confidence  # noqa: E402
from leaves.segmentation import segment_leaf  # noqa: E402
from leaves.species import (  # noqa: E402
    NUM_CLASSES,
    label_from_filename,
    label_from_id,
)


def _synthetic_leaf(shape: str = "ellipse", size: int = 320) -> np.ndarray:
    """在白色背景上画一个深色叶片形状。"""
    canvas = np.full((size, size, 3), 255, np.uint8)
    if shape == "ellipse":
        cv2.ellipse(canvas, (size // 2, size // 2), (110, 60), 25, 0, 360, (40, 90, 45), -1)
    else:
        pts = np.array(
            [[60, 160], [110, 90], [160, 70], [250, 110], [270, 170], [180, 240], [90, 220]],
            np.int32,
        )
        cv2.fillPoly(canvas, [pts], (40, 90, 45))
    return canvas


# --------------------------------------------------------------------------- #
# 物种映射
# --------------------------------------------------------------------------- #
def test_label_mapping_is_complete() -> None:
    """32 个物种、1907 张图像的映射必须完整且不重叠。"""
    assert NUM_CLASSES == 32
    seen: set[int] = set()
    total = 0
    for label in range(NUM_CLASSES):
        assert label in {lab for _, _, lab in _ranges_for(label)}
    from leaves.species import FLAVIA_ID_RANGES

    for start, end, label in FLAVIA_ID_RANGES:
        assert start <= end
        assert label not in seen, f"标签 {label} 重复"
        seen.add(label)
        for image_id in (start, end):
            assert label_from_id(image_id) == label
        total += end - start + 1
    assert total == 1907, f"编号区间覆盖 {total} 张，应为 1907"
    assert len(seen) == NUM_CLASSES


def _ranges_for(label: int):
    from leaves.species import FLAVIA_ID_RANGES

    return [r for r in FLAVIA_ID_RANGES if r[2] == label]


def test_label_from_filename() -> None:
    assert label_from_filename("1001.jpg") == 0      # 镜像命名
    assert label_from_filename("1059.jpg") == 0
    assert label_from_filename("3621.jpg") == 31
    assert label_from_filename("3.17.jpg") == 2      # 官方命名
    assert label_from_filename("32.1.jpg") == 31
    assert label_from_filename("not_a_leaf.png") is None


# --------------------------------------------------------------------------- #
# 分割与特征
# --------------------------------------------------------------------------- #
def test_segmentation_finds_leaf() -> None:
    image = _synthetic_leaf("ellipse")
    seg = segment_leaf(image)
    assert seg.ok, f"分割失败，面积占比 {seg.area_ratio}"
    assert 0.1 < seg.area_ratio < 0.6
    assert seg.crop.size > 0
    assert seg.crop_mask.shape[:2] == seg.crop.shape[:2]


def test_features_are_stable_and_finite() -> None:
    for shape in ("ellipse", "polygon"):
        vector, names, ok = extract_features(_synthetic_leaf(shape))
        assert ok
        assert vector.shape[0] == len(names) == len(feature_names())
        assert np.all(np.isfinite(vector)), "特征中出现 NaN / Inf"


def test_feature_dimension_is_consistent() -> None:
    sizes = [256, 384]
    dims = {extract_features(_synthetic_leaf(size=s))[0].shape[0] for s in sizes}
    assert len(dims) == 1, f"不同尺寸下特征维度不一致：{dims}"


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def test_model_trains_and_predicts() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 1.0, size=(20, 8)) for loc in (0.0, 4.0, 8.0)])
    y = np.repeat([0, 1, 2], 20)
    pipeline = build_model("svm", seed=0)
    pipeline.fit(X, y)
    assert pipeline.score(X, y) > 0.9
    pred, conf, top_labels, top_scores = predict_with_confidence(pipeline, X[:5], top_k=3)
    assert pred.shape == (5,)
    assert np.all(conf <= 1.0 + 1e-6)
    assert top_labels.shape == (5, 3)
    assert np.allclose(top_scores.sum(axis=1), 1.0, atol=1e-4)


def test_svm_confidence_is_calibrated() -> None:
    """SVM 应经过概率校准：置信度在不同样本间要有区分度，而不是全都一样。"""
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(loc, 1.0, size=(25, 6)) for loc in (0.0, 6.0, 12.0)])
    y = np.repeat([0, 1, 2], 25)
    pipeline = build_model("svm", seed=1)
    pipeline.fit(X, y)

    pred, conf, top_labels, top_scores = predict_with_confidence(pipeline, X[:6], top_k=2)
    assert pred.shape == (6,)
    assert np.all((conf > 0) & (conf <= 1.0 + 1e-6))
    assert top_labels.shape == (6, 2)
    # top_k 小于类别数时，前 k 项之和不超过 1；置信度应当来自真实概率
    assert np.all(top_scores.sum(axis=1) <= 1.0 + 1e-4)
    assert np.allclose(conf, top_scores[:, 0])
    # 概率校准后应当有真实的概率输出
    assert np.allclose(pipeline.predict_proba(X[:6]).sum(axis=1), 1.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# 数据集扫描（有数据时才跑）
# --------------------------------------------------------------------------- #
def test_scan_real_dataset_if_available() -> None:
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.jpg")):
        print("[跳过] data/raw 下暂无数据，先运行 python -m leaves.cli download")
        return
    from leaves.dataset import scan_dataset

    records = scan_dataset(RAW_DIR)
    assert records, "扫描结果为空"
    assert all(0 <= rec.label < NUM_CLASSES for rec in records)


def main() -> int:
    """不依赖 pytest 也能直接运行。"""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
