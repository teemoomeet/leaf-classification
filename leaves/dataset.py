"""数据集扫描、划分与特征缓存。

支持两种目录组织方式：

1. **Flavia 原始目录**：所有图片平铺在一个目录里，标签由文件名推断
   （``1001.jpg`` 或 ``3.17.jpg``）。对应 ``data/raw/``。
2. **按类分文件夹**：``root/<物种名>/xxx.jpg``，标签由文件夹名匹配物种表
   （支持中文名、拉丁学名、别名）。用户自己的照片可以这样组织。
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import FEATURE_CACHE, RAW_DIR, SPLIT_DIR, ensure_dirs
from .species import (
    ALIASES,
    CHINESE_NAMES,
    NUM_CLASSES,
    SCIENTIFIC_NAMES,
    label_from_filename,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Record:
    """一条样本记录。"""

    path: str
    label: int
    label_name: str


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #
def _match_species(folder_name: str) -> int | None:
    """把文件夹名匹配到物种标签，匹配不到返回 None。"""
    key = folder_name.strip().lower()
    for label in range(NUM_CLASSES):
        candidates = {
            CHINESE_NAMES[label].lower(),
            SCIENTIFIC_NAMES[label].lower(),
            ALIASES[label].lower(),
        }
        if key in candidates:
            return label
    for label in range(NUM_CLASSES):
        if CHINESE_NAMES[label].lower() in key or key in CHINESE_NAMES[label].lower():
            return label
    return None


def scan_dataset(root: Path | str = RAW_DIR, recursive: bool = True) -> list[Record]:
    """扫描目录，返回带标签的样本列表。

    :param root: 数据根目录
    :param recursive: 是否递归子目录
    :raises FileNotFoundError: 目录不存在
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"数据目录不存在：{root}\n请先运行  python -m leaves.cli download  下载数据集"
        )

    pattern = "**/*" if recursive else "*"
    records: list[Record] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        label: int | None = None
        # 优先用所在文件夹名判断（用户自建数据集）
        folder_label = _match_species(path.parent.name) if path.parent != root else None
        if folder_label is not None:
            label = folder_label
        else:
            label = label_from_filename(path.name)

        if label is None:
            continue
        records.append(Record(str(path), label, CHINESE_NAMES[label]))

    if not records:
        raise RuntimeError(
            f"在 {root} 中没有找到可识别的叶片图像。\n"
            "请确认目录中是 Flavia 原始图片（如 1001.jpg / 3.17.jpg），"
            "或按 <物种名>/ 子目录组织。"
        )
    return records


def class_distribution(records: list[Record]) -> dict[int, int]:
    dist: dict[int, int] = {}
    for rec in records:
        dist[rec.label] = dist.get(rec.label, 0) + 1
    return dict(sorted(dist.items()))


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def stratified_split(
    records: list[Record],
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Record]]:
    """按类别分层划分训练 / 验证 / 测试集。

    :param test_size: 测试集比例
    :param val_size: 验证集比例（在剩余数据中占比）
    :param seed: 随机种子
    """
    rng = random.Random(seed)
    by_label: dict[int, list[Record]] = {}
    for rec in records:
        by_label.setdefault(rec.label, []).append(rec)

    train: list[Record] = []
    val: list[Record] = []
    test: list[Record] = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round((n - n_test) * val_size)))
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    for group in (train, val, test):
        group.sort(key=lambda r: r.path)
    return {"train": train, "val": val, "test": test}


def save_split(splits: dict[str, list[Record]], path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (SPLIT_DIR / "split.json")
    payload = {
        name: [asdict(rec) for rec in group] for name, group in splits.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_split(path: Path | None = None) -> dict[str, list[Record]]:
    path = path or (SPLIT_DIR / "split.json")
    if not path.exists():
        raise FileNotFoundError(f"划分文件不存在：{path}\n请先运行训练命令生成划分文件。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: [Record(**item) for item in items] for name, items in payload.items()
    }


# --------------------------------------------------------------------------- #
# 特征缓存
# --------------------------------------------------------------------------- #
def save_features(paths: list[str], X: np.ndarray, y: np.ndarray,
                  names: list[str], path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or FEATURE_CACHE
    np.savez_compressed(
        path,
        paths=np.asarray(paths, dtype=object),
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        names=np.asarray(names, dtype=object),
    )
    return path


def load_features(path: Path | None = None):
    path = path or FEATURE_CACHE
    if not path.exists():
        raise FileNotFoundError(f"特征缓存不存在：{path}\n请先运行  python -m leaves.cli features")
    data = np.load(path, allow_pickle=True)
    return (
        [str(p) for p in data["paths"]],
        data["X"].astype(np.float32),
        data["y"].astype(np.int64),
        [str(n) for n in data["names"]],
    )


def records_to_arrays(records: list[Record]) -> tuple[list[str], np.ndarray]:
    return [rec.path for rec in records], np.asarray([rec.label for rec in records], dtype=np.int64)


__all__ = [
    "Record",
    "scan_dataset",
    "class_distribution",
    "stratified_split",
    "save_split",
    "load_split",
    "save_features",
    "load_features",
    "records_to_arrays",
    "IMAGE_SUFFIXES",
]
