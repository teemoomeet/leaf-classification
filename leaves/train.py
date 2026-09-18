"""模型训练。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score

from . import __version__
from .backends import BACKENDS, extract_matrix_from_paths
from .config import FEATURE_CACHE, RANDOM_SEED, ensure_dirs
from .dataset import (
    IMAGE_SUFFIXES,  # noqa: F401  (对外保持可用)
    Record,
    class_distribution,
    records_to_arrays,
    save_features,
    save_split,
    scan_dataset,
    stratified_split,
)
from .models import AVAILABLE_MODELS, build_model, save_model
from .registry import MODEL_CHOICES  # noqa: F401
from .species import CHINESE_NAMES, NUM_CLASSES, SCIENTIFIC_NAMES


def compute_features(
    records: list[Record],
    backend: str = "features",
    arch: str = "mobilenet_v3_small",
    max_side: int = 384,
    workers: int = 4,
    batch_size: int = 32,
    chunk_size: int = 250,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    """分批提取特征，便于显示进度。

    :return: ``(X, y, paths, feature_names, seg_ok)``
    """
    paths, y = records_to_arrays(records)
    total = len(paths)
    features: list[np.ndarray] = []
    names: list[str] = []
    flags: list[np.ndarray] = []

    t0 = time.time()
    done = 0
    for start in range(0, total, chunk_size):
        batch = paths[start : start + chunk_size]
        X_chunk, names, seg_ok = extract_matrix_from_paths(
            batch,
            backend=backend,
            arch=arch,
            max_side=max_side,
            workers=workers,
            batch_size=batch_size,
            verbose=verbose,
        )
        features.append(X_chunk)
        flags.append(seg_ok)
        done += len(batch)
        if verbose:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  特征提取 {done}/{total}  ({done / total * 100:.1f}%)  "
                f"{rate:.1f} 张/秒  预计剩余 {eta / 60:.1f} 分钟",
                flush=True,
            )

    X = np.vstack(features).astype(np.float32)
    seg_ok = np.concatenate(flags)
    if not names:
        names = [f"f{i}" for i in range(X.shape[1])]
    return X, y, paths, names, seg_ok


def run(
    data_dir: Path | str | None = None,
    model_name: str = "svm",
    backend: str = "features",
    arch: str = "mobilenet_v3_small",
    max_side: int = 384,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = RANDOM_SEED,
    workers: int = 4,
    batch_size: int = 32,
    model_path: Path | str | None = None,
    cv_folds: int = 0,
) -> Path:
    """完整训练流程：扫描数据 -> 提特征 -> 划分 -> 训练 -> 保存模型。

    :param data_dir: 数据目录，默认 ``data/raw``
    :param model_name: 分类器名称，见 :data:`leaves.registry.AVAILABLE_MODELS`
    :param backend: 特征后端，``features``（手工特征）或 ``cnn``（深度特征）
    :param arch: ``cnn`` 后端使用的骨干网络
    :param max_side: 特征提取时图像最长边
    :param test_size: 测试集比例
    :param val_size: 验证集比例
    :param seed: 随机种子
    :param workers: 手工特征提取的线程数
    :param batch_size: 深度特征的批大小
    :param model_path: 模型保存路径
    :param cv_folds: 大于 1 时，在全量数据上做 K 折交叉验证
    :return: 模型文件路径
    """
    ensure_dirs()

    print("=" * 68)
    print(" Flavia 树叶识别 —— 模型训练")
    print("=" * 68)
    print(f"特征后端: {backend} —— {BACKENDS.get(backend, '未知')}")
    if backend == "cnn":
        print(f"骨干网络: {arch}")
    print(f"分类器  : {model_name} —— {AVAILABLE_MODELS.get(model_name, '未知')}")

    records = scan_dataset(data_dir) if data_dir is not None else scan_dataset()
    print(f"扫描到 {len(records)} 张图像，共 {len(class_distribution(records))} 个物种")

    print("\n[1/4] 提取特征 ...")
    X, y, paths, feat_names, seg_ok = compute_features(
        records, backend=backend, arch=arch, max_side=max_side,
        workers=workers, batch_size=batch_size,
    )
    print(f"  特征维度 {X.shape[1]}")
    if backend == "features":
        print(f"  分割失败 {int((~seg_ok).sum())} 张（已退化为整图特征，不影响流程）")
    cache_name = "features_handcrafted" if backend == "features" else f"features_{backend}_{arch}"
    cache_path = FEATURE_CACHE.with_name(cache_name + ".npz")
    save_features(paths, X, y, feat_names, cache_path)
    print(f"  特征已缓存到 {cache_path}")

    print("\n[2/4] 划分数据集 ...")
    splits = stratified_split(records, test_size=test_size, val_size=val_size, seed=seed)
    split_file = save_split(splits)
    index = {path: i for i, path in enumerate(paths)}

    def _slice(group: list[Record]) -> tuple[np.ndarray, np.ndarray]:
        idx = [index[rec.path] for rec in group]
        return X[idx], y[idx]

    X_tr, y_tr = _slice(splits["train"])
    X_va, y_va = _slice(splits["val"])
    X_te, y_te = _slice(splits["test"])
    print(
        f"  训练集 {len(y_tr)} / 验证集 {len(y_va)} / 测试集 {len(y_te)}"
        f"  （划分已保存到 {split_file}）"
    )

    print("\n[3/4] 训练模型 ...")
    pipeline = build_model(model_name, seed=seed)
    t0 = time.time()
    pipeline.fit(X_tr, y_tr)
    print(f"  训练完成，用时 {time.time() - t0:.1f} 秒")

    train_acc = float(pipeline.score(X_tr, y_tr))
    val_acc = float(pipeline.score(X_va, y_va)) if len(y_va) else float("nan")
    test_acc = float(pipeline.score(X_te, y_te)) if len(y_te) else float("nan")
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  验证集准确率: {val_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")

    cv_score = None
    if cv_folds and cv_folds > 1:
        print(f"  正在进行 {cv_folds} 折交叉验证 ...")
        scores = cross_val_score(
            build_model(model_name, seed=seed), X, y, cv=cv_folds, n_jobs=1
        )
        cv_score = float(scores.mean())
        print(f"  {cv_folds} 折交叉验证准确率: {cv_score:.4f} ± {scores.std():.4f}")

    print("\n[4/4] 保存模型 ...")
    meta = {
        "project": "flavia-leaf-recognition",
        "version": __version__,
        "backend": backend,
        "arch": arch if backend == "cnn" else None,
        "estimator": model_name,
        "feature_names": feat_names,
        "n_features": int(X.shape[1]),
        "max_side": max_side,
        "num_classes": NUM_CLASSES,
        "species_cn": CHINESE_NAMES,
        "species_sci": SCIENTIFIC_NAMES,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "cv_accuracy": cv_score,
        "seed": seed,
    }
    out = save_model(pipeline, meta, model_path)
    print(f"  模型已保存到 {out}")
    print("\n下一步：python -m leaves.cli predict --input <图片目录>")
    return out


__all__ = ["run", "compute_features"]
