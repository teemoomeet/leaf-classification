"""统一特征后端：把两种技术路线封装成同一个接口。

* ``features`` —— 手工设计的形状/纹理/颜色特征 + 传统机器学习（默认，无需 GPU）
* ``cnn``      —— 预训练卷积网络的深度特征 + 线性分类器（需要 PyTorch）

上层（训练 / 评估 / 预测 / 命令行）只调用 :func:`extract_matrix`，
不必关心底层用的是哪条路线。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import extract_features, feature_names
from .registry import BACKENDS  # noqa: F401  (统一出口，保持纯标准库可用)
from .segmentation import SegmentedLeaf, segment_leaf


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def load_images(paths: list[Path | str], max_side: int = 384) -> list[np.ndarray]:
    """读取并等比缩放一批图片（兼容中文路径，失败时用白图占位）。"""
    import cv2

    images: list[np.ndarray] = []
    for path in paths:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None:
            image = np.full((max_side, max_side, 3), 255, np.uint8)
        images.append(_resize(image, max_side))
    return images


def extract_matrix(
    images: list[np.ndarray],
    backend: str = "features",
    arch: str = "mobilenet_v3_small",
    max_side: int = 384,
    workers: int = 4,
    batch_size: int = 32,
    segs: list[SegmentedLeaf] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """对一批已读入的图像提取特征矩阵。

    :param images: BGR 图像列表（已按 ``max_side`` 缩放）
    :param backend: ``features`` 或 ``cnn``
    :param arch: ``cnn`` 后端使用的骨干网络
    :param segs: 预先算好的分割结果，可为 None
    :return: ``(X, feature_names, seg_ok)``
    """
    if backend == "features":
        import concurrent.futures as cf

        n = len(images)
        seg_ok = np.zeros(n, dtype=bool)
        rows: list[np.ndarray | None] = [None] * n

        def _one(index: int) -> tuple[int, np.ndarray, bool]:
            seg = segs[index] if segs is not None else segment_leaf(images[index])
            vector, _, ok = extract_features(
                images[index], max_side=max_side, precomputed_seg=seg
            )
            return index, vector, ok

        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for index, vector, ok in pool.map(_one, range(n)):
                rows[index] = vector
                seg_ok[index] = ok
        X = np.vstack([r for r in rows if r is not None]) if n else np.zeros((0, 0), np.float32)
        return X.astype(np.float32), feature_names(), seg_ok

    if backend == "cnn":
        from .deep import extract_deep_features
        from .deep import feature_dim as _dim

        X = extract_deep_features(
            images, arch=arch, batch_size=batch_size, verbose=verbose
        )
        names = [f"deep_{i}" for i in range(_dim(arch))]
        seg_ok = np.ones(len(images), dtype=bool)
        return X, names, seg_ok

    raise ValueError(f"未知后端 {backend!r}，可选：{', '.join(BACKENDS)}")


def extract_matrix_from_paths(
    paths: list[Path | str],
    backend: str = "features",
    arch: str = "mobilenet_v3_small",
    max_side: int = 384,
    workers: int = 4,
    batch_size: int = 32,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """从文件路径直接提取特征矩阵。"""
    images = load_images(paths, max_side=max_side)
    return extract_matrix(
        images, backend=backend, arch=arch, max_side=max_side,
        workers=workers, batch_size=batch_size, verbose=verbose,
    )


__all__ = ["BACKENDS", "extract_matrix", "extract_matrix_from_paths", "load_images"]
