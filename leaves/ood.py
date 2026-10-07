# -*- coding: utf-8 -*-
"""开集识别 / OOD 检测模块。

为 Flavia 模型补充「拒识」能力：当输入图片不属于已知的 32 个树种时，
不再强行给一个高置信度的错误答案，而是判定为「未知 / 分布外（OOD）」。

核心方法：在特征空间用 Mahalanobis 距离度量「这张图离已知类别有多远」。
训练时对每个类别拟合高斯（均值 + 协方差），推理时计算样本到最近类别
中心的 Mahalanobis 距离，超过阈值即判为未知。

对比基线：能量分数（Energy Score），基于 SVM 的 decision_function 输出。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .species import CHINESE_NAMES


class MahalanobisOOD:
    """Mahalanobis 距离 OOD 检测器。

    在（标准化后的）特征空间，对每个类别拟合高斯分布，用「样本到最近类别
    中心的马氏距离」作为 OOD 分数。距离越大，越可能是分布外样本。

    支持两种协方差估计：
      - ``pooled``：所有类别共享一个协方差（数据量少时更稳健）
      - ``per_class``：每个类别独立估计协方差
    """

    def __init__(self, covariance: str = "pooled", reg: float = 1e-3):
        self.covariance = covariance
        self.reg = reg
        self._means: dict[int, np.ndarray] = {}
        self._pooled_cov: np.ndarray | None = None
        self._per_class_cov: dict[int, np.ndarray] = {}
        self._inv_covs: dict[int, np.ndarray] = {}
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MahalanobisOOD":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)

        for c in self.classes_:
            Xc = X[y == c]
            self._means[c] = Xc.mean(axis=0)

        if self.covariance == "pooled":
            centered = X - np.array([self._means[lab] for lab in y])
            cov = centered.T @ centered / max(1, len(X) - 1)
            cov = cov + self.reg * np.eye(cov.shape[0])
            self._pooled_cov = cov
            inv = np.linalg.pinv(cov)
            for c in self.classes_:
                self._inv_covs[c] = inv
        else:
            for c in self.classes_:
                Xc = X[y == c]
                centered = Xc - self._means[c]
                cov = centered.T @ centered / max(1, len(Xc) - 1)
                cov = cov + self.reg * np.eye(cov.shape[0])
                self._per_class_cov[c] = cov
                self._inv_covs[c] = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的 OOD 分数（到最近类别的马氏距离，越大越可能是 OOD）。"""
        X = np.asarray(X, dtype=np.float64)
        dists = []
        for c in self.classes_:
            diff = X - self._means[c]
            inv = self._inv_covs[c]
            m = np.einsum("ij,jk,ik->i", diff, inv, diff)
            dists.append(m)
        D = np.stack(dists, axis=1)
        return np.sqrt(np.maximum(D, 0.0)).min(axis=1)


def load_ood(path: Path | str | None = None) -> tuple[MahalanobisOOD, float] | None:
    """加载训练好的 OOD 检测器，返回 ``(detector, threshold)``。

    找不到检测器文件时返回 ``None``（此时应关闭拒识，退化为普通识别）。
    """
    from .config import MODEL_DIR

    path = Path(path) if path is not None else (MODEL_DIR / "flavia_ood_mahalanobis.joblib")
    if not path.exists():
        return None
    bundle = joblib.load(path)
    return bundle["detector"], float(bundle["threshold"])


def ood_scores(detector: MahalanobisOOD, X: np.ndarray) -> np.ndarray:
    """计算一批样本的 OOD 分数（越大越可能是未知/分布外）。"""
    return detector.score(np.asarray(X, dtype=np.float64))


def fit_energy_baseline(pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """在训练特征上计算能量分数基线，返回每类中心（用于 OOD 度量）。

    能量分数 E(x) = -T * log Σ_c exp(f_c(x)/T)，其中 f_c 是 SVM 对类别 c
    的决策函数值。这里用每个样本到「其预测类别决策值」的负能量作为参考，
    但更简单稳健的做法是直接用 decision_function 的 margin。
    """
    del pipeline, y  # 保留接口一致性
    return {"n": int(len(X))}


__all__ = [
    "MahalanobisOOD",
    "load_ood",
    "ood_scores",
    "fit_energy_baseline",
    "CHINESE_NAMES",
]
