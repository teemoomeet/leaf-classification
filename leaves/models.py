"""分类器工厂与模型持久化。"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .config import DEFAULT_MODEL_NAME, MODEL_DIR, RANDOM_SEED, ensure_dirs
from .registry import AVAILABLE_MODELS, MODEL_CHOICES  # noqa: F401  (对外统一出口)

__all__ = [
    "AVAILABLE_MODELS",
    "MODEL_CHOICES",
    "build_model",
    "save_model",
    "load_model",
    "predict_with_confidence",
]


def _with_probability(estimator, cv: int = 5):
    """给没有 ``predict_proba`` 的分类器套一层概率校准。

    为什么不用 ``SVC(probability=True)``？

    * 它自 scikit-learn 1.9 起已被弃用，官方建议改用 :class:`CalibratedClassifierCV`；
    * 多分类 SVM 的 ``decision_function`` 走的是 one-vs-one 投票聚合，取值区间很窄，
      直接做 softmax 会得到「所有样本置信度都差不多」的结果（实测全挤在 0.63 附近），
      完全无法判断「这张图到底识别得靠不靠谱」。

    实测（Flavia 测试集 380 张）几种方案的取舍：

    ====================================  ==========  ==========================
    方案                                   测试准确率   置信度分布
    ====================================  ==========  ==========================
    原始 SVC（无概率）                      0.9789      无
    sigmoid + ensemble=False + cv=3        0.9447      中位 0.633（区分度差）
    sigmoid + ensemble=True  + cv=5        0.8684      中位 0.365（掉点严重）
    **isotonic + ensemble=True + cv=5**     0.9789      中位 0.960，5%~95% 分位
                                                       0.755~0.996（区分度好）
    ====================================  ==========  ==========================

    所以这里选 isotonic + ensemble 方案：精度不掉，置信度可用来做阈值判断。
    """
    if hasattr(estimator, "predict_proba"):
        return estimator
    try:
        from sklearn.calibration import CalibratedClassifierCV

        return CalibratedClassifierCV(estimator, method="isotonic", ensemble=True, cv=cv)
    except Exception:  # noqa: BLE001 - 版本或数据不满足时退回原估计器
        return estimator


def build_model(name: str = "svm", seed: int = RANDOM_SEED):
    """构建「标准化 + 分类器」流水线。

    :param name: 分类器名称，见 :data:`AVAILABLE_MODELS`
    :param seed: 随机种子
    """
    name = name.lower()
    if name == "svm":
        clf = _with_probability(
            SVC(
                C=10.0,
                gamma="scale",
                kernel="rbf",
                class_weight="balanced",
                random_state=seed,
            )
        )
    elif name == "linearsvm":
        clf = _with_probability(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=8000, random_state=seed)
        )
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, max_features="sqrt", n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed,
        )
    elif name == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt", n_jobs=-1,
            class_weight="balanced", random_state=seed,
        )
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    elif name == "logreg":
        clf = LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=seed,
        )
    else:
        raise ValueError(
            f"未知分类器 {name!r}，可选：{', '.join(AVAILABLE_MODELS)}"
        )

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def save_model(pipeline, meta: dict, path: Path | str | None = None) -> Path:
    """保存模型与元信息（所用分类器、特征名、物种表等）。"""
    ensure_dirs()
    path = Path(path) if path is not None else (MODEL_DIR / DEFAULT_MODEL_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "meta": meta}, path, compress=3)
    return path


def load_model(path: Path | str | None = None) -> tuple[object, dict]:
    """加载模型，返回 ``(pipeline, meta)``。"""
    path = Path(path) if path is not None else (MODEL_DIR / DEFAULT_MODEL_NAME)
    if not path.exists():
        raise FileNotFoundError(
            f"模型文件不存在：{path}\n请先运行  python -m leaves.cli train  训练模型"
        )
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle.get("meta", {})


def predict_with_confidence(pipeline, X: np.ndarray, top_k: int = 3):
    """返回 ``(预测标签, 置信度, top-k 标签, top-k 置信度)``。

    分类器带 ``predict_proba`` 时直接使用其概率输出（SVM / LinearSVC 已在
    :func:`build_model` 中做过概率校准）。极少数情况下没有概率输出，
    则退化为对 ``decision_function`` 做 softmax，置信度仅供参考。
    """
    X = np.asarray(X, dtype=np.float32)
    pred = pipeline.predict(X)

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
    else:
        scores = pipeline.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        proba = exp / exp.sum(axis=1, keepdims=True)

    classes = np.asarray(pipeline.classes_)
    k = min(top_k, proba.shape[1])
    top_idx = np.argsort(-proba, axis=1)[:, :k]
    top_labels = classes[top_idx]
    top_scores = np.take_along_axis(proba, top_idx, axis=1)

    confidence = proba.max(axis=1)
    return pred, confidence, top_labels, top_scores


__all__ = [
    "AVAILABLE_MODELS",
    "build_model",
    "save_model",
    "load_model",
    "predict_with_confidence",
]
