# -*- coding: utf-8 -*-
"""训练并保存 Flavia 模型的 OOD 拒识检测器。

用法（在 D:\\leaves 下）：
    .\\.venv\\Scripts\\python.exe leaves\\train_ood.py

产物：models/flavia_ood_mahalanobis.joblib
  {
    "detector": MahalanobisOOD,
    "threshold": float,       # val 集 95% 分位校准出的阈值
    "backend": "features",
    "max_side": 384,
    "model_path": "models/flavia_svm.joblib",
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaves.ood import MahalanobisOOD  # noqa: E402


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    cache = np.load(project / "data" / "features_handcrafted.npz", allow_pickle=True)
    paths = [str(p) for p in cache["paths"]]
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    idx = {p: i for i, p in enumerate(paths)}

    split = json.loads((project / "data" / "splits" / "split.json").read_text("utf-8"))
    tr_ids = [idx[r["path"]] for r in split["train"]]
    va_ids = [idx[r["path"]] for r in split["val"]]

    detector = MahalanobisOOD(covariance="pooled").fit(X[tr_ids], y[tr_ids])
    threshold = float(np.percentile(detector.score(X[va_ids]), 95))

    out = project / "models" / "flavia_ood_mahalanobis.joblib"
    joblib.dump({
        "detector": detector,
        "threshold": threshold,
        "backend": "features",
        "max_side": 384,
        "model_path": str(project / "models" / "flavia_svm.joblib"),
    }, out, compress=3)
    print(f"OOD 检测器已保存到 {out}")
    print(f"阈值（已知类 95% 通过）: {threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
