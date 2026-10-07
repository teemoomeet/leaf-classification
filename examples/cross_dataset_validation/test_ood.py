# -*- coding: utf-8 -*-
"""方案 A 验证：为 Flavia 模型加入开集识别（OOD 拒识）能力后的效果测试。

测试目标：
  1. 已知类（Flavia 测试集，380 张）：模型应大部分「接受」并给出正确物种，
     拒识率应很低（≈5% 校准阈值）。
  2. 未知类（Folio，637 张）：模型应大部分「拒识」，说"我不认识"，
     而不是像之前那样硬给一个高置信度的错误答案。

方法：
  - Mahalanobis 距离 OOD 检测器（在特征空间拟合每类高斯）
  - 用 Flavia 验证集校准阈值，使已知类通过率 ≈ 95%
  - 对比「不加拒识」与「加拒识」两种口径下的表现

前置：需先在项目根目录运行训练脚本生成 OOD 检测器与特征缓存：
    python leaves/train_ood.py

用法（在项目根目录下，用项目 venv）：
    python examples/cross_dataset_validation/test_ood.py --folio <Folio 根目录>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 自动定位项目根（本文件位于 examples/cross_dataset_validation/ 下）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from leaves.ood import MahalanobisOOD  # noqa: E402
from leaves.predict import PredictOptions, recognize  # noqa: E402

FOLIO_ROOT = PROJECT_ROOT.parent / "folio"  # 占位，实际由 --folio 参数覆盖
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "flavia_svm.joblib"
FEATURE_CACHE = PROJECT_ROOT / "data" / "features_handcrafted.npz"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "split.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

FOLIO_SPECIES: list[str] = [
    "Beaumier du perou", "Eggplant", "Fruitcitere", "Guava", "Hibiscus",
    "Betel", "Rose", "Chrysanthemum", "Ficus", "Duranta gold",
    "Ashanti blood", "Bitter Orange", "Coeur Demoiselle", "Jackfruit",
    "Mulberry Leaf", "Pimento", "Pomme Jacquot", "Star Apple",
    "Barbados Cherry", "Sweet Olive", "Croton", "Thevetia",
    "Vieux Garcon", "Chocolate tree", "Carricature plant", "Coffee",
    "Ketembilla", "Chinese guava", "Lychee", "Geranium",
    "Sweet potato", "Papaya",
]
_FOLDER_ALIASES: dict[str, str] = {
    "beaumier du perou": "Beaumier du perou", "eggplant": "Eggplant",
    "fruitcitere": "Fruitcitere", "guava": "Guava", "hibiscus": "Hibiscus",
    "betel": "Betel", "rose": "Rose", "chrysanthemum": "Chrysanthemum",
    "ficus": "Ficus", "duranta gold": "Duranta gold",
    "ashanti blood": "Ashanti blood", "bitter orange": "Bitter Orange",
    "coeur demoiselle": "Coeur Demoiselle", "jackfruit": "Jackfruit",
    "mulberry leaf": "Mulberry Leaf", "pimento": "Pimento",
    "pomme jacquot": "Pomme Jacquot", "star apple": "Star Apple",
    "barbados cherry": "Barbados Cherry", "sweet olive": "Sweet Olive",
    "croton": "Croton", "thevetia": "Thevetia", "vieux garcon": "Vieux Garcon",
    "chocolate tree": "Chocolate tree", "caricature plant": "Carricature plant",
    "carricature plant": "Carricature plant", "coffee": "Coffee",
    "ketembilla": "Ketembilla", "chinese guava": "Chinese guava",
    "lychee": "Lychee", "geranium": "Geranium", "sweet potato": "Sweet potato",
    "papaya": "Papaya",
}


def resolve_label(folder_name: str) -> int | None:
    key = folder_name.strip().lower()
    if key in _FOLDER_ALIASES:
        return FOLIO_SPECIES.index(_FOLDER_ALIASES[key])
    for name in FOLIO_SPECIES:
        if name.lower() == key or name.lower() in key or key in name.lower():
            return FOLIO_SPECIES.index(name)
    return None


def collect_folio(folio_root: Path) -> list[str]:
    paths: list[str] = []
    for sub in sorted(folio_root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(str(f))
    return paths


def extract_features_for_paths(paths: list[str], backend: str, max_side: int,
                               whiten: bool) -> np.ndarray:
    """现场提取特征（与模型训练时一致，这里主要用于 Folio 测试图）。"""
    from leaves.backends import extract_matrix_from_paths
    X, _, _ = extract_matrix_from_paths(
        paths, backend=backend, arch="mobilenet_v3_small",
        max_side=max_side, workers=4, batch_size=32, verbose=False, whiten=whiten,
    )
    return X.astype(np.float32)


def main() -> int:
    import joblib

    parser = argparse.ArgumentParser(description="方案 A：OOD 拒识效果验证")
    parser.add_argument("--folio", required=True, help="Folio 数据根目录（含物种子文件夹）")
    args = parser.parse_args()
    folio_root = Path(args.folio)

    print("=" * 74)
    print(" 方案 A 验证：Mahalanobis 距离 OOD 拒识")
    print("=" * 74)

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    meta = bundle["meta"]
    backend = meta["backend"]
    max_side = meta["max_side"]
    print(f"模型后端: {backend}  特征维度: {meta['n_features']}  最大边长: {max_side}")

    # ------------------------------------------------------------------ #
    # 1. 加载 Flavia 特征缓存 + 划分，构建 train/val/test 特征
    # ------------------------------------------------------------------ #
    cache = np.load(FEATURE_CACHE, allow_pickle=True)
    cache_paths = [str(p) for p in cache["paths"]]
    cache_X = cache["X"].astype(np.float32)
    cache_y = cache["y"].astype(np.int64)
    idx = {p: i for i, p in enumerate(cache_paths)}

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    def _slice(group):
        ids = [idx[rec["path"]] for rec in group]
        return cache_X[ids], cache_y[ids]

    X_tr, y_tr = _slice(split["train"])
    X_va, y_va = _slice(split["val"])
    X_te, y_te = _slice(split["test"])
    print(f"\nFlavia 特征：train {len(y_tr)} / val {len(y_va)} / test {len(y_te)}")

    # ------------------------------------------------------------------ #
    # 2. 拟合 OOD 检测器（用 train 集）
    # ------------------------------------------------------------------ #
    print("\n[拟合] Mahalanobis OOD 检测器（pooled 协方差）...")
    ood = MahalanobisOOD(covariance="pooled").fit(X_tr, y_tr)
    print("  完成，类别数:", len(ood.classes_))

    # ------------------------------------------------------------------ #
    # 3. 用 val 集校准阈值：使已知类通过率 ≈ 95%
    # ------------------------------------------------------------------ #
    va_scores = ood.score(X_va)
    threshold = float(np.percentile(va_scores, 95))
    print(f"\n[校准] val 集 OOD 分数分位数：")
    print(f"  50% = {np.percentile(va_scores, 50):.3f}")
    print(f"  90% = {np.percentile(va_scores, 90):.3f}")
    print(f"  95% = {threshold:.3f}  <- 阈值（已知类 95% 通过）")
    print(f"  99% = {np.percentile(va_scores, 99):.3f}")

    # ------------------------------------------------------------------ #
    # 4. 在 Flavia test（已知类）上验证
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print(" [已知类] Flavia 测试集（380 张）")
    print("=" * 74)
    te_scores = ood.score(X_te)
    te_pass = te_scores <= threshold
    print(f"  通过率（被接受）: {te_pass.mean():.1%}")
    print(f"  误拒率（被错判为未知）: {(~te_pass).mean():.1%}")

    pred = pipeline.predict(X_te)
    accepted_correct = (pred == y_te) & te_pass
    print(f"  被接受样本中的识别正确率: {accepted_correct.sum()}/{te_pass.sum()} "
          f"= {accepted_correct.sum()/max(1,te_pass.sum()):.1%}")

    # ------------------------------------------------------------------ #
    # 5. 在 Folio（未知类）上验证
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print(" [未知类] Folio 数据集")
    print("=" * 74)
    folio_paths = collect_folio(folio_root)
    print(f"  收集到 {len(folio_paths)} 张图片")

    print("  提取 Folio 特征（与训练一致，whiten=False）...")
    X_folio = extract_features_for_paths(folio_paths, backend, max_side, whiten=False)
    fo_scores = ood.score(X_folio)
    fo_reject = fo_scores > threshold
    print(f"  拒识率（正确识别为「未知」）: {fo_reject.mean():.1%}")
    print(f"  漏判率（仍被误当成已知树种）: {(~fo_reject).mean():.1%}")

    # 对比：不加拒识时，Folio 的平均置信度（复现之前的"过度自信"问题）
    print("\n  对比 —— Folio 在「不加拒识」时的表现（复现之前结论）：")
    frame = recognize(
        folio_paths,
        PredictOptions(model_path=str(MODEL_PATH), top_k=1, save_visualization=False),
        verbose=False,
    )
    conf = frame["confidence"].to_numpy(dtype=float)
    print(f"    平均置信度: {conf.mean():.3f}  低置信度(<0.35)占比: {(conf<0.35).mean():.1%}")
    print(f"    （即：模型此前会给这些未知图 0.6+ 的高置信度错误答案）")

    print("\n  OOD 分数分布对比：")
    for name, scores in [("Flavia已知类(val)", va_scores), ("Folio未知类", fo_scores)]:
        print(f"    {name:<18} 均值 {scores.mean():.3f}  "
              f"中位 {np.median(scores):.3f}  "
              f"90%分位 {np.percentile(scores,90):.3f}  "
              f"超阈值占比 {(scores>threshold).mean():.1%}")

    # ------------------------------------------------------------------ #
    # 6. 按 Folio 物种分组统计拒识率
    # ------------------------------------------------------------------ #
    print("\n[分组] 各 Folio 物种的拒识率：")
    rows = []
    fo_labels = []
    for sub in sorted(folio_root.iterdir()):
        if not sub.is_dir():
            continue
        lab = resolve_label(sub.name)
        if lab is None:
            continue
        sub_files = [str(f) for f in sorted(sub.rglob("*"))
                     if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
        for f in sub_files:
            fo_labels.append((f, lab))
    fo_path_set = {p: i for i, p in enumerate(folio_paths)}
    by_lab = {}
    for p, lab in fo_labels:
        by_lab.setdefault(lab, []).append(fo_path_set[p])
    for lab in sorted(by_lab):
        ids = by_lab[lab]
        rej = fo_reject[ids].mean()
        rows.append({"Folio物种": FOLIO_SPECIES[lab], "样本数": len(ids),
                     "拒识率": round(float(rej), 3)})
    detail = pd.DataFrame(rows)
    print(detail.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 7. 汇总 + 导出
    # ------------------------------------------------------------------ #
    summary = pd.DataFrame([
        {"指标": "已知类通过率(Flavia test)", "数值": f"{te_pass.mean():.1%}"},
        {"指标": "已知类误拒率", "数值": f"{(~te_pass).mean():.1%}"},
        {"指标": "未知类拒识率(Folio)", "数值": f"{fo_reject.mean():.1%}"},
        {"指标": "未知类漏判率", "数值": f"{(~fo_reject).mean():.1%}"},
        {"指标": "OOD阈值", "数值": f"{threshold:.3f}"},
    ])
    detail.to_csv(OUT_DIR / "ood_folio_by_species.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "ood_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 74)
    print(" 结果已保存到", OUT_DIR)
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
