# -*- coding: utf-8 -*-
"""用本地 Flavia 模型测试 Folio 数据集的准确率（跨数据集迁移实验）。

Folio（UCI #338）：32 个物种（作物/热带植物）× 每类约 20 张 = 637 张，白底拍摄。
Flavia：32 个树种（毛竹、七叶树、柑橘……），白底扫描。

两个数据集的物种完全不相同，因此本实验量化「跨数据集直接迁移」的真实表现：
  - 严格正确率（预测物种名 == 真实物种名，理论上接近 0）
  - 平均置信度、低置信度占比
  - 每个 Folio 物种最常被误判成哪些 Flavia 树种（混淆分析）

用法（在项目根目录下，用项目自带 venv）：
    python examples/cross_dataset_validation/test_folio.py --folio <Folio 根目录>

Folio 根目录应是包含若干「以物种名命名的子文件夹」的目录，例如：
    D:/folio/Folio Leaf Dataset/Folio
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# 自动定位项目根（本文件位于 examples/cross_dataset_validation/ 下）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from leaves.predict import PredictOptions, recognize  # noqa: E402
from leaves.species import CHINESE_NAMES, SCIENTIFIC_NAMES  # noqa: E402

# --------------------------------------------------------------------------- #
# Folio 物种清单
# --------------------------------------------------------------------------- #
# Folio 32 物种（官方顺序），用于把文件夹名映射成标签
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

# 中文名（便于阅读），与 FOLIO_SPECIES 一一对应
FOLIO_CN: list[str] = [
    "秘鲁木", "茄子", "果实类", "番石榴", "木槿", "蒌叶", "玫瑰", "菊花",
    "榕树", "黄金假连翘", "阿散蒂血木", "酸橙", "玉叶金花", "菠萝蜜",
    "桑叶", "多香果", "紫果木", "星苹果", "巴巴多斯樱桃", "桂花",
    "变叶木", "黄花夹竹桃", "老男孩", "可可树", "彩叶草", "咖啡",
    "锡兰醋栗", "中国番石榴", "荔枝", "天竺葵", "红薯", "番木瓜",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_FOLDER_ALIASES: dict[str, str] = {
    "beaumier du perou": "Beaumier du perou",
    "eggplant": "Eggplant",
    "aubergine": "Eggplant",
    "fruitcitere": "Fruitcitere",
    "guava": "Guava",
    "hibiscus": "Hibiscus",
    "betel": "Betel",
    "rose": "Rose",
    "chrysanthemum": "Chrysanthemum",
    "ficus": "Ficus",
    "duranta gold": "Duranta gold",
    "ashanti blood": "Ashanti blood",
    "bitter orange": "Bitter Orange",
    "coeur demoiselle": "Coeur Demoiselle",
    "jackfruit": "Jackfruit",
    "mulberry leaf": "Mulberry Leaf",
    "mulberry": "Mulberry Leaf",
    "pimento": "Pimento",
    "pomme jacquot": "Pomme Jacquot",
    "star apple": "Star Apple",
    "barbados cherry": "Barbados Cherry",
    "sweet olive": "Sweet Olive",
    "croton": "Croton",
    "thevetia": "Thevetia",
    "vieux garcon": "Vieux Garcon",
    "chocolate tree": "Chocolate tree",
    "caricature plant": "Carricature plant",
    "carricature plant": "Carricature plant",
    "coffee": "Coffee",
    "ketembilla": "Ketembilla",
    "chinese guava": "Chinese guava",
    "lychee": "Lychee",
    "geranium": "Geranium",
    "sweet potato": "Sweet potato",
    "papaya": "Papaya",
}


def resolve_folio_label(folder_name: str) -> int | None:
    key = folder_name.strip().lower()
    if key in _FOLDER_ALIASES:
        return FOLIO_SPECIES.index(_FOLDER_ALIASES[key])
    for name in FOLIO_SPECIES:
        if name.lower() == key or name.lower() in key or key in name.lower():
            return FOLIO_SPECIES.index(name)
    return None


def collect_images(folio_root: Path) -> list[tuple[str, int]]:
    """返回 [(图片路径, folio标签)]。"""
    items: list[tuple[str, int]] = []
    if not folio_root.exists():
        raise FileNotFoundError(f"找不到 Folio 数据根目录：{folio_root}")
    for sub in sorted(folio_root.iterdir()):
        if not sub.is_dir():
            continue
        label = resolve_folio_label(sub.name)
        if label is None:
            continue
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                items.append((str(f), label))
    return items


def run_model(model_name: str, model_path: str, items, out_dir: Path):
    print("\n" + "=" * 74)
    print(f" 模型：{model_name}")
    print("=" * 74)

    image_paths = [p for p, _ in items]
    true_labels = np.array([lab for _, lab in items])

    frame = recognize(
        image_paths,
        PredictOptions(model_path=model_path, top_k=3, save_visualization=False),
        verbose=False,
    )
    pred_labels = frame["pred_label"].to_numpy(dtype=int)
    confidences = frame["confidence"].to_numpy(dtype=float)

    n = len(true_labels)

    # 1) 严格正确率：预测 Flavia 物种（中文名/学名）与 Folio 真实物种名是否相同
    strict_correct = 0
    for i, lab in enumerate(true_labels):
        folio_name = FOLIO_SPECIES[lab].lower()
        pred_sci = SCIENTIFIC_NAMES[pred_labels[i]].lower()
        pred_cn = CHINESE_NAMES[pred_labels[i]].lower()
        if (folio_name in pred_sci or folio_name in pred_cn
                or pred_sci in folio_name or pred_cn in folio_name):
            strict_correct += 1

    # 2) 置信度统计
    avg_conf = float(confidences.mean())
    low_conf_ratio = float((confidences < 0.35).mean())
    median_conf = float(np.median(confidences))

    print(f"\n[整体] 样本数：{n}")
    print(f"[整体] 平均置信度：{avg_conf:.3f}（中位数 {median_conf:.3f}）")
    print(f"[整体] 低置信度(<0.35) 占比：{low_conf_ratio:.1%}")
    print(f"[整体] 严格正确率：{strict_correct}/{n} = {strict_correct/n:.2%}")
    print(f"[整体] 预测类别数：{len(set(pred_labels))} / 32（模型把图都归到了哪几类）")

    # 3) 混淆统计
    rows = []
    for lab in sorted(set(true_labels)):
        mask = true_labels == lab
        preds = pred_labels[mask]
        top = Counter(preds).most_common(3)
        top_str = ", ".join(f"{CHINESE_NAMES[p]}({c})" for p, c in top)
        rows.append({
            "Folio物种": FOLIO_SPECIES[lab],
            "中文名": FOLIO_CN[lab],
            "样本数": int(mask.sum()),
            "Top3误判(Flavia树种)": top_str,
            "平均置信度": round(float(confidences[mask].mean()), 3),
        })
    detail = pd.DataFrame(rows)
    print("\n[混淆统计] 每个 Folio 物种 → 最常被误判成的 Flavia 树种：")
    print(detail.to_string(index=False))

    # 4) 预测分布
    print("\n[预测分布] Flavia 模型对整批 Folio 图的 Top-1 输出（前 15）：")
    for name, cnt in Counter(CHINESE_NAMES[p] for p in pred_labels).most_common(15):
        print(f"  {name:<12} {cnt:>4} 张 ({cnt/n:.1%})")

    # 导出
    safe = "svm" if "svm" in model_path else "cnn"
    detail.to_csv(out_dir / f"folio_{safe}_confusion.csv",
                  index=False, encoding="utf-8-sig")
    full = frame.copy()
    full.insert(0, "folio_true", [FOLIO_SPECIES[l] for l in true_labels])
    full.to_csv(out_dir / f"folio_{safe}_predictions.csv",
                index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out_dir}")

    return {
        "model": model_name,
        "n": n,
        "avg_conf": avg_conf,
        "median_conf": median_conf,
        "low_conf_ratio": low_conf_ratio,
        "strict_acc": strict_correct / n,
        "n_pred_classes": len(set(pred_labels)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="跨数据集迁移实验：Flavia 模型 → Folio")
    parser.add_argument("--folio", required=True, help="Folio 数据根目录（含物种子文件夹）")
    args = parser.parse_args()

    print("=" * 74)
    print(" Flavia 模型  →  Folio 数据集  跨数据集迁移实验")
    print("=" * 74)

    folio_root = Path(args.folio)
    out_dir = Path(__file__).resolve().parent
    items = collect_images(folio_root)
    print(f"\nFolio 数据根目录：{folio_root}")
    print(f"解析到 {len(items)} 张图片，覆盖 {len(set(l for _, l in items))} 个物种")

    dist = Counter(l for _, l in items)
    for lab in sorted(dist):
        print(f"  [{lab:>2}] {FOLIO_SPECIES[lab]:<20} {FOLIO_CN[lab]:<8} {dist[lab]} 张")

    results = []
    results.append(run_model(
        "手工特征 + SVM", str(PROJECT_ROOT / "models" / "flavia_svm.joblib"),
        items, out_dir))
    results.append(run_model(
        "深度特征(CNN) + SVM", str(PROJECT_ROOT / "models" / "flavia_cnn_mobilenetv3.joblib"),
        items, out_dir))

    # 汇总
    print("\n" + "=" * 74)
    print(" 汇总对比")
    print("=" * 74)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "folio_summary.csv", index=False, encoding="utf-8-sig")

    print("\n说明：Folio 的 32 物种与 Flavia 的 32 树种完全不同，")
    print("      严格正确率接近 0 是预期结果，不代表模型有 bug。")
    print("      关键观察指标是「置信度是否普遍偏低」，若低则说明模型能")
    print("      识别出这些图不属于训练分布（即模型有良好的 OOD 警觉性）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
