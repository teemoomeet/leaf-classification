# -*- coding: utf-8 -*-
"""analyze_leaves.py — 批量分析一个文件夹中的叶片图像，并校验识别正确性。

这是 leaf-classification 项目的示例脚本，演示如何在自己的代码里调用
项目 API 完成「批量识别 + 结果校验」的完整流程：

1. 扫描输入文件夹中的叶片图片；
2. 调用训练好的模型识别树种，导出 predictions.csv / predictions.json /
   可视化拼图；
3. 若文件名是 Flavia 数据集编号（如 3530.jpg），自动比对真值并输出
   判定报告（analysis_report.csv）——方便你评估模型在自己数据上的表现；
4. 非 Flavia 编号的文件会标记为「无真值」，只输出识别结果与置信度。

用法（在本仓库根目录下）：
    python examples/analyze_leaves.py --input D:/yezi
    python examples/analyze_leaves.py --input path/to/leaves --model models/flavia_cnn_mobilenetv3.joblib
    python examples/analyze_leaves.py --input path/to/leaves --output my_results

提示：
- 白底单片叶照片效果最好（与 Flavia 训练分布一致）；
- 场景实拍照建议加 --finetuned 使用微调模型，并关注 low_confidence 列。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本可以从任意工作目录运行：把仓库根目录加入 import 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出
except Exception:
    pass

from leaves.predict import PredictOptions, recognize_folder
from leaves.species import CHINESE_NAMES, label_from_filename


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="批量分析叶片文件夹：识别树种 + Flavia 编号图真值校验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="叶片图片所在文件夹")
    p.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "flavia_svm.joblib"),
        help="模型文件路径（.joblib 或 .pt）",
    )
    p.add_argument(
        "--finetuned",
        action="store_true",
        help="使用微调模型 models/flavia_ft_mobilenetv3.pt（适合场景实拍照）",
    )
    p.add_argument(
        "--output",
        default=None,
        help="结果输出目录（默认 <脚本旁>/analysis_results）",
    )
    p.add_argument("--top-k", type=int, default=3, help="输出前 K 个候选")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[错误] 目录不存在: {input_dir}")
        return 1

    model_path = (
        PROJECT_ROOT / "models" / "flavia_ft_mobilenetv3.pt"
        if args.finetuned
        else Path(args.model)
    )
    if not model_path.is_file():
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 `python -m leaves.cli train` 或 `python -m leaves.cli finetune`")
        return 1

    output_dir = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "analysis_results"
    )

    print(f"分析目录: {input_dir}（模型: {model_path.name}）\n")

    # recognize_folder 自带 predictions.csv / predictions.json / 可视化拼图导出
    frame = recognize_folder(
        input_dir,
        output_dir=output_dir / "predict",
        recursive=False,
        options=PredictOptions(model_path=model_path, top_k=args.top_k),
        verbose=False,
    )
    if frame.empty:
        print(f"[错误] {input_dir} 中没有图片")
        return 1

    # ---- 真值校验：Flavia 编号图自动判定对错 ----
    rows = []
    correct = total_with_truth = 0
    print("\n" + "-" * 60)
    for _, r in frame.iterrows():
        truth_label = label_from_filename(r["file"])
        truth = CHINESE_NAMES[truth_label] if truth_label is not None else None
        if truth is not None:
            total_with_truth += 1
            ok = int(r["pred_label"]) == truth_label
            correct += ok
            verdict = "正确" if ok else f"错误（真值: {truth}）"
        else:
            verdict = "无真值（非 Flavia 编号图）"
        rows.append({
            "文件": r["file"],
            "识别结果": r["pred_name"],
            "置信度": f"{r['confidence']:.3f}",
            "真值": truth or "-",
            "判定": verdict,
        })
        print(f"{r['file']:<12} -> {r['pred_name']}（{r['confidence']:.3f}）  {verdict}")

    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    report_csv = output_dir / "analysis_report.csv"
    pd.DataFrame(rows).to_csv(report_csv, index=False, encoding="utf-8-sig")

    if total_with_truth:
        print(f"\n正确率: {correct}/{total_with_truth} = {correct / total_with_truth:.1%}")
    print(f"判定报告已保存: {report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
