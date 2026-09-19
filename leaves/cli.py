"""命令行入口。

::

    python -m leaves.cli download      # 下载 Flavia 数据集
    python -m leaves.cli info          # 查看数据集概况
    python -m leaves.cli train         # 训练模型
    python -m leaves.cli evaluate      # 评估模型
    python -m leaves.cli predict -i DIR  # 批量识别树叶图片
    python -m leaves.cli demo          # 一键跑通全流程
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_MODEL_NAME, OUTPUT_DIR, RAW_DIR, ensure_dirs
from .registry import ARCH_CHOICES, BACKEND_CHOICES, MODEL_CHOICES


def _cmd_download(args: argparse.Namespace) -> int:
    from . import download

    download.run(source=args.source, force=args.force, workers=args.workers)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from .dataset import class_distribution, scan_dataset
    from .species import CHINESE_NAMES, SCIENTIFIC_NAMES, SPECIES
    from .viz import plot_class_distribution

    data_dir = Path(args.data) if args.data else RAW_DIR
    print("=" * 68)
    print(" Flavia 数据集概况")
    print("=" * 68)
    print(f"物种总数: {len(SPECIES)}")
    print(f"{'标签':<5}{'中文名':<12}{'拉丁学名':<32}{'别名'}")
    for i, (sci, cn, alias) in enumerate(SPECIES):
        print(f"{i:<5}{cn:<12}{sci:<32}{alias}")

    try:
        records = scan_dataset(data_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[提示] 尚未准备好数据（{exc}）")
        return 0

    dist = class_distribution(records)
    print(f"\n数据目录: {data_dir}")
    print(f"图片总数: {len(records)}，已覆盖 {len(dist)} 个物种")
    print(f"每类样本数: 最少 {min(dist.values())}，最多 {max(dist.values())}")
    if args.plot:
        out = plot_class_distribution(dist, OUTPUT_DIR / "class_distribution.png")
        print(f"分布图已保存: {out}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    from .dataset import save_features, scan_dataset
    from .train import compute_features

    records = scan_dataset(Path(args.data) if args.data else RAW_DIR)
    print(f"提取 {len(records)} 张图像的特征 ...")
    X, y, paths, names, seg_ok = compute_features(
        records, max_side=args.max_side, workers=args.workers
    )
    out = save_features(paths, X, y, names)
    print(f"特征矩阵 {X.shape}，已保存到 {out}")
    print(f"分割失败 {int((~seg_ok).sum())} 张")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import run

    run(
        data_dir=Path(args.data) if args.data else None,
        model_name=args.classifier,
        backend=args.backend,
        arch=args.arch,
        max_side=args.max_side,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        workers=args.workers,
        batch_size=args.batch_size,
        model_path=Path(args.model) if args.model else None,
        cv_folds=args.cv,
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import run

    run(
        model_path=Path(args.model) if args.model else None,
        data_dir=Path(args.data) if args.data else None,
        split=args.split,
        output_dir=Path(args.output) if args.output else None,
        top_k=args.top_k,
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from .predict import PredictOptions, recognize_folder

    options = PredictOptions(
        model_path=Path(args.model) if args.model else None,
        top_k=args.top_k,
        low_confidence=args.threshold,
        save_visualization=not args.no_vis,
        max_visualized=args.max_vis,
    )
    recognize_folder(
        args.input,
        output_dir=Path(args.output) if args.output else None,
        recursive=not args.no_recursive,
        options=options,
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """一键跑通：检查数据 -> 训练 -> 评估 -> 批量识别演示。"""
    from . import download
    from .dataset import load_split
    from .evaluate import run as evaluate_run
    from .predict import PredictOptions, recognize
    from .train import run as train_run

    print(">>> 步骤 1/4：准备数据")
    download.run(source=args.source, workers=args.workers, force=args.force)

    print("\n>>> 步骤 2/4：训练模型")
    model_path = train_run(
        model_name=args.classifier,
        backend=args.backend,
        arch=args.arch,
        max_side=args.max_side,
        workers=args.workers,
        batch_size=args.batch_size,
        cv_folds=args.cv,
    )

    print("\n>>> 步骤 3/4：评估模型")
    evaluate_run(model_path=model_path, split="test")

    print("\n>>> 步骤 4/4：批量识别演示（从测试集里挑 12 个不同树种）")
    splits = load_split()
    demo_images: list[str] = []
    seen_labels: set[int] = set()
    for rec in splits["test"]:
        if rec.label in seen_labels:
            continue
        seen_labels.add(rec.label)
        demo_images.append(rec.path)
        if len(demo_images) >= 12:
            break
    if len(demo_images) < 12:  # 类别不足时用剩余样本补齐
        demo_images = [rec.path for rec in splits["test"][:12]]

    frame = recognize(
        demo_images,
        PredictOptions(model_path=model_path, save_visualization=True,
                       max_visualized=12),
    )
    ensure_dirs()
    demo_csv = OUTPUT_DIR / "demo_predictions.csv"
    frame.to_csv(demo_csv, index=False, encoding="utf-8-sig")
    print(f"\n演示结果已保存：{demo_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leaves",
        description="Flavia 树叶图像批量识别：识别图片中的树叶属于哪一种树",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m leaves.cli download\n"
            "  python -m leaves.cli train --classifier svm\n"
            "  python -m leaves.cli predict --input D:/my_leaves\n"
            "  python -m leaves.cli demo\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="下载 Flavia 数据集（1907 张，32 个物种）")
    p_dl.add_argument("--source", choices=["auto", "github", "sourceforge"],
                      default="auto", help="下载源，默认自动尝试")
    p_dl.add_argument("--force", action="store_true", help="强制重新下载")
    p_dl.add_argument("--workers", type=int, default=16, help="并行下载线程数，默认 16")
    p_dl.set_defaults(func=_cmd_download)

    p_info = sub.add_parser("info", help="查看物种表与数据集概况")
    p_info.add_argument("--data", default=None, help="数据目录，默认 data/raw")
    p_info.add_argument("--plot", action="store_true", help="导出类别分布图")
    p_info.set_defaults(func=_cmd_info)

    p_feat = sub.add_parser("features", help="仅提取特征并缓存（调试用）")
    p_feat.add_argument("--data", default=None)
    p_feat.add_argument("--max-side", type=int, default=384)
    p_feat.add_argument("--workers", type=int, default=4)
    p_feat.set_defaults(func=_cmd_features)

    p_train = sub.add_parser("train", help="训练分类模型")
    p_train.add_argument("--classifier", "-c", default="svm", choices=MODEL_CHOICES,
                         help="分类器，默认 svm")
    p_train.add_argument("--backend", "-b", default="features", choices=BACKEND_CHOICES,
                         help="特征后端：features=手工特征（默认，无需 GPU），cnn=深度特征")
    p_train.add_argument("--arch", default="mobilenet_v3_small", choices=ARCH_CHOICES,
                         help="--backend cnn 时使用的骨干网络")
    p_train.add_argument("--data", default=None, help="数据目录，默认 data/raw")
    p_train.add_argument("--model", default=None, help=f"模型保存路径，默认 models/{DEFAULT_MODEL_NAME}")
    p_train.add_argument("--max-side", type=int, default=384, help="特征提取时图像最长边")
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--val-size", type=float, default=0.1)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--workers", type=int, default=4, help="特征提取线程数")
    p_train.add_argument("--batch-size", type=int, default=32, help="深度特征批大小")
    p_train.add_argument("--cv", type=int, default=0, help="大于 1 时做 K 折交叉验证")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="在测试集上评估模型")
    p_eval.add_argument("--model", default=None)
    p_eval.add_argument("--data", default=None, help="指定后重新划分，否则用训练时的划分")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--output", default=None)
    p_eval.add_argument("--top-k", type=int, default=3)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_pred = sub.add_parser("predict", help="批量识别树叶图片（核心功能）")
    p_pred.add_argument("--input", "-i", required=True, help="待识别的图片目录或单张图片")
    p_pred.add_argument("--model", default=None)
    p_pred.add_argument("--output", "-o", default=None, help="结果输出目录")
    p_pred.add_argument("--top-k", type=int, default=3)
    p_pred.add_argument("--threshold", type=float, default=0.35,
                        help="置信度低于该值标记为低置信度，默认 0.35")
    p_pred.add_argument("--no-vis", action="store_true", help="不导出可视化拼图")
    p_pred.add_argument("--max-vis", type=int, default=60, help="可视化最多张数")
    p_pred.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    p_pred.set_defaults(func=_cmd_predict)

    p_demo = sub.add_parser("demo", help="一键跑通：下载 -> 训练 -> 评估 -> 识别")
    p_demo.add_argument("--source", choices=["auto", "github", "sourceforge"], default="auto")
    p_demo.add_argument("--classifier", "-c", default="svm", choices=MODEL_CHOICES)
    p_demo.add_argument("--backend", "-b", default="features", choices=BACKEND_CHOICES)
    p_demo.add_argument("--arch", default="mobilenet_v3_small", choices=ARCH_CHOICES)
    p_demo.add_argument("--max-side", type=int, default=384)
    p_demo.add_argument("--workers", type=int, default=4)
    p_demo.add_argument("--batch-size", type=int, default=32)
    p_demo.add_argument("--cv", type=int, default=0)
    p_demo.add_argument("--force", action="store_true", help="强制重新下载数据集")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def _ensure_utf8_stdio() -> None:
    """让 stdout/stderr 以 UTF-8 输出，避免 Windows GBK 控制台遇到 ⚠/中文 等字符崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\n[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
