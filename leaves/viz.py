"""可视化工具：混淆矩阵、样本网格、预测结果拼图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt  # noqa: E402

from .species import CHINESE_NAMES, NUM_CLASSES, SCIENTIFIC_NAMES  # noqa: E402


def setup_chinese_font() -> None:
    """让 matplotlib 能正常显示中文（Windows 优先雅黑 / 黑体）。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _short_names(labels: list[int]) -> list[str]:
    return [f"{i}:{CHINESE_NAMES[i]}" for i in labels]


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[int],
    title: str = "混淆矩阵",
    output: Path | str | None = None,
    normalize: bool = True,
) -> Path | None:
    """绘制混淆矩阵热力图。"""
    setup_chinese_font()
    n = len(labels)
    size = max(9, n * 0.42)

    data = cm.astype(np.float64)
    if normalize:
        row_sum = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sum, out=np.zeros_like(data), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    names = _short_names(labels)
    ax.set_xticks(range(n), names, rotation=90, fontsize=7)
    ax.set_yticks(range(n), names, fontsize=7)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title, fontsize=13, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return output
    plt.close(fig)
    return None


def plot_samples(
    images: list[np.ndarray],
    titles: list[str],
    output: Path | str,
    cols: int = 6,
    figsize_per: float = 2.0,
) -> Path:
    """把多张小图拼成网格。``images`` 为 RGB 数组。"""
    setup_chinese_font()
    n = len(images)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_per, rows * figsize_per * 1.15)
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def plot_class_distribution(
    per_class: dict[int, int], output: Path | str, title: str = "各类别样本数"
) -> Path:
    setup_chinese_font()
    labels = sorted(per_class)
    values = [per_class[k] for k in labels]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.bar([f"{k}:{CHINESE_NAMES[k]}" for k in labels], values, color="#3b7dd8")
    ax.set_ylabel("样本数")
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def plot_per_class_accuracy(
    labels: list[int], accuracies: list[float], output: Path | str
) -> Path:
    setup_chinese_font()
    order = np.argsort(accuracies)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.barh(
        [f"{labels[i]}:{CHINESE_NAMES[labels[i]]}" for i in order],
        [accuracies[i] for i in order],
        color="#2f9e6e",
    )
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("准确率")
    ax.set_title("各类别识别准确率（由低到高）", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


__all__ = [
    "setup_chinese_font",
    "plot_confusion_matrix",
    "plot_samples",
    "plot_class_distribution",
    "plot_per_class_accuracy",
]
