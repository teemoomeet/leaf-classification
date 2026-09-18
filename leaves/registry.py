"""分类器注册表（纯标准库，供命令行参数解析等轻量场景使用）。

把注册表单独放在这里，``python -m leaves.cli download`` 之类的命令就不需要
先安装 scikit-learn 等机器学习依赖。
"""

from __future__ import annotations

#: 可用分类器名称 -> 说明
AVAILABLE_MODELS: dict[str, str] = {
    "svm": "支持向量机（RBF 核），推荐默认选项，精度最高",
    "linearsvm": "线性支持向量机，速度最快",
    "rf": "随机森林，可输出特征重要度，抗过拟合",
    "extratrees": "极端随机树，训练更快",
    "knn": "K 近邻，简单直观",
    "logreg": "逻辑回归，线性基线",
}

#: 分类器名元组，可直接用于 argparse 的 choices
MODEL_CHOICES: tuple[str, ...] = tuple(AVAILABLE_MODELS)

#: 特征后端名称 -> 说明
BACKENDS: dict[str, str] = {
    "features": "手工特征（形状 + 纹理 + 颜色）+ SVM，无需 GPU，可解释",
    "cnn": "预训练 CNN（MobileNetV3 / ResNet18 / EfficientNet-B0）深度特征 + SVM",
}

#: 支持的骨干网络 -> 特征维度
ARCHS: dict[str, int] = {
    "mobilenet_v3_small": 576,
    "resnet18": 512,
    "efficientnet_b0": 1280,
}

BACKEND_CHOICES: tuple[str, ...] = tuple(BACKENDS)
ARCH_CHOICES: tuple[str, ...] = tuple(ARCHS)

__all__ = [
    "AVAILABLE_MODELS",
    "MODEL_CHOICES",
    "BACKENDS",
    "ARCHS",
    "BACKEND_CHOICES",
    "ARCH_CHOICES",
]
