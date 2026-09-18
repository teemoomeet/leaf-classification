"""深度学习路线：用预训练 CNN 骨干网络提取深度特征，再接线性分类器。

为什么是"提取特征"而不是"端到端微调"？

* 在 CPU 上对 1907 张图做完整微调要几十分钟，而只做一次前向推理只要一两分钟；
* 冻结的 ImageNet 骨干 + SVM 在 Flavia 上已经能到 99% 左右，性价比极高；
* 特征向量缓存下来后，换分类器做实验几乎零成本。

预处理：先用 :mod:`leaves.segmentation` 把叶片抠出来，把背景填成白色，
再按 224×224 送入骨干网络，取全局池化后的特征向量。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .registry import ARCHS  # noqa: F401  (统一出口)
from .segmentation import segment_leaf

DEFAULT_ARCH = "mobilenet_v3_small"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _torch():
    """延迟导入 torch，未安装时给出清晰提示。"""
    try:
        import torch
        from torchvision import models  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "使用深度学习路线需要先安装 PyTorch：\n"
            "    pip install -r requirements-cnn.txt\n"
            "详见 README 的「两种技术路线」一节。"
        ) from exc
    return torch


def load_backbone(arch: str = DEFAULT_ARCH, device: str = "cpu"):
    """加载预训练骨干网络，返回 ``(feature_extractor, dim)``。

    返回的模型已切入 eval 模式并关闭梯度。
    """
    torch = _torch()
    from torch import nn
    from torchvision import models

    if arch not in ARCHS:
        raise ValueError(f"不支持的骨干网络 {arch!r}，可选：{', '.join(ARCHS)}")

    if arch == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights as W

        net = models.mobilenet_v3_small(weights=W.DEFAULT)
        net.classifier = nn.Identity()
    elif arch == "resnet18":
        from torchvision.models import ResNet18_Weights as W

        net = models.resnet18(weights=W.DEFAULT)
        net.fc = nn.Identity()
    else:  # efficientnet_b0
        from torchvision.models import EfficientNet_B0_Weights as W

        net = models.efficientnet_b0(weights=W.DEFAULT)
        net.classifier = nn.Identity()

    net.eval().to(device)
    for param in net.parameters():
        param.requires_grad_(False)
    return net, ARCHS[arch]


def prepare_input(image_bgr: np.ndarray, size: int = 224, pad_ratio: float = 0.08) -> np.ndarray:
    """把一张叶片图处理成骨干网络的输入张量（CHW float32，已归一化）。

    :param image_bgr: BGR 原图
    :param size: 输出边长
    :param pad_ratio: 叶片四周留白比例
    """
    seg = segment_leaf(image_bgr)
    if not seg.ok or seg.crop.size == 0:
        crop = image_bgr
        mask = np.full(image_bgr.shape[:2], 255, np.uint8)
    else:
        crop, mask = seg.crop, seg.crop_mask

    # 背景填白，让模型只关注叶片本体
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    white = np.full_like(rgb, 255)
    rgb = np.where(mask[..., None] > 0, rgb, white)

    h, w = rgb.shape[:2]
    scale = (1.0 - 2.0 * pad_ratio) * size / float(max(h, w, 1))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), 255, np.uint8)
    oy, ox = (size - new_h) // 2, (size - new_w) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized

    tensor = canvas.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(tensor, (2, 0, 1))


def extract_deep_features(
    images: list[np.ndarray],
    arch: str = DEFAULT_ARCH,
    batch_size: int = 32,
    size: int = 224,
    device: str = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """对一批 BGR 图像提取深度特征。

    :param images: BGR 图像列表（未裁剪的原图即可，内部会做分割）
    :return: ``(N, dim)`` 的 float32 特征矩阵
    """
    torch = _torch()
    if not images:
        return np.zeros((0, ARCHS.get(arch, 0)), dtype=np.float32)

    net, dim = load_backbone(arch, device=device)

    tensors: list[np.ndarray] = []
    for image in images:
        tensors.append(prepare_input(image, size=size))

    out: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            chunk = np.stack(tensors[start : start + batch_size])
            batch = torch.from_numpy(chunk).to(device)
            feats = net(batch).cpu().numpy().astype(np.float32)
            out.append(feats)
            if verbose and (start // batch_size) % 10 == 0:
                done = min(start + batch_size, len(tensors))
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tensors) - done) / rate if rate > 0 else 0
                print(
                    f"  深度特征 {done}/{len(tensors)}  "
                    f"({done / len(tensors) * 100:.1f}%)  "
                    f"{rate:.1f} 张/秒  预计剩余 {eta / 60:.1f} 分钟",
                    flush=True,
                )

    matrix = np.vstack(out) if out else np.zeros((0, dim), dtype=np.float32)
    if matrix.shape[1] != dim:
        raise RuntimeError(f"特征维度异常：期望 {dim}，实际 {matrix.shape[1]}")
    return matrix


def load_images(paths: list[Path | str]) -> list[np.ndarray]:
    """读取图片（兼容中文路径），读取失败的用白图占位。"""
    images: list[np.ndarray] = []
    for path in paths:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None:
            image = np.full((224, 224, 3), 255, np.uint8)
        images.append(image)
    return images


def feature_dim(arch: str = DEFAULT_ARCH) -> int:
    return ARCHS.get(arch, ARCHS[DEFAULT_ARCH])


__all__ = [
    "ARCHS",
    "DEFAULT_ARCH",
    "load_backbone",
    "prepare_input",
    "extract_deep_features",
    "load_images",
    "feature_dim",
]
