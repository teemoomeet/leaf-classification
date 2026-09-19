"""微调后的 torch 分类模型推理封装（``.pt`` 模型文件）。

与 :mod:`leaves.models` 的 joblib 管线并列：``.joblib`` = 冻结特征 +
传统分类器，``.pt`` = 端到端微调网络（见 :mod:`leaves.finetune`）。
对调用方都输出校准过的概率与 Top-K 候选。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .deep import IMAGENET_MEAN, IMAGENET_STD, _torch
from .species import NUM_CLASSES


class TorchLeafClassifier:
    """加载 ``flavia_ft_*.pt`` 并做批量预测。"""

    def __init__(self, net, size: int, device: str = "cpu") -> None:
        self.net = net
        self.size = size
        self.device = device

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "TorchLeafClassifier":
        torch = _torch()
        import torch.nn as nn
        from torchvision import models

        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if payload.get("format") != "leaves-finetune-v1":
            raise ValueError(f"不是本项目微调模型格式：{path}")
        arch = payload.get("arch", "mobilenet_v3_small")
        if arch != "mobilenet_v3_small":
            raise ValueError(f"暂不支持的微调骨干：{arch}")
        net = models.mobilenet_v3_small(weights=None)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features,
                                      int(payload.get("num_classes", NUM_CLASSES)))
        net.load_state_dict(payload["state_dict"])
        net.eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)
        return cls(net, int(payload.get("size", 160)), device)

    def _tensor(self, image_bgr: np.ndarray) -> np.ndarray:
        """等比缩放到白底方形画布（与训练时的扫描视图协议一致）。"""
        h, w = image_bgr.shape[:2]
        scale = (self.size * 0.96) / float(max(h, w))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.size, self.size, 3), 255, np.uint8)
        oy, ox = (self.size - new_h) // 2, (self.size - new_w) // 2
        canvas[oy : oy + new_h, ox : ox + new_w] = resized
        t = canvas.astype(np.float32) / 255.0
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(t, (2, 0, 1))

    def predict_proba(self, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """输入 BGR 图像列表（已白底化），返回 ``(N, num_classes)`` 概率矩阵。"""
        torch = _torch()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                chunk = np.stack(
                    [self._tensor(im) for im in images[start : start + batch_size]]
                )
                logits = self.net(torch.from_numpy(chunk).to(self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                out.append(probs)
        return np.vstack(out) if out else np.zeros((0, NUM_CLASSES), np.float32)

    def topk(self, probs_row: np.ndarray, k: int = 3):
        order = np.argsort(-probs_row)[: max(1, k)]
        return order, probs_row[order]


__all__ = ["TorchLeafClassifier"]
