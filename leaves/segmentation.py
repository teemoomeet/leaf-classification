"""叶片分割：把叶片从背景中抠出来。

Flavia 数据集的图像是"白底叶片"，分割相对简单，但不同副本的背景亮度差异较大
（纯白扫描件 / 灰底照片），因此这里用「边缘环亮度 + Otsu」自适应判断前景极性，
再做形态学清理、取最大连通域、填充孔洞，最后按外接矩形紧致裁剪。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentedLeaf:
    """分割结果。"""

    mask: np.ndarray        # 二值掩码，叶片=255，背景=0，与 crop 同尺寸
    crop: np.ndarray        # 紧致裁剪后的原图（BGR）
    crop_mask: np.ndarray   # 紧致裁剪后的掩码
    bbox: tuple[int, int, int, int]  # 原图中的 (x, y, w, h)
    area_ratio: float       # 叶片面积 / 原图面积
    ok: bool                # 是否成功分割出叶片


def _border_is_bright(gray: np.ndarray, margin_ratio: float = 0.04) -> bool:
    """判断背景是亮色还是暗色（看图像最外圈像素的平均亮度）。"""
    h, w = gray.shape[:2]
    my = max(1, int(h * margin_ratio))
    mx = max(1, int(w * margin_ratio))
    border = np.concatenate(
        [
            gray[:my, :].ravel(),
            gray[-my:, :].ravel(),
            gray[:, :mx].ravel(),
            gray[:, -mx:].ravel(),
        ]
    )
    return float(np.mean(border)) >= 127.0


def _otsu_foreground(gray: np.ndarray) -> np.ndarray:
    """Otsu 阈值分割，自动判断叶片比背景亮还是暗，返回前景掩码。"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 背景亮 -> 叶片是暗的 -> 用 THRESH_BINARY_INV 的结果
    if _border_is_bright(gray):
        foreground = dark
    else:
        foreground = cv2.bitwise_not(dark)
    return foreground


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """保留面积最大的连通域，其余置 0。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return np.zeros_like(binary)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """填充掩码内部的孔洞。"""
    h, w = mask.shape[:2]
    flood = mask.copy()
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, canvas, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def segment_leaf(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.995,
    close_kernel: int = 7,
) -> SegmentedLeaf:
    """分割出单张图像中的叶片。

    :param image_bgr: BGR 原图
    :param min_area_ratio: 叶片面积占比下限，低于该值认为分割失败
    :param max_area_ratio: 叶片面积占比上限，高于该值认为「叶片撑满画面」而失败
    :param close_kernel: 形态学闭运算核大小
    :return: :class:`SegmentedLeaf`
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("输入图像为空")

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    foreground = _otsu_foreground(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)

    mask = _largest_component(foreground)
    mask = _fill_holes(mask)

    total = float(h * w)
    area = float(np.count_nonzero(mask))
    area_ratio = area / total

    ok = min_area_ratio <= area_ratio <= max_area_ratio
    if area > 0:
        x, y, bw, bh = cv2.boundingRect(mask)
    else:
        x = y = 0
        bw, bh = w, h

    crop = image_bgr[y : y + bh, x : x + bw].copy()
    crop_mask = mask[y : y + bh, x : x + bw].copy()
    return SegmentedLeaf(
        mask=mask,
        crop=crop,
        crop_mask=crop_mask,
        bbox=(x, y, bw, bh),
        area_ratio=area_ratio,
        ok=ok,
    )


def resize_keep_ratio(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    """等比例缩小到最长边不超过 ``max_side``（用于加速特征提取）。"""
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def binarize_to_outline(mask: np.ndarray, size: int = 256) -> np.ndarray:
    """把掩码归一化为方块画布上的轮廓图（用于 CNN 的轮廓输入）。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8)
    patch = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = patch.shape[:2]
    scale = (size - 8) / float(max(h, w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), np.uint8)
    oy, ox = (size - new_h) // 2, (size - new_w) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return canvas


__all__ = ["SegmentedLeaf", "segment_leaf", "resize_keep_ratio", "binarize_to_outline"]
