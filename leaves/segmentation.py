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


def _fill_small_holes(mask: np.ndarray, max_ratio: float = 0.05) -> np.ndarray:
    """只填充面积不超过图像 ``max_ratio`` 的小孔洞。

    与 :func:`_fill_holes` 的区别：真实照片里叶片之间的深色缝隙、
    枝条穿插的大洞是有信息量的轮廓，不应一并填掉。
    """
    h, w = mask.shape[:2]
    flood = mask.copy()
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, canvas, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    keep = np.zeros_like(holes)
    total = h * w
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= max_ratio * total:
            keep[labels == i] = 255
    return cv2.bitwise_or(mask, keep)


def leaf_on_white(image_bgr: np.ndarray, margin: float = 0.08) -> np.ndarray | None:
    """把叶片抠出并贴到纯白画布上（真实照片 -> 白底扫描分布的对齐）。

    用「绿色占优」的 HSV 掩码定位叶片：只要背景不是绿色（灰桌面、白纸、
    土壤、木头……）都稳健，天然免疫「背景比叶片亮还是暗」的极性问题。

    针对「背景也是绿色植被」的复杂场景，做了两级处理：

    1. 只填小洞 —— 叶片间深色缝隙、枝条穿插的大洞保留为背景；
    2. 全场景绿占比过高时（>0.55，说明前景背景连成一片），改用
       「绿掩码内亮度阶梯阈值」分离受光主体：自然照片中主体叶片通常
       比阴影里的背景更亮，从 Otsu 阈值起逐级升高取第一个面积合理的
       连通域。

    :param margin: 叶片外接矩形四周留白比例
    :return: 白底 BGR 图像；找不到绿色区域时返回 ``None``（调用方回退原图）
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("输入图像为空")

    H, W = image_bgr.shape[:2]
    total = H * W
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # 绿色 hue 范围放宽到 25~95（含黄绿与深绿），要求一定的饱和度以排除灰白背景
    green = cv2.inRange(hsv, (25, 40, 30), (95, 255, 255))
    # 绿色占优的像素（G 分量明显高于 R/B）作为兜底补充，处理偏蓝绿/暗绿叶片
    b, g, r = image_bgr[..., 0].astype(int), image_bgr[..., 1].astype(int), image_bgr[..., 2].astype(int)
    dominant = ((g > r + 12) & (g > b + 12) & (sat > 40)).astype(np.uint8) * 255
    foreground = cv2.bitwise_or(green, dominant)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)

    if np.count_nonzero(foreground) < 0.02 * total:
        return None  # 几乎没有绿色前景（可能是非绿植物或异常图），交由调用方回退

    mask = _largest_component(foreground)
    if not np.count_nonzero(mask):
        return None
    mask = _fill_small_holes(mask, 0.05)

    if np.count_nonzero(mask) > 0.55 * total:
        # 全场景绿色连片：绿掩码内按亮度阶梯分离受光主体
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        vals = gray[mask > 0]
        if len(vals):
            t_otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates = [float(t_otsu)] + [np.percentile(vals, p) for p in (60, 65, 70, 75, 80, 85)]
            kernel_sub = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            for t in candidates:
                sub = np.where((gray >= t) & (foreground > 0), 255, 0).astype(np.uint8)
                sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kernel_sub, iterations=2)
                sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel, iterations=1)
                sub = _largest_component(sub)
                if not np.count_nonzero(sub):
                    continue
                sub = _fill_small_holes(sub, 0.03)
                area = np.count_nonzero(sub)
                if 0.05 * total < area < 0.55 * total:
                    mask = sub
                    break

    x, y, bw, bh = cv2.boundingRect(mask)
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + bw + mx), min(H, y + bh + my)

    crop = image_bgr[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    white = np.full_like(crop, 255)
    white[crop_mask > 0] = crop[crop_mask > 0]
    return white


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


__all__ = [
    "SegmentedLeaf", "segment_leaf", "leaf_on_white",
    "resize_keep_ratio", "binarize_to_outline",
]
