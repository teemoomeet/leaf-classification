"""叶片特征提取。

一条完整的特征向量由 4 组可解释特征拼接而成：

============  ===================================================================
分组           内容
============  ===================================================================
shape         面积、周长、凸包、solidity、圆度、离心率、矩形度、长宽轴、等效直径、
              周长比、生理长宽、Hu 不变矩
fourier       轮廓复数坐标 FFT 后的归一化幅度谱（旋转/缩放/起点不变的形状描述子）
texture       多尺度 LBP 直方图 + 灰度共生矩阵（GLCM）统计量
color         叶片区域内的 RGB / HSV 均值与标准差
============  ===================================================================

这些特征全部手工推导、量纲清晰，配合 SVM 即可在 Flavia 上取得 95%+ 的准确率，
而且不需要 GPU。
"""

from __future__ import annotations

import cv2
import numpy as np

from .segmentation import SegmentedLeaf, segment_leaf

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
FOURIER_DESCRIPTORS = 16     # 傅里叶描述子个数
LBP_SPECS = ((8, 1.0), (16, 2.0))  # (邻居数, 半径)
GLCM_LEVELS = 16
GLCM_DISTANCES = (1, 2, 4)
GLCM_ANGLES = (0, 45, 90, 135)


# --------------------------------------------------------------------------- #
# 形状特征
# --------------------------------------------------------------------------- #
def _hu_moments(mask: np.ndarray) -> list[float]:
    moments = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(moments).ravel()
    return [float(-np.sign(v) * np.log10(abs(v) + 1e-30)) for v in hu]


def _fourier_descriptors(contour: np.ndarray, n: int = FOURIER_DESCRIPTORS) -> list[float]:
    """复数坐标 FFT 的归一化幅度谱，具备平移/旋转/缩放/起点不变性。"""
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 8:
        return [0.0] * n

    # 按弧长等间隔重采样到固定点数
    closed = np.vstack([points, points[:1]])  # 闭合轮廓，长度 n+1
    diff = np.diff(closed, axis=0)
    seg_len = np.hypot(diff[:, 0], diff[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # 长度 n+1，与 closed 对齐
    if cum[-1] <= 0:
        return [0.0] * n
    samples = np.linspace(0, cum[-1], 64, endpoint=False)
    xs = np.interp(samples, cum, closed[:, 0])
    ys = np.interp(samples, cum, closed[:, 1])

    z = (xs - xs.mean()) + 1j * (ys - ys.mean())
    spectrum = np.fft.fft(z)
    mag = np.abs(spectrum)
    denom = mag[1] if mag[1] > 1e-9 else (mag.max() if mag.max() > 1e-9 else 1.0)
    out: list[float] = []
    for k in range(1, n + 1):
        out.append(float(mag[k % len(mag)] / denom))
    return out


def _physiological_axes(contour: np.ndarray) -> tuple[float, float]:
    """生理长度/生理宽度。

    生理长度 = 叶片轮廓上相距最远两点的距离（叶基到叶尖）；
    生理宽度 = 轮廓在垂直于该连线方向上的最大跨度。
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 2:
        return 0.0, 0.0

    # 先用凸包降点数，再找凸包中距离最远的一对点（旋转卡壳的简化实现）
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        hull = pts
    best_len, best_pair = -1.0, None
    for i in range(len(hull)):
        d = np.hypot(hull[:, 0] - hull[i, 0], hull[:, 1] - hull[i, 1])
        j = int(np.argmax(d))
        if d[j] > best_len:
            best_len, best_pair = float(d[j]), (hull[i], hull[j])

    if best_pair is None or best_len <= 0:
        return 0.0, 0.0
    (x1, y1), (x2, y2) = best_pair
    ux, uy = (x2 - x1) / best_len, (y2 - y1) / best_len
    # 垂直方向的投影跨度
    proj = -(pts[:, 0] - x1) * uy + (pts[:, 1] - y1) * ux
    width = float(proj.max() - proj.min())
    return best_len, width


def shape_features(seg: SegmentedLeaf) -> tuple[list[str], list[float]]:
    """由分割结果计算形状特征。"""
    mask = seg.crop_mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        names = ["shape_" + n for n in _SHAPE_NAMES] + [
            f"fourier_{i}" for i in range(1, FOURIER_DESCRIPTORS + 1)
        ]
        return names, [0.0] * len(names)

    contour = max(contours, key=cv2.contourArea)
    area = float(np.count_nonzero(mask))
    perimeter = float(cv2.arcLength(contour, True))

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    (_, _), (d1, d2), _ = cv2.fitEllipse(contour)  # 拟合椭圆，角度不参与统计
    major, minor = float(max(d1, d2)), float(min(d1, d2))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2))) if major > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    rect_area = float(w * h)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    rot_area = float(max(rw, 1.0) * max(rh, 1.0))

    phys_len, phys_width = _physiological_axes(contour)

    values = [
        area,
        perimeter,
        hull_area,
        hull_perimeter,
        float(area / hull_area) if hull_area > 0 else 0.0,
        float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0,
        float(perimeter**2 / (4.0 * np.pi * area)) if area > 0 else 0.0,
        float(hull_perimeter / perimeter) if perimeter > 0 else 0.0,
        major,
        minor,
        float(major / minor) if minor > 0 else 0.0,
        eccentricity,
        float(area / rect_area) if rect_area > 0 else 0.0,
        float(area / rot_area) if rot_area > 0 else 0.0,
        float(np.sqrt(4.0 * area / np.pi)),
        phys_len,
        phys_width,
        float(phys_len / phys_width) if phys_width > 0 else 0.0,
        float(perimeter / phys_len) if phys_len > 0 else 0.0,
        float(phys_len / perimeter) if perimeter > 0 else 0.0,
        float(area / (phys_len * phys_width)) if phys_len * phys_width > 0 else 0.0,
    ]

    hu = _hu_moments(mask)
    values.extend(hu)
    names = ["shape_" + n for n in _SHAPE_NAMES]

    fourier = _fourier_descriptors(contour)
    names.extend(f"fourier_{i}" for i in range(1, FOURIER_DESCRIPTORS + 1))
    values.extend(fourier)
    return names, [float(v) for v in values]


_SHAPE_NAMES = [
    "area",
    "perimeter",
    "hull_area",
    "hull_perimeter",
    "solidity",
    "circularity",
    "compactness",
    "convexity",
    "major_axis",
    "minor_axis",
    "aspect_ratio",
    "eccentricity",
    "rectangularity",
    "rot_rect_fill",
    "equivalent_diameter",
    "physiological_length",
    "physiological_width",
    "physiological_ratio",
    "perimeter_ratio_length",
    "length_ratio_perimeter",
    "area_ratio_phys",
] + [f"hu_{i}" for i in range(1, 8)]


# --------------------------------------------------------------------------- #
# 纹理特征
# --------------------------------------------------------------------------- #
def _bilinear(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """按浮点坐标对图像做双线性插值采样。"""
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)
    return (
        img[y0c, x0c] * wa
        + img[y0c, x1c] * wb
        + img[y1c, x0c] * wc
        + img[y1c, x1c] * wd
    )


def _uniform_lbp_hist(gray: np.ndarray, mask: np.ndarray, points: int, radius: float) -> np.ndarray:
    """旋转不变 uniform LBP 直方图（uniform 模式 + 1 个非 uniform 桶）。"""
    g = gray.astype(np.float32)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    center = g
    codes = np.zeros((h, w), np.int32)
    for i in range(points):
        theta = 2.0 * np.pi * i / points
        ys = yy - radius * np.sin(theta)
        xs = xx + radius * np.cos(theta)
        neighbour = _bilinear(g, ys, xs)
        codes |= ((neighbour >= center).astype(np.int32) << i)

    # 旋转不变编码：0/1 环形序列中 0->1 的跳变次数
    transitions = np.zeros((h, w), np.int32)
    for i in range(points):
        a = (codes >> i) & 1
        b = (codes >> ((i + 1) % points)) & 1
        transitions += (a != b).astype(np.int32)

    # uniform 模式共 (points + 1) 种；其余归入最后一个桶
    n_uniform = points + 1
    weight = np.where(codes == 0, 0, np.where(codes == (1 << points) - 1, 0, 1))
    bin_idx = np.where(transitions <= 2, (transitions * (points // 2) // 2 + weight), n_uniform)
    bin_idx = np.clip(bin_idx, 0, n_uniform)

    valid = mask > 0
    hist = np.bincount(bin_idx[valid].ravel(), minlength=n_uniform + 1).astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _glcm(gray_q: np.ndarray, distance: int, angle_deg: int) -> np.ndarray:
    """量化灰度图的灰度共生矩阵（对称、已归一化）。"""
    h, w = gray_q.shape
    theta = np.deg2rad(angle_deg)
    dy = int(round(-distance * np.sin(theta)))
    dx = int(round(distance * np.cos(theta)))

    y0a, y1a = max(0, dy), h + min(0, dy)
    x0a, x1a = max(0, dx), w + min(0, dx)
    a = gray_q[y0a:y1a, x0a:x1a].astype(np.int64)
    b = gray_q[y0a - dy : y1a - dy, x0a - dx : x1a - dx].astype(np.int64)

    counts = np.bincount(
        (a * GLCM_LEVELS + b).ravel(), minlength=GLCM_LEVELS * GLCM_LEVELS
    ).reshape(GLCM_LEVELS, GLCM_LEVELS).astype(np.float64)
    matrix = counts + counts.T
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def _glcm_properties(matrix: np.ndarray) -> list[float]:
    i = np.arange(GLCM_LEVELS)[:, None]
    j = np.arange(GLCM_LEVELS)[None, :]
    diff = i - j

    contrast = float((matrix * diff**2).sum())
    dissimilarity = float((matrix * np.abs(diff)).sum())
    homogeneity = float((matrix / (1.0 + diff**2)).sum())
    energy = float(np.sqrt((matrix**2).sum()))
    mu_i = float((i * matrix).sum())
    mu_j = float((j * matrix).sum())
    sig_i = float(np.sqrt(((i - mu_i) ** 2 * matrix).sum()))
    sig_j = float(np.sqrt(((j - mu_j) ** 2 * matrix).sum()))
    if sig_i > 1e-9 and sig_j > 1e-9:
        correlation = float(((i - mu_i) * (j - mu_j) * matrix).sum() / (sig_i * sig_j))
    else:
        correlation = 0.0
    return [contrast, dissimilarity, homogeneity, energy, correlation]


_GLCM_PROP_NAMES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def texture_features(gray: np.ndarray, mask: np.ndarray) -> tuple[list[str], list[float]]:
    """LBP + GLCM 纹理特征。"""
    names: list[str] = []
    values: list[float] = []

    for points, radius in LBP_SPECS:
        hist = _uniform_lbp_hist(gray, mask, points, radius)
        tag = f"lbp_p{points}_r{radius:g}"
        names.extend(f"texture_{tag}_{i}" for i in range(len(hist)))
        values.extend(float(v) for v in hist)

    quantized = np.clip(
        (gray.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.int32),
        0,
        GLCM_LEVELS - 1,
    ).astype(np.uint8)
    for distance in GLCM_DISTANCES:
        props = np.zeros(5, dtype=np.float64)
        for angle in GLCM_ANGLES:
            props += np.asarray(_glcm_properties(_glcm(quantized, distance, angle)))
        props /= len(GLCM_ANGLES)
        names.extend(f"texture_glcm_d{distance}_{p}" for p in _GLCM_PROP_NAMES)
        values.extend(float(v) for v in props)

    return names, values


# --------------------------------------------------------------------------- #
# 颜色特征
# --------------------------------------------------------------------------- #
def color_features(crop_bgr: np.ndarray, mask: np.ndarray) -> tuple[list[str], list[float]]:
    """叶片区域内 RGB / HSV 的均值与标准差（共 12 维）。"""
    valid = mask > 0
    if not np.any(valid):
        names = [f"color_{c}_{s}" for c in ("b", "g", "r", "h", "s", "v") for s in ("mean", "std")]
        return names, [0.0] * len(names)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    names: list[str] = []
    values: list[float] = []
    for channel, array in zip(("b", "g", "r"), cv2.split(crop_bgr)):
        data = array[valid].astype(np.float64)
        names += [f"color_{channel}_mean", f"color_{channel}_std"]
        values += [float(data.mean()), float(data.std())]
    for channel, array in zip(("h", "s", "v"), cv2.split(hsv)):
        data = array[valid].astype(np.float64)
        names += [f"color_{channel}_mean", f"color_{channel}_std"]
        values += [float(data.mean()), float(data.std())]
    return names, values


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
_CACHED_NAMES: list[str] | None = None


def extract_features(
    image_bgr: np.ndarray,
    max_side: int = 384,
    precomputed_seg: SegmentedLeaf | None = None,
) -> tuple[np.ndarray, list[str], bool]:
    """提取一张图像的完整特征向量。

    :param image_bgr: BGR 图像
    :param max_side: 特征提取前把图像缩放到最长边不超过该值，以加快速度
    :param precomputed_seg: 已算好的分割结果，避免重复计算
    :return: (特征向量, 特征名列表, 分割是否成功)
    """
    global _CACHED_NAMES

    if precomputed_seg is not None:
        seg = precomputed_seg
    else:
        image = image_bgr
        h, w = image.shape[:2]
        scale = max_side / float(max(h, w))
        if scale < 1.0:
            image = cv2.resize(
                image, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        seg = segment_leaf(image)

    # 分割失败时退化为"整幅图都是叶片"，保证特征维度一致
    if not seg.ok:
        crop = seg.crop if seg.crop.size else image_bgr
        crop_mask = np.full(crop.shape[:2], 255, np.uint8)
        seg = SegmentedLeaf(
            mask=crop_mask, crop=crop, crop_mask=crop_mask,
            bbox=seg.bbox, area_ratio=seg.area_ratio, ok=False,
        )

    gray = cv2.cvtColor(seg.crop, cv2.COLOR_BGR2GRAY)

    names: list[str] = []
    values: list[float] = []
    for group in (shape_features(seg), texture_features(gray, seg.crop_mask),
                  color_features(seg.crop, seg.crop_mask)):
        names.extend(group[0])
        values.extend(group[1])

    vector = np.asarray(values, dtype=np.float32)
    if np.any(~np.isfinite(vector)):
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    if _CACHED_NAMES is None:
        _CACHED_NAMES = names
    return vector, names, seg.ok


def feature_names() -> list[str]:
    """返回特征名列表（与 :func:`extract_features` 的输出顺序一致）。"""
    global _CACHED_NAMES
    if _CACHED_NAMES is None:
        blank = np.full((64, 64, 3), 255, np.uint8)
        blank[16:48, 16:48] = 0
        _, names, _ = extract_features(blank)
        _CACHED_NAMES = names
    return _CACHED_NAMES


__all__ = [
    "extract_features",
    "feature_names",
    "shape_features",
    "texture_features",
    "color_features",
    "FOURIER_DESCRIPTORS",
]
