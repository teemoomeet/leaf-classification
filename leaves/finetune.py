"""端到端微调：让 CNN 适应真实场景照片（域随机化增强）。

冻结特征 + 线性分类器在 Flavia 白底扫描图上接近 99%，但真实照片
（多叶簇生、背景杂物、色偏、水珠反光、轻微失焦）与扫描图分布差异
太大，推理时错误率明显上升。

本模块用「域随机化」合成训练数据弥合域差：

* 扫描模式 —— 原始白底图随机旋转/抖色，保住扫描图上的判别力；
* 野生模式 —— 把同物种的叶片抠出来，随机数量（1~6 片）、随机
  位置/缩放/旋转贴到随机背景（白纸/灰桌/暗色虚化/绿色灌丛/木纹色）
  上，再叠加色彩抖动与模糊，模拟手机拍摄的真实条件。

两种模式混训，模型同时保住扫描图精度与真实照片鲁棒性。

::

    python -m leaves.cli finetune            # 训练（CPU 约十几分钟）
    python -m leaves.cli predict -i DIR --model models/flavia_ft_mobilenetv3.pt
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .dataset import load_split
from .deep import IMAGENET_MEAN, IMAGENET_STD, _torch
from .segmentation import leaf_on_white, segment_leaf
from .species import NUM_CLASSES

DEFAULT_FT_NAME = "flavia_ft_mobilenetv3.pt"
DEFAULT_SIZE = 160

# --------------------------------------------------------------------------- #
# 合成增强
# --------------------------------------------------------------------------- #
def _random_background(rng: np.random.Generator, shape: tuple[int, int],
                       real_bg_pool: list[np.ndarray] | None = None) -> np.ndarray:
    """随机合成一张背景图（BGR）。

    一半概率用纯白/浅灰（覆盖「白底对齐后多叶簇」的推理形态），
    其余用真实 Flavia 图虚化暗化得到的自然纹理或程序噪声。
    """
    h, w = shape
    if real_bg_pool and rng.uniform() < 0.5:
        if rng.uniform() < 0.45:
            # 纯白/浅色（模拟白纸、白底对齐后的画布）
            base = np.full((h, w, 3), 255, np.float32)
            base *= rng.uniform(0.92, 1.0)
            img = np.clip(base, 0, 255).astype(np.uint8)
        else:
            # 真实图片虚化：有真实的叶片纹理与虚化质感
            bg = real_bg_pool[int(rng.integers(0, len(real_bg_pool)))]
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
            k = 2 * int(max(6, min(h, w) / 8)) + 1
            bg = cv2.GaussianBlur(bg, (k, k), 0)
            img = np.clip(bg.astype(np.float32) * rng.uniform(0.25, 0.8), 0, 255).astype(np.uint8)
        return img
    kind = rng.integers(0, 5)
    base = np.zeros((h, w, 3), np.float32)
    if kind == 0:      # 白纸
        base[:] = rng.uniform(235, 255)
    elif kind == 1:    # 灰桌面
        base[:] = rng.uniform(150, 215)
    elif kind == 2:    # 暗色虚化（夜景/阴影灌丛）
        base[..., 0] = rng.uniform(10, 60)
        base[..., 1] = rng.uniform(30, 90)
        base[..., 2] = rng.uniform(10, 60)
    elif kind == 3:    # 绿色灌丛
        base[..., 0] = rng.uniform(20, 70)
        base[..., 1] = rng.uniform(70, 140)
        base[..., 2] = rng.uniform(20, 70)
    else:              # 木桌/泥土色
        base[..., 0] = rng.uniform(40, 110)
        base[..., 1] = rng.uniform(70, 140)
        base[..., 2] = rng.uniform(90, 160)
    noise = rng.normal(0, 22, (h, w, 1)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    k = 2 * int(max(3, min(h, w) / 12)) + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _perspective(image: np.ndarray, mask: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """轻微透视畸变，模拟实拍视角。"""
    h, w = image.shape[:2]
    jitter = max(2, int(0.08 * max(h, w)))
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    size = (int(abs(dst[:, 0]).max()) + 1, int(abs(dst[:, 1]).max()) + 1)
    return (
        cv2.warpPerspective(image, M, size, borderMode=cv2.BORDER_REPLICATE),
        cv2.warpPerspective(mask, M, size, flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0),
    )


def _color_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.75, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.7, 1.25), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
    return out


def _load_leaf_cutout(path: str, max_side: int = 320) -> tuple[np.ndarray, np.ndarray] | None:
    """读一张 Flavia 图并分割出 (BGR, mask)。失败返回 None。"""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    seg = segment_leaf(image)
    if not seg.ok or seg.area_ratio >= 0.99:
        return None
    return seg.crop, seg.crop_mask


def _paste_leaf(canvas: np.ndarray, leaf: np.ndarray, mask: np.ndarray,
                rng: np.random.Generator) -> None:
    """把叶片按随机缩放/旋转/透视贴到画布上（原地修改，带软阴影）。"""
    ch, cw = canvas.shape[:2]
    # 逐叶变色：模拟不同光照/叶龄/水光（贴前做，不影响背景）
    leaf = _color_jitter(leaf, rng) if rng.uniform() < 0.7 else leaf
    if rng.uniform() < 0.4:
        leaf, mask = _perspective(leaf, mask, rng)
    lh, lw = leaf.shape[:2]
    scale = rng.uniform(0.3, 0.75) * min(ch, cw) / float(max(lh, lw))
    nh, nw = max(8, int(lh * scale)), max(8, int(lw * scale))
    leaf_s = cv2.resize(leaf, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    angle = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw2, nh2 = int(nw * cos + nh * sin), int(nw * sin + nh * cos)
    M[0, 2] += nw2 / 2 - nw / 2
    M[1, 2] += nh2 / 2 - nh / 2
    leaf_r = cv2.warpAffine(leaf_s, M, (nw2, nh2), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_r = cv2.warpAffine(mask_s, M, (nw2, nh2), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_r = cv2.GaussianBlur(mask_r, (3, 3), 0)  # 羽化边缘
    x = int(rng.uniform(0, max(1, cw - nw2)))
    y = int(rng.uniform(0, max(1, ch - nh2)))
    x1, y1 = min(cw, x + nw2), min(ch, y + nh2)
    m = (mask_r[: y1 - y, : x1 - x][..., None] / 255.0)
    # 软阴影：把掩码偏移几像素、压暗下方画布
    if rng.uniform() < 0.5:
        dx, dy = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        sx0, sy0 = min(cw, x + dx), min(ch, y + dy)
        sx1, sy1 = min(cw, x + dx + (x1 - x)), min(ch, y + dy + (y1 - y))
        roi_s = canvas[sy0:sy1, sx0:sx1]
        sm = (mask_r[: sy1 - sy0, : sx1 - sx0][..., None] / 255.0)
        canvas[sy0:sy1, sx0:sx1] = (roi_s * (1 - 0.45 * sm)).astype(np.uint8)
    roi = canvas[y:y1, x:x1]
    canvas[y:y1, x:x1] = (leaf_r[: y1 - y, : x1 - x] * m + roi * (1 - m)).astype(np.uint8)


class WildComposer:
    """把同一物种的 1~6 片叶子合成到随机背景上，模拟真实照片。

    背景一半概率为纯白/浅色 —— 专门覆盖 ``predict --whiten auto``
    「白底对齐后多叶簇」的推理形态；其余为真实虚化图或程序噪声。
    """

    def __init__(self, paths: list[str], labels: np.ndarray, canvas: int = 256) -> None:
        self.canvas = canvas
        self.by_label: dict[int, list[str]] = {}
        for p, lab in zip(paths, labels):
            self.by_label.setdefault(int(lab), []).append(p)
        self._cutout_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # 真实背景池：随机抽一些训练图做虚化纹理
        self.real_bg_pool: list[np.ndarray] = []
        for p in paths[:: max(1, len(paths) // 40)]:
            cut = self._cutout(p)
            if cut is not None:
                self.real_bg_pool.append(cut[0])

    def _cutout(self, path: str):
        if path not in self._cutout_cache:
            self._cutout_cache[path] = _load_leaf_cutout(path)
        return self._cutout_cache.get(path)

    def compose(self, label: int, rng: np.random.Generator) -> np.ndarray:
        pool = self.by_label[label]
        canvas = _random_background(rng, (self.canvas, self.canvas),
                                    real_bg_pool=self.real_bg_pool)
        n_leaf = int(rng.integers(1, 7))
        for _ in range(n_leaf * 2):
            if n_leaf <= 0:
                break
            path = pool[int(rng.integers(0, len(pool)))]
            cut = self._cutout(path)
            if cut is None:
                continue
            _paste_leaf(canvas, cut[0], cut[1], rng)
            n_leaf -= 1
        return canvas


def _pad_square(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """等比缩放到白底方形画布（与推理协议 :meth:`torch_model._tensor` 一致）。"""
    h, w = image_bgr.shape[:2]
    scale = (size * 0.96) / float(max(h, w))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    oy, ox = (size - new_h) // 2, (size - new_w) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return canvas


def _scan_view(image_bgr: np.ndarray, rng: np.random.Generator,
               size: int) -> np.ndarray:
    """扫描模式视图：白底图随机旋转 + 适度抖色，最后按推理协议补边。"""
    seg = segment_leaf(image_bgr)
    if seg.ok:
        crop = np.where(seg.crop_mask[..., None] > 0, seg.crop, 255)
    else:
        crop = image_bgr
    angle = rng.uniform(0, 360)
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(w * cos + h * sin), int(w * sin + h * cos)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rotated = cv2.warpAffine(crop, M, (nw, nh), borderValue=(255, 255, 255))
    view = _pad_square(rotated, size)
    return _color_jitter(view, rng)


def _random_window(view: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随机局部裁窗（模拟推理时的峰值窗视角）。"""
    h, w = view.shape[:2]
    area = rng.uniform(0.35, 0.85)
    ch, cw = int(h * area ** 0.5), int(w * area ** 0.5)
    y0 = int(rng.uniform(0, max(1, h - ch)))
    x0 = int(rng.uniform(0, max(1, w - cw)))
    return view[y0 : y0 + ch, x0 : x0 + cw]


def _tensorize(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """按推理协议张量化：补边成方形（不压扁长宽比）再归一化。"""
    square = _pad_square(image_bgr, size)
    t = square.astype(np.float32) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(t, (2, 0, 1))


# --------------------------------------------------------------------------- #
# 训练主流程
# --------------------------------------------------------------------------- #
def run(
    model_path: Path | str | None = None,
    arch: str = "mobilenet_v3_small",
    size: int = DEFAULT_SIZE,
    epochs: int = 12,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_backbone: float = 2e-4,
    wild_prob: float = 0.65,
    seed: int = 42,
    device: str = "cpu",
) -> Path:
    """执行微调，返回保存的模型路径。"""
    torch = _torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    splits = load_split()
    train_recs = splits["train"]
    val_recs = splits["val"]
    train_paths = [r.path for r in train_recs]
    train_labels = np.array([r.label for r in train_recs])
    composer = WildComposer(train_paths, train_labels)

    class FTDataSet(Dataset):
        def __init__(self, records, train: bool) -> None:
            self.records = records
            self.train = train

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int):
            rec = self.records[idx]
            data = np.fromfile(rec.path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            if image is None:
                image = np.full((size, size, 3), 255, np.uint8)
            r = np.random.default_rng(seed + idx * 7919 + epoch * 104729)
            if self.train and r.uniform() < wild_prob:
                # 野生模式：合成场景照，再按推理协议做白底对齐 + 补边，
                # 教会模型「先抠叶贴白底 -> 再识别」的完整行为
                view = composer.compose(rec.label, r)
                whitened = leaf_on_white(view)
                if whitened is None:
                    whitened = view
                if r.uniform() < 0.5:
                    k = 2 * int(r.integers(1, 3)) + 1
                    whitened = cv2.GaussianBlur(whitened, (k, k), 0)
                view = _color_jitter(whitened, r)
            else:
                # 扫描模式：白底对齐（对扫描图近似无损）+ 旋转 + 抖色
                whitened = leaf_on_white(image)
                if whitened is None:
                    whitened = image
                view = _scan_view(whitened, r, size)
            if self.train and r.uniform() < 0.3:
                view = _pad_square(_random_window(view, r), size)
            return _tensorize(view, size), int(rec.label)

    from torchvision.models import MobileNet_V3_Small_Weights as W

    net = models.mobilenet_v3_small(weights=W.DEFAULT)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, NUM_CLASSES)

    head_params = list(net.classifier.parameters())
    backbone_params = [p for n_, p in net.named_parameters() if not n_.startswith("classifier")]
    for p in net.features[:-3].parameters():      # 浅层冻结，只微调后 3 个 stage
        p.requires_grad_(False)
    backbone_params = [p for n_, p in net.named_parameters()
                       if n_.startswith("features") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_ds = FTDataSet(train_recs, train=True)
    val_ds = FTDataSet(val_recs, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_acc, best_state, out_path = 0.0, None, Path(model_path) if model_path else Path("models") / DEFAULT_FT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"微调开始：{len(train_ds)} 训练 / {len(val_ds)} 验证，"
          f"输入 {size}px，野生模式占比 {wild_prob:.0%}，共 {epochs} 轮")

    for epoch in range(epochs):
        net.train()
        t0, running, seen, correct = time.time(), 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        scheduler.step()
        train_acc = correct / max(1, seen)

        net.eval()
        vcorrect = 0
        with torch.no_grad():
            for images, labels in val_loader:
                vcorrect += int((net(images).argmax(1) == labels).sum())
        val_acc = vcorrect / max(1, len(val_ds))
        mark = ""
        if val_acc >= best_acc:
            best_acc, mark = val_acc, "  <- 保存最优"
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"  轮次 {epoch + 1:>2}/{epochs}  loss={running / max(1, seen):.4f}  "
              f"训练={train_acc:.3f}  验证={val_acc:.3f}  "
              f"({time.time() - t0:.0f}s){mark}", flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "arch": arch,
            "size": size,
            "num_classes": NUM_CLASSES,
            "format": "leaves-finetune-v1",
            "val_acc": best_acc,
        },
        out_path,
    )
    print(f"已保存：{out_path}（验证准确率 {best_acc:.3f}）")
    return out_path


__all__ = ["run", "DEFAULT_FT_NAME", "DEFAULT_SIZE", "WildComposer"]
