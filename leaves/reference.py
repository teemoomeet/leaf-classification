"""参考图库匹配：识别 Flavia 32 种树之外的任意自定义物种。

与 :mod:`leaves.train` 的闭集分类不同，这个模式**不需要训练**：

* 参考库构建 —— 扫描一个文件夹，图片文件名（去掉扩展名）即物种标签，
  提取特征向量存为特征库；
* 匹配 —— 对待识别图片提取特征，与库中参考向量算余弦相似度，
  最相似的类即预测结果；
* 开集行为 —— top1 相似度低于阈值时给出「可能不在参考库中」的提示，
  而不是硬猜一个答案。

适用场景：农作物、园艺植物、其他地区的树种……每类哪怕只有 1 张参考图
也能用；放进更多同名图片（每类多张）会自动变成多样本投票。

::

    # 构建/缓存参考库并对未知图片批量匹配
    python -m leaves.cli match --library D:/my_refs --input D:/unknown

    # 对参考库自身做留一法（leave-one-out）自检
    python -m leaves.cli match --library D:/my_refs
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .config import OUTPUT_DIR, ensure_dirs
from .registry import BACKENDS

#: 参考库支持的图片格式
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

#: 常见农作物/园艺植物的拉丁学名 -> 中文名（不在表内的标签原样显示）
CROP_NAMES: dict[str, str] = {
    "Capsicum annuum": "辣椒",
    "Fragaria x ananassa": "草莓",
    "Glycine max": "大豆",
    "Malus domestica": "苹果",
    "Prunus avium": "欧洲甜樱桃",
    "Prunus persica": "桃",
    "Rubus idaeus": "覆盆子",
    "Solanum lycopersicum": "番茄",
    "Solanum tuberosum": "马铃薯",
    "Vaccinium corymbosum": "高丛蓝莓",
    "Vitis vinifera": "葡萄",
    "Zea mays": "玉米",
    # 常见园艺/树木补充
    "Acer palmatum": "鸡爪槭",
    "Acer buergerianum": "三角槭",
    "Liriodendron chinense": "鹅掌楸",
    "Magnolia grandiflora": "荷花木兰",
    "Ginkgo biloba": "银杏",
}


def display_name(label: str) -> str:
    """拉丁学名 -> 「学名 (中文名)」；查不到中文名时原样返回。"""
    cn = CROP_NAMES.get(label) or CROP_NAMES.get(label.lower())
    return f"{label} ({cn})" if cn else label


def scan_labeled(folder: Path | str) -> list[tuple[Path, str]]:
    """扫描参考图库目录，返回 ``(路径, 标签)`` 列表，标签 = 文件名去扩展名。

    支持子目录（标签仍取文件名），同名图片自动归为同一类（多样本）。
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"参考图库目录不存在: {folder}")
    items = sorted(
        (p, p.stem.strip())
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    )
    if not items:
        raise FileNotFoundError(
            f"目录里没有图片: {folder}\n请把参考图放进去，文件名 = 物种名，例如「苹果.jpg」"
        )
    return items


@dataclasses.dataclass
class MatchResult:
    """单张图片的匹配结果。"""

    path: str
    true_label: str | None  # 留一法自检时有值
    top_labels: list[str]
    top_sims: list[float]
    uncertain: bool

    @property
    def pred(self) -> str:
        return self.top_labels[0]

    @property
    def sim(self) -> float:
        return float(self.top_sims[0])

    @property
    def correct(self) -> bool | None:
        if self.true_label is None:
            return None
        return self.pred == self.true_label


class ReferenceLibrary:
    """参考特征库：标准化 + L2 归一化的特征矩阵，余弦相似度匹配。

    :param paths: 参考图片路径
    :param labels: 每张图的物种标签（可重复 = 一类多张）
    :param matrix: 原始特征矩阵 ``(n_ref, dim)``
    :param backend: 使用的特征后端
    """

    def __init__(
        self,
        paths: list[str],
        labels: list[str],
        matrix: np.ndarray,
        backend: str = "features",
        arch: str = "",
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.backend = backend
        self.arch = arch
        X = np.asarray(matrix, dtype=np.float64)
        # 标准化平衡各维量纲，再 L2 归一化，点积即余弦相似度。
        # 注：留一法自检时标准化统计量包含查询自身，12~几千张规模下影响可忽略。
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0
        self.matrix = self._normalize((X - self._mean) / self._std)
        self.classes: list[str] = list(dict.fromkeys(self.labels))
        self._class_rows = {
            c: [i for i, lab in enumerate(self.labels) if lab == c] for c in self.classes
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return X / norms

    def transform(self, raw_matrix: np.ndarray) -> np.ndarray:
        """把查询特征用库内统计量标准化 + 归一化。"""
        X = np.asarray(raw_matrix, dtype=np.float64)
        return self._normalize((X - self._mean) / self._std)

    # ------------------------------------------------------------------ #
    def match_matrix(self, queries: np.ndarray, exclude: int | None = None) -> np.ndarray:
        """返回 ``(n_query, n_class)`` 的类别相似度矩阵（类内取最大）。

        :param exclude: 留一法时排除的参考库行号（该行不参与投票）
        :note: 某类所有参考都被排除时，该列填 -inf（不可能被选中），
               避免空类以 0.0 分「幻影获胜」。
        """
        refs = self.matrix
        if exclude is not None:
            keep = [i for i in range(len(self.labels)) if i != exclude]
            refs = refs[keep]
        sims = queries @ refs.T  # (n_query, n_ref_kept)
        label_arr = np.array(
            [lab for i, lab in enumerate(self.labels) if i != exclude]
            if exclude is not None else self.labels
        )
        out = np.full((len(queries), len(self.classes)), -np.inf)
        for ci, c in enumerate(self.classes):
            cols = np.where(label_arr == c)[0]
            if len(cols):
                out[:, ci] = sims[:, cols].max(axis=1)
        return out

    def match_one(self, query: np.ndarray, top_k: int = 3) -> tuple[list[str], list[float]]:
        """匹配单个查询向量，返回 ``(top-k 类名, top-k 相似度)``。"""
        sims = self.match_matrix(query[None, :])[0]
        order = np.argsort(-sims)[: max(1, top_k)]
        return [self.classes[i] for i in order], [float(sims[i]) for i in order]

    # ------------------------------------------------------------------ #
    def leave_one_out(self, top_k: int = 3) -> list[MatchResult]:
        """留一法自检：每张参考图轮流当查询（自身排除），验证库的区分力。"""
        results: list[MatchResult] = []
        for i in range(len(self.labels)):
            sims = self.match_matrix(self.matrix[i : i + 1], exclude=i)[0]
            order = np.argsort(-sims)[: max(1, top_k)]
            results.append(
                MatchResult(
                    path=self.paths[i],
                    true_label=self.labels[i],
                    top_labels=[self.classes[j] for j in order],
                    top_sims=[float(sims[j]) for j in order],
                    uncertain=False,
                )
            )
        return results


# --------------------------------------------------------------------------- #
# 构建 / 缓存
# --------------------------------------------------------------------------- #
def _cache_path(folder: Path, backend: str, arch: str, whiten: bool = False) -> Path:
    suffix = "_w" if whiten else ""
    name = f".leaves_ref_{backend}_{arch}{suffix}.npz"
    return folder / name


def build_library(
    folder: Path | str,
    backend: str = "features",
    arch: str = "mobilenet_v3_small",
    max_side: int = 512,
    workers: int = 4,
    batch_size: int = 32,
    rebuild: bool = False,
    verbose: bool = True,
    whiten: bool = True,
) -> ReferenceLibrary:
    """构建参考特征库（带缓存）。文件名 = 物种名。

    :param rebuild: 忽略缓存强制重建（往库里加了新图后用它刷新）
    :param whiten: 白底对齐预处理（真实场景照片推荐开启）
    """
    folder = Path(folder)
    items = scan_labeled(folder)
    cache = _cache_path(folder, backend, arch, whiten)

    if not rebuild and cache.exists():
        try:
            payload = np.load(cache, allow_pickle=True)
            paths = [str(p) for p in payload["paths"]]
            labels = [str(l) for l in payload["labels"]]
            if paths == [str(p) for p, _ in items]:
                if verbose:
                    print(f"[参考库] 使用缓存 {cache.name}（{len(labels)} 张，"
                          f"共 {len(set(labels))} 类；加入新图后加 --rebuild 刷新）")
                return ReferenceLibrary(paths, labels, payload["X"], backend, arch)
        except Exception:  # noqa: BLE001 - 缓存损坏时静默重建
            pass

    from .backends import extract_matrix_from_paths

    paths = [str(p) for p, _ in items]
    labels = [lab for _, lab in items]
    if verbose:
        print(f"[参考库] 提取 {len(paths)} 张参考图特征（后端 {backend}）...")
    X, _, _seg_ok = extract_matrix_from_paths(
        paths, backend=backend, arch=arch, max_side=max_side,
        workers=workers, batch_size=batch_size, verbose=verbose, whiten=whiten,
    )
    lib = ReferenceLibrary(paths, labels, X, backend, arch)
    try:
        np.savez_compressed(cache, X=X, paths=paths, labels=labels)
        if verbose:
            print(f"[参考库] 已缓存到 {cache.name}")
    except Exception:  # noqa: BLE001 - 只读目录等场景忽略缓存失败
        pass
    return lib


# --------------------------------------------------------------------------- #
# 批量匹配入口
# --------------------------------------------------------------------------- #
def collect_images(folder_or_file: Path | str) -> list[Path]:
    """收集待识别图片（目录则递归，兼容单文件）。"""
    p = Path(folder_or_file)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {p}")
    files = sorted(
        q for q in p.rglob("*")
        if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES and not q.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"目录里没有图片: {p}")
    return files


def run_match(
    library_dir: Path | str,
    input_dir: Path | str | None = None,
    backend: str = "features",
    arch: str = "mobilenet_v3_small",
    top_k: int = 3,
    threshold: float = 0.55,
    max_side: int = 512,
    workers: int = 4,
    batch_size: int = 32,
    rebuild: bool = False,
    output_dir: Path | str | None = None,
    save_vis: bool = True,
    max_vis: int = 60,
    whiten: bool = True,
) -> dict:
    """执行参考库匹配：``input_dir`` 为空时对参考库做留一法自检。

    :return: 汇总 dict（含 ``results``/``accuracy``/``csv`` 等键）
    """
    import pandas as pd

    lib = build_library(
        library_dir, backend=backend, arch=arch, max_side=max_side,
        workers=workers, batch_size=batch_size, rebuild=rebuild, whiten=whiten,
    )
    self_test = input_dir is None
    print(
        f"[匹配] 参考库 {len(lib.paths)} 张 / {len(lib.classes)} 类；"
        f"模式：{'留一法自检' if self_test else '批量匹配'}"
    )

    from .backends import extract_matrix_from_paths

    if self_test:
        results = lib.leave_one_out(top_k=top_k)
        # 自检时按是否低于阈值补标 uncertain
        for r in results:
            r.uncertain = r.sim < threshold
        query_paths = [r.path for r in results]
    else:
        query_paths = [str(p) for p in collect_images(input_dir)]
        Q, _, _ = extract_matrix_from_paths(
            query_paths, backend=backend, arch=arch, max_side=max_side,
            workers=workers, batch_size=batch_size, verbose=True, whiten=whiten,
        )
        Q = lib.transform(Q)
        results = []
        for path, vec in zip(query_paths, Q):
            top_labels, top_sims = lib.match_one(vec, top_k=top_k)
            results.append(
                MatchResult(
                    path=path, true_label=None, top_labels=top_labels,
                    top_sims=top_sims, uncertain=top_sims[0] < threshold,
                )
            )

    # ---- 打印 ---- #
    correct = sum(1 for r in results if r.correct)
    known = sum(1 for r in results if r.correct is not None)
    print("-" * 76)
    for r in results:
        tag = ""
        if r.correct is not None:
            tag = "  ✓" if r.correct else "  ✗ 应为 " + display_name(r.true_label)
        warn = "  ⚠ 相似度低" if r.uncertain else ""
        name = Path(r.path).name
        print(f"  {name:<32} -> {display_name(r.pred):<28} 相似度 {r.sim:.3f}{tag}{warn}")
    print("-" * 76)
    if known:
        print(f"留一法自检：正确 {correct}/{known}，准确率 {correct / known:.1%}")
        if len(lib.classes) == len(lib.labels):
            # 每类恰好 1 张时，排除自身后该类必然为空，留一法不可能答对
            print("（提示：每类只有 1 张参考图时，留一法在数学上不可能答对；"
                  "每类补充 2 张以上再看准确率才有意义）")
    n_uncertain = sum(1 for r in results if r.uncertain)
    if n_uncertain:
        print(f"⚠ {n_uncertain} 张 top1 相似度低于 {threshold}，可能不在参考库中，建议人工复核")

    # ---- 输出 ---- #
    ensure_dirs()
    out_dir = Path(output_dir) if output_dir else (
        OUTPUT_DIR / ("match_selftest" if self_test else f"match_{Path(query_paths[0]).parent.name}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        row = {
            "file": Path(r.path).name,
            "pred_label": r.pred,
            "pred_name": display_name(r.pred),
            "similarity": round(r.sim, 4),
            "uncertain": r.uncertain,
        }
        for k in range(1, len(r.top_labels)):
            row[f"top{k + 1}_label"] = r.top_labels[k]
            row[f"top{k + 1}_sim"] = round(r.top_sims[k], 4)
        if r.true_label is not None:
            row["true_label"] = r.true_label
            row["correct"] = r.correct
        rows.append(row)
    frame = pd.DataFrame(rows)
    csv_path = out_dir / "match_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    (out_dir / "match_results.json").write_text(
        json.dumps(
            [
                {**r.__dict__, "pred": r.pred, "sim": r.sim}
                for r in results
            ],
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已保存：\n  {csv_path}\n  {out_dir / 'match_results.json'}")

    # ---- 可视化 ---- #
    if save_vis:
        import cv2

        from .backends import load_images
        from .viz import plot_samples

        imgs = load_images(query_paths[:max_vis], max_side=320)
        titles = []
        for r in results[:max_vis]:
            mark = ""
            if r.correct is not None:
                mark = "✓ " if r.correct else "✗ "
            warn = " ⚠" if r.uncertain else ""
            titles.append(f"{mark}{display_name(r.pred)} {r.sim:.2f}{warn}")
        vis_dir = out_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        grid = plot_samples(
            [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs],
            titles, vis_dir / "match_grid.png", cols=4, figsize_per=2.6,
        )
        print(f"  {grid}")

    accuracy = (correct / known) if known else None
    return {
        "results": results, "accuracy": accuracy,
        "n_uncertain": n_uncertain, "csv": str(csv_path), "out_dir": str(out_dir),
    }


__all__ = [
    "CROP_NAMES", "MatchResult", "ReferenceLibrary",
    "build_library", "collect_images", "display_name", "run_match", "scan_labeled",
]
