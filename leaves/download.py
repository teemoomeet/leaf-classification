"""Flavia 数据集下载与校验。

官方地址（SourceForge）：http://flavia.sourceforge.net/
但 SourceForge 在部分网络环境下会被 Cloudflare 拦截，因此脚本内置了多个
下载源，按顺序自动尝试，任意一个成功即可。

用法::

    python -m leaves.cli download              # 自动选择可用源
    python -m leaves.cli download --source github
    python -m leaves.cli download --source sourceforge
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import RAW_DIR, ensure_dirs
from .species import FLAVIA_ID_RANGES, label_from_filename

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: GitHub 镜像：包含 1907 张原始图像的公开仓库
GITHUB_REPO = "Leena377/-Leaf-Species-Recognition"
GITHUB_BRANCH = "main"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}"

#: 文件下载镜像。raw.githubusercontent.com 在部分网络会被限流，
#: 因此同时准备了 jsDelivr / ghproxy 等公共 CDN，逐个回退。
IMAGE_MIRRORS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/{repo}/{branch}/{path}",
    "https://gcore.jsdelivr.net/gh/{repo}@{branch}/{path}",
    "https://cdn.jsdelivr.net/gh/{repo}@{branch}/{path}",
    "https://ghproxy.net/https://raw.githubusercontent.com/{repo}/{branch}/{path}",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#: 官方 tar 包内图片总量，用于校验
EXPECTED_COUNT = sum(end - start + 1 for start, end, _ in FLAVIA_ID_RANGES)


def _request(url: str, timeout: int = 60) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _download(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> int:
    """带重试的文件下载，返回字节数。"""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url, timeout), timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise IOError("下载内容为空")
            dest.write_bytes(data)
            return len(data)
        except Exception as exc:  # noqa: BLE001 - 需要兜住各种网络异常
            last_err = exc
            time.sleep(0.6 * (attempt + 1))
    raise IOError(f"下载失败 {url}: {last_err}")


# --------------------------------------------------------------------------- #
# 源 1：GitHub 镜像（并行下载原始图像）
# --------------------------------------------------------------------------- #
def _mirror_urls(name: str) -> list[str]:
    """同一个文件在所有镜像上的候选 URL。"""
    return [
        tpl.format(repo=GITHUB_REPO, branch=GITHUB_BRANCH, path=f"data/raw/{name}")
        for tpl in IMAGE_MIRRORS
    ]


def _download_image(name: str, dest_dir: Path, offset: int = 0) -> None:
    """下载单张图片：在镜像间轮转重试。"""
    dest = dest_dir / name
    urls = _mirror_urls(name)
    last_err: Exception | None = None
    for attempt in range(len(urls) * 2):
        url = urls[(offset + attempt) % len(urls)]
        try:
            _download(url, dest, retries=1, timeout=45)
            if dest.stat().st_size > 0:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.3)
    raise IOError(f"所有镜像均失败 {name}: {last_err}")


def _github_file_list() -> list[str]:
    """通过 GitHub API 获取镜像仓库 data/raw 下的全部图片文件名。"""
    url = f"{GITHUB_API}/git/trees/{GITHUB_BRANCH}?recursive=1"
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(_request(url), timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            names = [
                Path(item["path"]).name
                for item in payload.get("tree", [])
                if item.get("type") == "blob"
                and item["path"].startswith("data/raw/")
                and Path(item["path"]).suffix.lower() in IMAGE_SUFFIXES
            ]
            if not names:
                raise IOError("GitHub 镜像返回的文件列表为空")
            return sorted(names)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise IOError(f"获取文件清单失败：{last_err}")


def download_from_github(dest_dir: Path = RAW_DIR, workers: int = 16) -> int:
    """从 GitHub 镜像并行下载 Flavia 原始图像（支持断点续传）。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    names = _github_file_list()
    todo = [n for n in names if not (dest_dir / n).exists()]
    print(f"[github] 清单共 {len(names)} 张，待下载 {len(todo)} 张，并发 {workers}")
    if not todo:
        return len(names)

    done = 0
    failed: list[str] = []
    start = time.time()

    def _one(item: tuple[int, str]) -> None:
        index, name = item
        _download_image(name, dest_dir, offset=index % len(IMAGE_MIRRORS))

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, item): item[1] for item in enumerate(todo)}
        for fut in cf.as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                done += 1
                if done % 100 == 0 or done == len(todo):
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - done) / rate if rate > 0 else 0
                    print(
                        f"  已下载 {done}/{len(todo)} ({done / len(todo) * 100:.1f}%)  "
                        f"{rate:.1f} 张/秒  预计剩余 {eta / 60:.1f} 分钟",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                failed.append(name)
                print(f"  [警告] {name} 下载失败: {exc}", file=sys.stderr)

    if failed:
        print(
            f"[github] 有 {len(failed)} 张失败，重新运行本命令可断点续传",
            file=sys.stderr,
        )
    return len(names) - len(failed)


# --------------------------------------------------------------------------- #
# 源 2：SourceForge 官方 tar.bz2
# --------------------------------------------------------------------------- #
def download_from_sourceforge(dest_dir: Path = RAW_DIR) -> int:
    """下载官方 Leaves.tar.bz2 并解压。"""
    from .config import SOURCEFORGE_URL

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "Leaves.tar.bz2"
        print("[sourceforge] 正在下载官方数据集（约 30MB）...")
        size = _download(SOURCEFORGE_URL, archive, retries=2, timeout=120)
        print(f"[sourceforge] 下载完成 {size / 1e6:.1f} MB，正在解压...")
        with tarfile.open(archive, "r:bz2") as tar:
            members = [
                m for m in tar.getmembers()
                if m.isfile() and Path(m.name).suffix.lower() in IMAGE_SUFFIXES
            ]
            for member in members:
                member.name = Path(member.name).name  # 去掉目录层级，平铺
                tar.extract(member, dest_dir)
    return len(list(dest_dir.glob("*")))


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def verify(dest_dir: Path = RAW_DIR, strict: bool = False) -> dict:
    """校验图像数量与标签分布。

    :param strict: 为 True 时，数量不符会抛出异常
    """
    files = sorted(
        p for p in dest_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    per_class: dict[int, int] = {}
    bad: list[str] = []
    for path in files:
        label = label_from_filename(path.name)
        if label is None:
            bad.append(path.name)
            continue
        per_class[label] = per_class.get(label, 0) + 1

    info = {
        "root": str(dest_dir),
        "count": len(files),
        "expected": EXPECTED_COUNT,
        "unlabeled": bad,
        "num_classes": len(per_class),
        "per_class": {k: per_class[k] for k in sorted(per_class)},
    }
    if strict and (len(files) != EXPECTED_COUNT or bad):
        raise RuntimeError(
            f"数据校验未通过：找到 {len(files)} 张图像（期望 {EXPECTED_COUNT}），"
            f"无法解析标签 {len(bad)} 张"
        )
    return info


def print_verify(info: dict) -> None:
    from .species import label_to_display

    print(f"数据目录 : {info['root']}")
    print(f"图像数量 : {info['count']} / {info['expected']}")
    print(f"物种数量 : {info['num_classes']}")
    if info["unlabeled"]:
        print(f"无法解析标签: {len(info['unlabeled'])} 张，例如 {info['unlabeled'][:3]}")
    print("每类样本数:")
    for label, n in info["per_class"].items():
        print(f"  {label:>2d}  {label_to_display(label):<36} {n:>3d}")


def run(
    source: str = "auto",
    dest_dir: Path = RAW_DIR,
    force: bool = False,
    workers: int = 16,
) -> Path:
    """下载（如需要）+ 校验，返回数据目录。

    :param source: ``auto`` / ``github`` / ``sourceforge``
    :param force: 已有数据时是否强制重新下载
    :param workers: 并行下载线程数
    """
    ensure_dirs()
    info = verify(dest_dir)
    if not force and info["count"] >= EXPECTED_COUNT:
        print(f"检测到 {info['count']} 张图像，跳过下载。")
        print_verify(info)
        return dest_dir

    errors: list[str] = []
    candidates = ["github", "sourceforge"] if source == "auto" else [source]
    for name in candidates:
        try:
            if name == "github":
                download_from_github(dest_dir, workers=workers)
            elif name == "sourceforge":
                download_from_sourceforge(dest_dir)
            else:
                raise ValueError(f"未知下载源: {name}")
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            print(f"[{name}] 失败：{exc}", file=sys.stderr)

    info = verify(dest_dir)
    print_verify(info)
    if info["count"] == 0:
        raise RuntimeError(
            "所有下载源均失败。请手动下载 Flavia 数据集后解压到 "
            f"{dest_dir}\n" + "\n".join(errors)
        )
    if info["count"] < EXPECTED_COUNT:
        print(
            f"[提示] 当前只有 {info['count']} 张（完整数据集为 {EXPECTED_COUNT} 张），"
            "可再次运行 download 命令续传。",
            file=sys.stderr,
        )
    return dest_dir


def cleanup_partial(dest_dir: Path = RAW_DIR) -> int:
    """删除体积异常/损坏的图片文件，返回删除数量。"""
    removed = 0
    for path in dest_dir.iterdir():
        if path.is_file() and path.stat().st_size < 1024:
            path.unlink()
            removed += 1
    return removed


__all__ = [
    "run",
    "verify",
    "print_verify",
    "download_from_github",
    "download_from_sourceforge",
    "cleanup_partial",
    "EXPECTED_COUNT",
]
