"""项目路径与全局配置。"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

#: 数据根目录，可通过环境变量 ``LEAVES_DATA_DIR`` 覆盖
DATA_DIR = Path(os.environ.get("LEAVES_DATA_DIR", PROJECT_ROOT / "data")).resolve()
RAW_DIR = DATA_DIR / "raw"            # 原始图片
PROCESSED_DIR = DATA_DIR / "processed"  # 分割后的叶片 / 掩码（缓存）
SPLIT_DIR = DATA_DIR / "splits"        # 训练/验证/测试划分

MODEL_DIR = Path(os.environ.get("LEAVES_MODEL_DIR", PROJECT_ROOT / "models")).resolve()
OUTPUT_DIR = Path(os.environ.get("LEAVES_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()

#: 默认模型文件名
DEFAULT_MODEL_NAME = "flavia_svm.joblib"

#: 随机种子，保证可复现
RANDOM_SEED = 42

#: 缓存的特征文件
FEATURE_CACHE = DATA_DIR / "features.npz"
FEATURE_CSV = DATA_DIR / "features.csv"

#: 下载源提示（SourceForge 与 GitHub 镜像）
SOURCEFORGE_URL = (
    "https://sourceforge.net/projects/flavia/files/"
    "Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download"
)


def ensure_dirs() -> None:
    """创建项目运行所需的全部目录。"""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, SPLIT_DIR, MODEL_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
