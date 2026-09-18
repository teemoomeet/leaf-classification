"""Flavia 树叶图像识别 —— 批量识别树叶种类。

快速开始::

    python -m leaves.cli download    # 下载数据集
    python -m leaves.cli train       # 训练模型
    python -m leaves.cli predict --input some_folder --recursive

也可以直接使用 Python API::

    from leaves import predict
    results = predict.recognize_folder("my_leaves")
"""

from .config import PROJECT_ROOT, RAW_DIR, MODEL_DIR, OUTPUT_DIR  # noqa: F401
from .species import CHINESE_NAMES, NUM_CLASSES, SCIENTIFIC_NAMES, SPECIES  # noqa: F401

__version__ = "1.0.0"
__all__ = [
    "__version__",
    "CHINESE_NAMES",
    "SCIENTIFIC_NAMES",
    "SPECIES",
    "NUM_CLASSES",
    "PROJECT_ROOT",
    "RAW_DIR",
    "MODEL_DIR",
    "OUTPUT_DIR",
]
