"""Flavia 数据集 32 个树种的定义与文件名 -> 标签 的映射。

Flavia 数据集由南京大学与南京中山植物园采集，共 1907 张叶片图像、32 个树种，
每个树种 50~77 张。图像为白色背景下的扫描/拍摄叶片，只含叶片本体（无叶柄）。

数据集官方主页：http://flavia.sourceforge.net/

本模块提供两套文件名解析方案，兼容不同来源的副本：

1. ``ID 区间方案``（镜像仓库常用的重命名方式，如 ``1001.jpg``）
   通过 ``FLAVIA_ID_RANGES`` 把图片编号映射到物种。
2. ``类.序号 方案``（官方 tar 包中的原始命名，如 ``3.17.jpg``）
   文件名第一个数字即物种序号（从 1 开始），减一得到标签。
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# 物种表
# --------------------------------------------------------------------------- #
# 每一项: (拉丁学名, 中文名, 常见别名/备注)
SPECIES: list[tuple[str, str, str]] = [
    ("Phyllostachys edulis", "毛竹", "楠竹、猫头竹"),
    ("Aesculus chinensis", "七叶树", "娑罗树"),
    ("Berberis anhweiensis", "安徽小檗", "小檗属"),
    ("Cercis chinensis", "紫荆", "满条红"),
    ("Indigofera tinctoria", "木蓝", "槐蓝"),
    ("Acer palmatum", "鸡爪槭", "鸡爪枫"),
    ("Phoebe nanmu", "楠木", "滇楠、桢楠"),
    ("Kalopanax septemlobus", "刺楸", "刺楸树"),
    ("Cinnamomum japonicum", "天竺桂", "山肉桂"),
    ("Koelreuteria paniculata", "栾树", "灯笼树"),
    ("Ilex macrocarpa", "大果冬青", "冬青属"),
    ("Pittosporum tobira", "海桐", "海桐花"),
    ("Chimonanthus praecox", "蜡梅", "腊梅"),
    ("Cinnamomum camphora", "香樟", "樟树"),
    ("Viburnum awabuki", "日本珊瑚树", "法国冬青"),
    ("Osmanthus fragrans", "桂花", "木犀"),
    ("Cedrus deodara", "雪松", "喜马拉雅雪松"),
    ("Ginkgo biloba", "银杏", "白果树"),
    ("Lagerstroemia indica", "紫薇", "百日红"),
    ("Nerium oleander", "夹竹桃", "红花夹竹桃"),
    ("Podocarpus macrophyllus", "罗汉松", "罗汉杉"),
    ("Prunus serrulata", "日本晚樱", "晚樱"),
    ("Ligustrum lucidum", "女贞", "大叶女贞"),
    ("Toona sinensis", "香椿", "椿树"),
    ("Prunus persica", "桃", "毛桃"),
    ("Manglietia fordiana", "木莲", "木莲属"),
    ("Acer buergerianum", "三角槭", "三角枫"),
    ("Mahonia bealei", "阔叶十大功劳", "十大功劳"),
    ("Magnolia grandiflora", "荷花木兰", "广玉兰"),
    ("Populus x canadensis", "沙兰杨", "加杨"),
    ("Liriodendron chinense", "鹅掌楸", "马褂木"),
    ("Citrus reticulata", "柑橘", "橘子"),
]

NUM_CLASSES = len(SPECIES)

#: 拉丁学名列表（按标签顺序）
SCIENTIFIC_NAMES: list[str] = [s[0] for s in SPECIES]
#: 中文名列表（按标签顺序）
CHINESE_NAMES: list[str] = [s[1] for s in SPECIES]
#: 别名列表（按标签顺序）
ALIASES: list[str] = [s[2] for s in SPECIES]

#: 标签 -> 展示名（中文 + 拉丁学名）
ID_TO_LABEL: dict[int, str] = {i: f"{s[1]}({s[0]})" for i, s in enumerate(SPECIES)}

# --------------------------------------------------------------------------- #
# 方案 1：ID 区间映射
# --------------------------------------------------------------------------- #
#: (起始编号, 结束编号, 标签)。区间互不重叠且完整覆盖 1907 张图像。
FLAVIA_ID_RANGES: list[tuple[int, int, int]] = [
    (1001, 1059, 0),    # 毛竹
    (1060, 1122, 1),    # 七叶树
    (1552, 1616, 2),    # 安徽小檗
    (1123, 1194, 3),    # 紫荆
    (1195, 1267, 4),    # 木蓝
    (1268, 1323, 5),    # 鸡爪槭
    (1324, 1385, 6),    # 楠木
    (1386, 1437, 7),    # 刺楸
    (1497, 1551, 8),    # 天竺桂
    (1438, 1496, 9),    # 栾树
    (2001, 2050, 10),   # 大果冬青
    (2051, 2113, 11),   # 海桐
    (2114, 2165, 12),   # 蜡梅
    (2166, 2230, 13),   # 香樟
    (2231, 2290, 14),   # 日本珊瑚树
    (2291, 2346, 15),   # 桂花
    (2347, 2423, 16),   # 雪松
    (2424, 2485, 17),   # 银杏
    (2486, 2546, 18),   # 紫薇
    (2547, 2612, 19),   # 夹竹桃
    (2616, 2675, 20),   # 罗汉松
    (3001, 3055, 21),   # 日本晚樱
    (3056, 3110, 22),   # 女贞
    (3111, 3175, 23),   # 香椿
    (3176, 3229, 24),   # 桃
    (3230, 3281, 25),   # 木莲
    (3282, 3334, 26),   # 三角槭
    (3335, 3389, 27),   # 阔叶十大功劳
    (3390, 3446, 28),   # 荷花木兰
    (3447, 3510, 29),   # 沙兰杨
    (3511, 3563, 30),   # 鹅掌楸
    (3566, 3621, 31),   # 柑橘
]

#: 方案 2：官方 tar 包命名 ``<物种序号>.<图片序号>.<扩展名>``
_OFFICIAL_RE = re.compile(r"^(\d+)\s*[._-]\s*(\d+)")


def label_from_id(image_id: int) -> int:
    """把镜像副本中的图片编号（如 1001）映射到 0~31 的标签。"""
    for start, end, label in FLAVIA_ID_RANGES:
        if start <= image_id <= end:
            return label
    raise ValueError(f"图片编号 {image_id} 不在任何已知的 Flavia 编号区间内")


def label_from_filename(filename: str) -> int | None:
    """从文件名推断标签，识别不了则返回 ``None``。

    依次尝试：

    * 官方命名 ``3.17.jpg`` -> 标签 2
    * 镜像命名 ``1001.jpg`` -> 标签 0

    :param filename: 文件名或完整路径
    """
    stem = filename.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if stem.count(".") >= 1 else stem

    m = _OFFICIAL_RE.match(stem)
    if m:
        species_no = int(m.group(1))
        if 1 <= species_no <= NUM_CLASSES:
            return species_no - 1

    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits and len(digits) >= 4:
        try:
            return label_from_id(int(digits))
        except ValueError:
            return None
    return None


def label_to_display(label: int, with_scientific: bool = True) -> str:
    """标签 -> 展示用字符串。"""
    cn, sci = CHINESE_NAMES[label], SCIENTIFIC_NAMES[label]
    return f"{cn} ({sci})" if with_scientific else cn


__all__ = [
    "SPECIES",
    "NUM_CLASSES",
    "SCIENTIFIC_NAMES",
    "CHINESE_NAMES",
    "ALIASES",
    "ID_TO_LABEL",
    "FLAVIA_ID_RANGES",
    "label_from_id",
    "label_from_filename",
    "label_to_display",
]
