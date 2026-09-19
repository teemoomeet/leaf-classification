# 🍃 Flavia 树叶识别 · Leaf Species Recognition

给一批树叶照片，自动识别出**这是哪一种树**。

项目基于经典的 **Flavia 叶片数据集**（南京大学 / 南京中山植物园采集，32 个树种、1907 张叶片图像），
提供了一条从**数据下载 → 分割 → 特征提取 → 训练 → 批量识别 → 结果可视化**的完整可运行流水线。
纯 Python，**不需要 GPU**，一台普通笔记本几分钟就能跑完整个流程。

---

## 目录

- [效果预览](#效果预览)
- [快速开始（3 步）](#快速开始3-步)
- [环境配置](#环境配置)
- [命令行用法](#命令行用法)
- [两种技术路线](#两种技术路线)
- [批量识别自己的照片](#批量识别自己的照片)
- [输出结果说明](#输出结果说明)
- [项目结构](#项目结构)
- [算法流程](#算法流程)
- [数据集说明](#数据集说明)
- [常见问题](#常见问题)
- [引用与致谢](#引用与致谢)

---

## 效果预览

识别一张图片的完整过程：**抠出叶片 → 比对 32 个树种 → 给出结论与置信度**。

```text
$ python -m leaves.cli predict --input samples

待识别图片 11 张，目录：samples
--------------------------------------------------------------------
  1001.jpg                     -> 毛竹      (Phyllostachys edulis)  置信度 0.897
  1060.jpg                     -> 七叶树     (Aesculus chinensis)  置信度 0.852
  1268.jpg                     -> 鸡爪槭     (Acer palmatum)  置信度 0.981
  2001.jpg                     -> 大果冬青    (Ilex macrocarpa)  置信度 0.995
  2424.jpg                     -> 银杏      (Ginkgo biloba)  置信度 0.952
  2616.jpg                     -> 罗汉松     (Podocarpus macrophyllus)  置信度 0.994
  3056.jpg                     -> 日本晚樱    (Prunus serrulata)  置信度 0.212  ⚠ 低置信度
  3282.jpg                     -> 三角槭     (Acer buergerianum)  置信度 0.964
  3390.jpg                     -> 荷花木兰    (Magnolia grandiflora)  置信度 0.949
  3511.jpg                     -> 鹅掌楸     (Liriodendron chinense)  置信度 0.960
  3566.jpg                     -> 柑橘      (Citrus reticulata)  置信度 0.927
--------------------------------------------------------------------
识别完成：成功 11 / 11 张
```

11 张全部识别正确，其中 `3056.jpg` 置信度只有 0.21，被自动打上 **⚠ 低置信度** 标记 ——
这正是置信度校准的作用：模型知道「这张我拿不准」，把这类样本交给人工复核。

同时在 `outputs/predict_samples/` 下导出：

| 文件 | 内容 |
| --- | --- |
| `predictions.csv` | 每张图的预测结果（中文名 / 拉丁学名 / 置信度 / Top-3 候选 / 是否低置信度） |
| `predictions.json` | 同样的结果，JSON 格式，便于程序二次消费 |
| `vis/recognition_grid.png` | 可视化拼图：原图 + 分割轮廓 + 预测标签 |

### 实测性能

Flavia 全集（1907 张，32 类）按 8:1:1 分层划分，测试集 380 张：

| 技术路线 | 测试集准确率 | Top-3 准确率 | 5 折交叉验证 | 特征提取耗时 |
| --- | --- | --- | --- | --- |
| `--backend features`（手工特征 + SVM，默认） | **97.89%** | **98.95%** | 95.54% ± 2.99% | ~2.4 分钟（6 线程） |
| `--backend cnn --arch mobilenet_v3_small` | **99.21%** | — | — | ~1.0 分钟（CPU） |

两种路线都能用，深度学习路线精度更高，手工特征路线可解释性更好、零额外依赖。

---

## 快速开始（3 步）

> 前提：已安装 [Python 3.9+](https://www.python.org/downloads/)（安装时记得勾选 *Add Python to PATH*）

```bash
# 0. 进入项目目录
cd D:\leaves

# 1. 配置环境（自动建虚拟环境 + 装依赖，Windows 用这条）
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

#    Linux / macOS / Git Bash 用这条：
#    python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. 激活虚拟环境
.\.venv\Scripts\Activate.ps1      # Windows
# source .venv/bin/activate       # Linux / macOS

# 3. 一键跑通：下载数据 → 训练 → 评估 → 批量识别演示
python -m leaves.cli demo
```

跑完在 `outputs/` 里就能看到评估报告和识别结果。

想分步执行、看得更清楚：

```bash
python -m leaves.cli download             # 下载 1907 张 Flavia 叶片图像到 data/raw
python -m leaves.cli train                # 提特征 + 训练 SVM，模型存到 models/
python -m leaves.cli evaluate             # 在测试集上评估，输出混淆矩阵等
python -m leaves.cli predict -i samples   # 批量识别 samples/ 目录下的图片
```

---

## 环境配置

### 方式一：一键脚本（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

脚本会创建 `.venv`、用清华镜像安装依赖、并自检所有关键库。

### 方式二：手动安装

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows
pip install --upgrade pip
pip install -r requirements.txt         # 基础依赖
pip install -r requirements-cnn.txt     # 可选：深度学习路线
```

基础依赖（`requirements.txt`）：

| 库 | 用途 |
| --- | --- |
| `numpy` / `scipy` | 数值计算 |
| `opencv-python` | 图像读写、分割、轮廓、形态学 |
| `scikit-learn` | 特征标准化、SVM / 随机森林 / KNN 等分类器、评估指标 |
| `pandas` | 结果表格与 CSV 导出 |
| `matplotlib` | 混淆矩阵、类别分布、识别结果拼图 |
| `joblib` | 模型序列化 |
| `Pillow` / `tqdm` | 图片兼容与进度显示 |

可选依赖（`requirements-cnn.txt`）：`torch` + `torchvision`，仅 `--backend cnn` 时需要。

### 自检

```bash
python tests/test_smoke.py
```

不依赖完整数据集，几秒钟跑完，会检查物种映射表、叶片分割、特征维度一致性、模型训练等。

---

## 命令行用法

统一入口：`python -m leaves.cli <子命令>`（`pip install -e .` 之后也可以直接写 `leaves <子命令>`）

### `download` — 下载数据集

```bash
python -m leaves.cli download                       # 自动选择可用下载源
python -m leaves.cli download --source github       # 指定 GitHub 镜像
python -m leaves.cli download --source sourceforge  # 指定 SourceForge 官方源
python -m leaves.cli download --workers 24          # 调大并发线程数
```

支持**断点续传**：中途断掉再跑一次即可，已下载的文件会跳过。

### `info` — 查看物种表与数据概况

```bash
python -m leaves.cli info
python -m leaves.cli info --plot        # 顺便导出类别分布图
```

### `train` — 训练模型

```bash
python -m leaves.cli train                                  # 默认 SVM + 手工特征
python -m leaves.cli train -c rf                            # 换随机森林
python -m leaves.cli train --backend cnn --arch resnet18    # 深度特征
python -m leaves.cli train --cv 5                           # 额外做 5 折交叉验证
```

主要参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-c, --classifier` | `svm` | `svm` / `linearsvm` / `rf` / `extratrees` / `knn` / `logreg` |
| `-b, --backend` | `features` | `features` 手工特征、`cnn` 深度特征 |
| `--arch` | `mobilenet_v3_small` | `--backend cnn` 时的骨干网络 |
| `--max-side` | `384` | 特征提取前把图缩到最长边不超过该值（速度 / 精度权衡） |
| `--test-size` / `--val-size` | `0.2` / `0.1` | 测试集 / 验证集比例 |
| `--workers` | `4` | 手工特征提取线程数 |
| `--batch-size` | `32` | 深度特征批大小 |
| `--cv` | `0` | 大于 1 时做 K 折交叉验证 |
| `--model` | `models/flavia_svm.joblib` | 模型保存路径 |

### `evaluate` — 评估模型

```bash
python -m leaves.cli evaluate                  # 默认在测试集上评估
python -m leaves.cli evaluate --split val      # 换成验证集
```

输出：总体准确率、Top-K 准确率、逐类分类报告、混淆矩阵图、各类别准确率图、错分样本清单。

### `predict` — 批量识别（核心功能）

```bash
python -m leaves.cli predict --input D:\my_leaves
python -m leaves.cli predict -i photo.jpg -o outputs\single
python -m leaves.cli predict -i D:\my_leaves --no-recursive --threshold 0.5
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-i, --input` | 必填 | 待识别的图片目录或单张图片 |
| `--model` | 自动 | 模型文件，支持 `.joblib`（训练产物）与 `.pt`（微调产物） |
| `-o, --output` | `outputs/predict_<目录名>` | 结果输出目录 |
| `--top-k` | `3` | 输出前 K 个候选 |
| `--threshold` | `0.35` | 置信度低于该值标记为「低置信度」，提示人工复核 |
| `--whiten` | `auto` | 白底对齐预处理：先把叶片抠到纯白背景再识别，真实照片推荐开启；`off` 关闭 |
| `--no-recursive` | 关 | 不递归子目录 |
| `--no-vis` | 关 | 不导出可视化拼图 |
| `--max-vis` | `60` | 可视化最多输出多少张 |

### `finetune` — 微调 CNN（真实照片鲁棒性）

冻结特征 + 分类器在「白底扫描图」上接近 99%，但手机实拍照片（多叶簇生、
背景杂物、色偏、水珠反光）与扫描图分布差异大，直接推理容易误判。
`finetune` 用**域随机化**合成数据做端到端微调：

* **扫描模式**：原始白底图随机旋转/抖色 —— 保住扫描图上的判别力；
* **野生模式**：把同物种 1~6 片叶随机贴到随机背景（白纸/灰桌/暗色虚化/
  绿色灌丛/木纹色），叠加色彩抖动与模糊 —— 模拟真实拍摄条件。

```bash
python -m leaves.cli finetune                  # CPU 约 25 分钟，输出 models/flavia_ft_mobilenetv3.pt
python -m leaves.cli predict -i 照片目录 --model models/flavia_ft_mobilenetv3.pt
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model` | `models/flavia_ft_mobilenetv3.pt` | 微调模型保存路径 |
| `--size` | `160` | 输入边长（CPU 友好） |
| `--epochs` | `12` | 训练轮数 |
| `--wild-prob` | `0.65` | 野生模式样本占比 |
| `--batch-size` | `32` | 批大小 |
| `--seed` | `42` | 随机种子 |

### `match` — 参考图库匹配（识别 32 种之外的植物）

闭集分类只能识别 Flavia 的 32 种树。遇到农作物、园艺植物等**不在表内**
的物种时，用 `match`：准备一个参考图库目录，**图片文件名即物种名**
（支持子目录、每类多张），程序提取特征构建参考库，按余弦相似度匹配
未知图片。不指定 `--input` 时自动做留一法自检，验证参考库区分力。

```bash
# 构建参考库并自检
python -m leaves.cli match --library D:/my_refs
# 批量识别未知图片
python -m leaves.cli match --library D:/my_refs --input D:/unknown
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-l, --library` | 必填 | 参考图库目录：文件名即物种名，如「苹果.jpg」「Vitis vinifera.jpg」 |
| `-i, --input` | 无 | 待识别图片；缺省时对参考库做留一法自检 |
| `-b, --backend` | `cnn` | 特征后端：`cnn` 区分度最好（需 PyTorch），`features` 无需 GPU |
| `--threshold` | `0.55` | top1 相似度低于该值提示「可能不在参考库中」 |
| `--rebuild` | 关 | 参考库加了新图后强制重建缓存 |
| `--whiten` | `auto` | 白底对齐预处理，真实照片推荐开启 |

开集行为：top1 相似度低于阈值时给出「可能不在参考库中」的提示，而不是硬猜。

---

## 两种技术路线

项目把两条路线封装在同一个命令行接口下，靠 `--backend` 切换。

### `--backend features`（默认）：手工特征 + 传统机器学习

对每张图提取 **99 维可解释特征**：

| 分组 | 维度 | 内容 |
| --- | --- | --- |
| **形状** | 21 + 16 | 面积、周长、凸包面积、solidity（凸度）、circularity（圆度）、compactness、convexity、长短轴、aspect ratio、eccentricity、rectangularity、等效直径、**生理长宽**、7 个 **Hu 不变矩**，以及 16 个**傅里叶描述子**（轮廓 FFT 归一化幅度谱，具备旋转 / 缩放 / 起点不变性） |
| **纹理** | 28 + 15 | 两套尺度的**旋转不变 uniform LBP** 直方图（8 邻域 / 半径 1、16 邻域 / 半径 2），以及 3 个距离上平均 4 个方向的 **GLCM** 统计量（对比度、相异性、同质性、能量、相关性） |
| **颜色** | 12 | 叶片区域内 B / G / R 与 H / S / V 的均值与标准差 |

然后 `StandardScaler` + `SVC(RBF, C=10)` 分类。

**优点**：零 GPU 依赖、训练通常在 10 秒级、每个特征都能说清含义、便于做特征分析。
**适用**：教学、可解释性要求高、算力受限的场景。

### `--backend cnn`：预训练 CNN 深度特征

先用分割把叶片抠出来、背景填白，再送入 ImageNet 预训练骨干网络，取全局池化特征，最后接 SVM：

| `--arch` | 特征维度 |
| --- | --- |
| `mobilenet_v3_small`（默认） | 576 |
| `resnet18` | 512 |
| `efficientnet_b0` | 1280 |

**为什么冻结骨干网络而不做端到端微调？**
在 CPU 上对 1907 张图做完整微调要几十分钟，而只做一次前向推理只要一两分钟；
冻结特征的 ImageNet 骨干 + SVM 在这个数据集上精度已经很高，性价比明显更好。
特征向量会缓存下来，换分类器做实验几乎零成本。

---

## 批量识别自己的照片

### 只想识别，不想训练

```bash
python -m leaves.cli predict --input D:\我的树叶照片
```

把手机拍的树叶照片丢进一个文件夹即可，程序会自动抠出叶片、复用已训练好的模型、输出 CSV 结果。

> **拍摄建议**：单片叶子放在浅色平整背景（白纸、浅色桌面）上拍摄，尽量让叶片占满画面、
> 正面朝上、光照均匀。Flavia 数据集里的图像都是这种「白底单片叶」，背景越接近，识别越准。

### 真实场景照片：白底对齐 + 微调模型

`predict` 默认开启 `--whiten auto` 白底对齐：先用「绿色 HSV 掩码 + 亮度阶梯
阈值」把叶片从复杂背景里抠出来贴到纯白背景上，再送入与训练分布一致的识别
流程。对白纸/灰桌/暗背景/绿色灌丛里的叶片都稳健。

对多叶簇生的实拍照片（枝条、叶序、竹叶），推荐配合微调模型使用：

```bash
python -m leaves.cli finetune                                    # 一次性微调
python -m leaves.cli predict -i D:\实拍照片 --model models/flavia_ft_mobilenetv3.pt
```

**实测案例**（`samples/scene_photos/`，4 张网络获取的真实场景照）：

| 照片内容 | 肉眼参考 | 冻结特征 SVM | 微调模型 + 多视角 |
| --- | --- | --- | --- |
| 暗背景竹叶簇 | 毛竹 | 荷花木兰（误） | 鸡爪槭（误） |
| 黄绿心形叶 | 紫荆 | 鹅掌楸（误） | 沙兰杨（误） |
| 白底双叶 | 珊瑚树 / 柑橘 | 日本珊瑚树（合理） | 阔叶十大功劳（低置信 ⚠） |
| 绿色灌丛轮生枝 | 珊瑚树 / 柑橘 | 三角槭（误） | 柑橘 0.39（合理） |

> **坦率的结论**：扫描图 → 实拍照片的域差是真实存在的。白底单片叶是本项目的
> 可靠区间（测试集 97.9%~99.2%）；复杂场景照属于「尽力而为 + 明确报告不确定」
> ——置信度会显著下降并触发 `low_confidence` 提示，请以 top-3 候选与人工复核
> 为准。要对 32 类之外的植物获得可靠结果，请使用 `match` 自建参考图库路线；
> 要提升场景照识别，最有效的办法是补充**真实拍摄**的样本继续微调，
> 而不是依赖合成数据。

### 想用自己的数据重新训练

按 **`<物种名>/图片.jpg`** 组织目录，物种名支持中文名、拉丁学名或别名（见 `python -m leaves.cli info`）：

```text
my_dataset/
├── 银杏/
│   ├── 001.jpg
│   └── 002.jpg
├── 桂花/
│   └── 003.jpg
└── Ginkgo biloba/       # 拉丁学名也认
    └── 004.jpg
```

然后：

```bash
python -m leaves.cli train --data my_dataset --model models/my_model.joblib
python -m leaves.cli predict -i some_photos --model models/my_model.joblib
```

### 在 Python 里调用

```python
from leaves.predict import PredictOptions, recognize_folder

frame = recognize_folder("D:/我的树叶照片", options=PredictOptions(top_k=3))
print(frame[["file", "pred_name", "scientific_name", "confidence"]])
```

---

## 输出结果说明

`predict` 产出的 `predictions.csv` 主要列：

| 列名 | 含义 |
| --- | --- |
| `file` / `path` | 文件名 / 完整路径 |
| `pred_name` | 预测的中文物种名 |
| `scientific_name` | 预测的拉丁学名 |
| `confidence` | 置信度（0 ~ 1） |
| `low_confidence` | 是否低于阈值，`True` 表示「可能不属于这 32 种树，建议人工复核」 |
| `top1_name` / `top1_score` … | Top-K 候选及对应分数 |
| `leaf_area_ratio` | 叶片在画面中的面积占比（可用于排查拍摄问题） |
| `segmented` | 叶片分割是否成功 |
| `status` | `ok` / `读取失败` |

`evaluate` 产出的文件（在 `outputs/evaluation/`）：

| 文件 | 内容 |
| --- | --- |
| `classification_report.txt` | 每个树种的 precision / recall / F1 |
| `confusion_matrix.png` | 32×32 混淆矩阵热力图（行归一化） |
| `per_class_accuracy.png` | 各类别准确率排序条形图 |
| `predictions.csv` | 测试集每条样本的详细预测 |
| `summary.json` | 汇总指标 |

---

## 项目结构

```text
D:\leaves\
├── leaves/                     # 主程序包
│   ├── __init__.py
│   ├── __main__.py             # python -m leaves 入口
│   ├── cli.py                  # 命令行：download / info / train / finetune / evaluate / predict / match / demo
│   ├── config.py               # 路径与全局配置
│   ├── species.py              # 32 个树种的名称表 + 文件名→标签映射
│   ├── registry.py             # 分类器 / 后端 / 骨干网络注册表（纯标准库）
│   ├── download.py             # 数据集下载（多源自动回退 + 断点续传 + 校验）
│   ├── dataset.py              # 数据集扫描、分层划分、特征缓存
│   ├── segmentation.py         # 叶片分割（自适应 Otsu + 绿色掩码白底对齐 + 亮度阶梯阈值）
│   ├── features.py             # 99 维手工特征（形状 / 傅里叶 / LBP / GLCM / 颜色）
│   ├── deep.py                 # 预训练 CNN 深度特征提取
│   ├── finetune.py             # 端到端微调（域随机化：扫描模式 + 野生模式合成）
│   ├── torch_model.py          # 微调 .pt 模型的推理封装
│   ├── backends.py             # 统一特征后端接口
│   ├── models.py               # 分类器工厂 + 模型存取 + 置信度计算
│   ├── train.py                # 训练流水线
│   ├── evaluate.py             # 评估流水线
│   ├── predict.py              # 批量识别（核心功能，支持 .joblib 与 .pt 模型）
│   ├── reference.py            # 参考图库匹配（识别 32 种之外的植物，开集）
│   └── viz.py                  # 可视化（混淆矩阵 / 拼图 / 分布图）
├── scripts/
│   ├── setup_env.ps1           # 一键环境配置（Windows）
│   ├── quickstart.ps1|.sh      # 一键跑通全流程
│   ├── fix_github_hosts.ps1|.bat  # 修复 github.com 无法访问（网络受限环境）
│   └── push_via_api.py         # 通过 GitHub API 推送代码（不依赖 github.com 直连）
├── tests/
│   └── test_smoke.py           # 冒烟测试（无需完整数据集）
├── samples/                    # 示例图片：白底扫描叶片 + scene_photos 实拍场景照
├── docs/
│   └── sample_predict_output.txt  # 示例识别输出（README 中的效果预览来源）
├── requirements.txt            # 基础依赖
├── requirements-cnn.txt        # 可选深度学习依赖
├── pyproject.toml
├── data/                       # 【运行时生成】数据集与特征缓存
│   ├── raw/                    #   1907 张原始图片
│   ├── splits/split.json       #   训练 / 验证 / 测试划分
│   └── features_handcrafted.npz  #   特征缓存（按后端分别缓存）
├── models/                     # 【运行时生成】训练好的模型
└── outputs/                    # 【运行时生成】评估报告与识别结果
```

> `data/`、`models/`、`outputs/` 体积较大，已在 `.gitignore` 中排除，不会进入版本库。

---

## 算法流程

```text
                  ┌───────────────────────────────────────────┐
   输入图片  ───►  │ 1. 分割：自适应 Otsu + 形态学 + 最大连通域  │ ───► 叶片掩码
                  └───────────────────────────────────────────┘
                                     │
                  ┌──────────────────┴───────────────────┐
                  ▼                                      ▼
      ┌────────────────────────┐            ┌───────────────────────────┐
      │ 2a. 手工特征（99 维）    │            │ 2b. 深度特征（576/512/…）  │
      │  形状+傅里叶+纹理+颜色   │            │  预训练 CNN 全局池化特征    │
      └────────────────────────┘            └───────────────────────────┘
                                     │
                  ┌──────────────────┴───────────────────┐
                  ▼                                      │
      ┌────────────────────────┐                          │
      │ 3. 标准化 + 分类器       │ ◄────────────────────────┘
      │   SVC(RBF) / RF / KNN   │
      └────────────────────────┘
                                     │
                  ┌──────────────────┴───────────────────┐
                  ▼
      ┌───────────────────────────────────────────┐
      │ 4. 输出：中文名 / 拉丁学名 / 置信度 / Top-3 │
      │    导出 CSV、JSON、可视化拼图               │
      └───────────────────────────────────────────┘
```

分割这一步用了个小技巧：先看图像最外圈像素的平均亮度判断背景是亮还是暗，
再决定 Otsu 阈值分割取正还是取反。这样白底扫描件和灰底照片都能正确处理。

---

## 数据集说明

**Flavia 叶片数据集** —— 官方主页：<http://flavia.sourceforge.net/>

| 项目 | 说明 |
| --- | --- |
| 图像总数 | 1907 张 |
| 物种数 | 32 个 |
| 每类数量 | 50 ~ 77 张 |
| 采集方式 | 扫描仪 / 数码相机，浅色平整背景，仅含叶片本体（无叶柄） |
| 采集地点 | 南京大学校园、南京中山植物园 |
| 特点 | 长江三角洲地区常见树种 |

32 个树种（标签顺序）：

> 毛竹、七叶树、安徽小檗、紫荆、木蓝、鸡爪槭、楠木、刺楸、天竺桂、栾树、大果冬青、海桐、
> 蜡梅、香樟、日本珊瑚树、桂花、雪松、银杏、紫薇、夹竹桃、罗汉松、日本晚樱、女贞、香椿、
> 桃、木莲、三角槭、阔叶十大功劳、荷花木兰、沙兰杨、鹅掌楸、柑橘

运行 `python -m leaves.cli info` 可以看到完整的中文名 / 拉丁学名 / 别名对照表。

### 下载源

官方 SourceForge 源在部分网络环境下会被 Cloudflare 拦截，因此 `download` 命令内置了多个来源，
按顺序自动回退，任何一个成功即可：

1. **GitHub 镜像**（默认首选，含完整 1907 张原图）
2. **SourceForge 官方 tar.bz2**

并且对单个文件也准备了多条 CDN 线路（`raw.githubusercontent.com` / jsDelivr / ghproxy），
在不同线路之间自动轮转重试。

数据量约 **1.1 GB**。如果只想快速体验，可以随时中断——
项目支持断点续传，已下载的图片不会重复下载，而且用部分数据也能完成训练（精度会低一些）。

---

## 常见问题

<details>
<summary><b>github.com 打不开、git push 超时怎么办？</b></summary>

部分网络（校园网 / 公司出口）会把 `github.com` 解析到一个被屏蔽的接入 IP
（常见是 `20.205.243.166`），表现是网页打不开、`git push` 卡死，但
`api.github.com` 仍然正常。两个办法：

**办法一：修复 hosts（推荐，一次修好后网页和 git 都正常）**

```powershell
# 双击 scripts\fix_github_hosts.bat（会弹 UAC 确认）
# 或以管理员身份运行 PowerShell：
powershell -ExecutionPolicy Bypass -File scripts\fix_github_hosts.ps1
```

脚本会自动探测一批 GitHub 接入 IP，挑出当前最快且稳定的一个写入 hosts，
并刷新 DNS。可用 `-Restore` 参数一键还原。

> 这些 IP 的可用性会随时间变化，如果哪天又连不上，重新跑一次脚本即可换 IP。

**办法二：改用 API 推送（不依赖 github.com 直连）**

只要能拿到一个 Personal Access Token，就可以走 `api.github.com` 把代码推上去：

```powershell
$env:GITHUB_TOKEN = "ghp_xxxxxxxx"
python scripts/push_via_api.py --repo flavia-leaf-recognition
```
</details>

<details>
<summary><b>下载很慢或者总是失败怎么办？</b></summary>

1. 用 `--workers` 调大并发：`python -m leaves.cli download --workers 32`
2. 多跑几次 `download`，它是断点续传的，只会补缺失的文件。
3. 手动下载：从 <http://flavia.sourceforge.net/> 或
   [SourceForge 官方链接](https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download)
   下载 `Leaves.tar.bz2`，解压后把所有 `.jpg` 直接放到 `data/raw/` 下即可。
   脚本同时兼容官方 `3.17.jpg` 与镜像 `1001.jpg` 两种命名。
</details>

<details>
<summary><b>提示「没有找到可识别的叶片图像」？</b></summary>

检查 `data/raw/` 里是不是真的有图片。文件名必须能解析出物种，支持两种命名：
官方 `3.17.jpg`（第 3 种第 17 张）或镜像 `1001.jpg`（按编号区间映射）。
如果是自己的数据，请按 `<物种名>/图片.jpg` 组织目录。
</details>

<details>
<summary><b>用自己的照片识别，结果不太准？</b></summary>

这是正常现象。Flavia 是**白底单片叶、正面朝上**的受控图像，模型学到的分布也是这个。
换成自然背景下拍的照片，域差异会明显拉低精度。可以：

- 拍摄时尽量贴近数据集风格（浅色平整背景、叶片正面朝上、光照均匀）；
- 或者用自己的数据按 `<物种名>/` 重新训练（几十张 / 类就能有明显提升）；
- 也可以试试 `--backend cnn --arch resnet18`，深度特征对背景变化更鲁棒。
</details>

<details>
<summary><b>置信度准不准？是怎么算出来的？</b></summary>

SVM 本身没有概率输出，而且多分类 SVM 的决策值走的是 one-vs-one 投票聚合，
取值区间很窄——直接 softmax 会得到「所有样本都约 0.63」这种毫无区分度的结果。
所以项目在 `leaves/models.py` 里给 SVM / LinearSVC 套了一层**概率校准**
（`CalibratedClassifierCV`，isotonic + 5 折 + ensemble）。

实测对比（测试集 380 张）：

| 方案 | 测试准确率 | 置信度分布 |
| --- | --- | --- |
| 原始 SVC（无概率） | 0.9789 | 无 |
| sigmoid + ensemble=False + cv=3 | 0.9447 | 中位数 0.633（几乎没有区分度） |
| sigmoid + ensemble=True + cv=5 | 0.8684 | 中位数 0.365（掉点严重） |
| **isotonic + ensemble=True + cv=5（本项目采用）** | **0.9789** | 中位数 0.960，5%~95% 分位 0.755~0.996 |

也就是说，校准后精度没有任何损失，而置信度变成了可以真正用来做判断的概率。
配合 `predict --threshold`（默认 0.35）就能自动把「模型拿不准」的样本挑出来送人工复核。
</details>

<details>
<summary><b>想加新的树种怎么办？</b></summary>

1. 在 `leaves/species.py` 的 `SPECIES` 列表末尾追加 `(拉丁学名, 中文名, 别名)`；
2. 如果沿用 Flavia 的编号命名，还要在 `FLAVIA_ID_RANGES` 里加上编号区间；
3. 准备好该树种的图片，重新训练。

`species.py` 里的物种表和模型元信息是一一对应的，改动后建议重跑一遍测试：
`python tests/test_smoke.py`
</details>

<details>
<summary><b>训练要多长时间？</b></summary>

以 1907 张、普通笔记本 CPU 为例（本项目实测）：

| 步骤 | 耗时 |
| --- | --- |
| 手工特征提取 | 约 2.4 分钟（6 线程，13 ~ 15 张/秒） |
| SVM + 概率校准训练 | 0.5 秒 |
| 5 折交叉验证 | 约 20 秒 |
| 深度特征提取（MobileNetV3-Small） | 约 1.0 分钟（CPU，30 张/秒） |

从零到出结果（下载数据除外）总共不到 5 分钟，完全不需要 GPU。
</details>

---

## 引用与致谢

如果这个项目对你的研究有帮助，请引用原始数据集论文：

> Wu, S. G., Bao, F. S., Xu, E. Y., Wang, Y.-X., Chang, Y.-F., & Xiang, Q.-L. (2007).
> A Leaf Recognition Algorithm for Plant Classification Using Probabilistic Neural Network.
> *IEEE International Symposium on Signal Processing and Information Technology (ISSPIT)*.

**Flavia 数据集**由南京大学与南京中山植物园采集并公开发布，版权归原作者所有，
本项目仅用于学习与研究用途。数据集官方地址：<http://flavia.sourceforge.net/>

相关的公开镜像仓库用于数据获取便利，在此一并致谢。

本项目代码采用 [MIT License](LICENSE)。
