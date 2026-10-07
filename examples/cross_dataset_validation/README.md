# 跨数据集验证（Folio → Flavia 模型）

本目录包含两个跨数据集实验脚本，用于回答一个核心问题：

> 用 Flavia 数据集（32 种中国树木）训练的模型，遇到完全不同的物种时表现如何？

## 背景

| 数据集 | 物种 | 数量 | 类型 |
|--------|------|------|------|
| **Flavia**（训练） | 32 种中国树木（毛竹、七叶树、柑橘……） | 1907 张 | 白底扫描 |
| **Folio**（UCI #338） | 32 种作物/热带植物（茄子、咖啡、荔枝……） | 637 张 | 白底拍摄 |

两个数据集的物种**零重叠**，因此这是一个典型的「开集 / 分布外（OOD）」测试场景。

## 实验结果（已验证）

### 实验 1：直接迁移（`test_folio.py`）

**结论：准确率 0%（0/637），但这不是 bug，而是必然结果** —— 模型只能输出 32 个训练过的树种，喂给它茄子叶只能"硬选"一个最像的树种。

真正重要的是另一个发现：

> **模型判错时却异常自信** —— 平均置信度高达 0.686，低置信度仅占 4.1%。
> 个别误判置信度高达 0.902（多香果→安徽小檗）、0.939（锡兰醋栗）。

这说明模型遇到没见过的植物时，不会说"我不认识"，而是信心十足地给错误答案——这是部署到真实场景的最大风险。

### 实验 2：OOD 拒识（`test_ood.py`，方案 A）

用 **Mahalanobis 距离**在特征空间度量「这张图离已知 32 类有多远」，超过阈值就判为"未知"。

| 指标 | 结果 |
|------|------|
| 未知类（Folio）拒识率 | **94.5%** |
| 已知类（Flavia）通过率 | 95%（误拒仅 5%） |
| 接受样本识别正确率 | 98.9% |

**彻底解决了"过度自信"问题**：模型从"100% 硬判错误答案"变成"94.5% 正确说我不认识"。

## 使用方法

### 前置条件

1. 先训练模型（或已有 `models/flavia_svm.joblib` 等）
2. 生成 OOD 检测器（实验 2 需要）：
   ```bash
   python leaves/train_ood.py
   ```

### 实验 1：直接迁移测试

```bash
python examples/cross_dataset_validation/test_folio.py --folio <Folio 根目录>
```

`<Folio 根目录>` 是包含「以物种名命名的子文件夹」的目录，例如 `D:/folio/Folio Leaf Dataset/Folio`。

### 实验 2：OOD 拒识验证

```bash
python examples/cross_dataset_validation/test_ood.py --folio <Folio 根目录>
```

### 实际使用 OOD 拒识（已接入主流程）

```bash
# 命令行：识别目录，陌生叶子自动判"未知"
python -m leaves.cli predict --input <图片或目录> --reject-unknown
```

```python
# Python 接口
from leaves.predict import PredictOptions, recognize_folder
recognize_folder("D:/some/leaves", options=PredictOptions(reject_unknown=True))
```

## 核心实现

- `leaves/ood.py` —— `MahalanobisOOD` 检测器（pooled 协方差）
- `leaves/train_ood.py` —— 训练脚本，输出 `models/flavia_ood_mahalanobis.joblib`
- `leaves/predict.py` —— `recognize()` 已接入 `reject_unknown` 参数

## 参考

- Folio 数据集：https://archive.ics.uci.edu/dataset/338/folio
- Flavia 数据集：http://flavia.sourceforge.net/
