#!/usr/bin/env bash
# =====================================================================
# Flavia 树叶识别 —— 一键跑通全流程（Linux / macOS / Git Bash）
#
# 用法：
#     bash scripts/quickstart.sh
# =====================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLASSIFIER="${CLASSIFIER:-svm}"
BACKEND="${BACKEND:-features}"

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
else
  echo "未找到虚拟环境，请先执行："
  echo "  python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo "=== Flavia 树叶识别：一键运行（$BACKEND / $CLASSIFIER）==="

echo; echo "[1/3] 下载数据集 ..."
"$PY" -m leaves.cli download

echo; echo "[2/3] 训练模型 ..."
"$PY" -m leaves.cli train --backend "$BACKEND" --classifier "$CLASSIFIER"

echo; echo "[3/3] 评估并演示批量识别 ..."
"$PY" -m leaves.cli evaluate --split test
"$PY" -m leaves.cli demo --backend "$BACKEND" --classifier "$CLASSIFIER"

echo; echo "全部完成！结果在 outputs/ 目录下。"
