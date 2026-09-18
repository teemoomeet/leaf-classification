# =====================================================================
# Flavia 树叶识别 —— 一键跑通全流程
#
# 用法（在项目根目录 D:\leaves 下执行）：
#     powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1
#
# 参数：
#     -Classifier svm|linearsvm|rf|extratrees|knn|logreg   （默认 svm）
#     -Backend    features|cnn                            （默认 features）
#     -SkipDownload                                       跳过数据集下载
# =====================================================================

param(
    [string]$Classifier = "svm",
    [string]$Backend = "features",
    [string]$Arch = "mobilenet_v3_small",
    [string]$InputDir = "",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "未找到虚拟环境，请先运行：powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "=== Flavia 树叶识别：一键运行 ===" -ForegroundColor Cyan

if (-not $SkipDownload) {
    Write-Host "`n[1/4] 下载数据集 ..." -ForegroundColor Yellow
    & $PythonExe -m leaves.cli download
} else {
    Write-Host "`n[1/4] 跳过数据集下载" -ForegroundColor Yellow
}

Write-Host "`n[2/4] 训练模型（$Backend / $Classifier）..." -ForegroundColor Yellow
& $PythonExe -m leaves.cli train --backend $Backend --arch $Arch --classifier $Classifier

Write-Host "`n[3/4] 评估模型 ..." -ForegroundColor Yellow
& $PythonExe -m leaves.cli evaluate --split test

Write-Host "`n[4/4] 批量识别 ..." -ForegroundColor Yellow
if ([string]::IsNullOrWhiteSpace($InputDir)) {
    # 没指定目录就挑测试集里的 12 张做演示
    & $PythonExe -m leaves.cli demo --backend $Backend --arch $Arch --classifier $Classifier
} else {
    & $PythonExe -m leaves.cli predict --input $InputDir
}

Write-Host "`n全部完成！结果在 outputs\ 目录下。" -ForegroundColor Green
