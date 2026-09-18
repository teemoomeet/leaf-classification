# =====================================================================
# Flavia 树叶识别 —— 环境一键配置（Windows PowerShell）
#
# 用法（在项目根目录 D:\leaves 下执行）：
#     powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# 脚本会：
#   1. 创建虚拟环境 .venv
#   2. 安装 requirements.txt 中的依赖（使用清华镜像加速）
#   3. 可选安装深度学习依赖 requirements-cnn.txt
#   4. 自检所有关键库能否正常导入
# =====================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Mirror  = "https://pypi.tuna.tsinghua.edu.cn/simple"
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "=== Flavia 树叶识别：环境配置 ===" -ForegroundColor Cyan
Write-Host "项目目录: $ProjectRoot"

# ---------------------------------------------------------------- 1. 找 Python
$BasePython = $null
foreach ($candidate in @("python", "py -3.12", "py -3.11", "py -3.10", "py -3")) {
    try {
        $cmd = Get-Command ($candidate.Split(" ")[0]) -ErrorAction Stop
        & $cmd.Source -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $BasePython = $candidate; break }
    } catch { }
}

if (-not $BasePython) {
    Write-Host "未找到 Python，请先安装 Python 3.9+ ：https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "使用 Python: $BasePython"

# ---------------------------------------------------------------- 2. 建虚拟环境
if (Test-Path $PythonExe) {
    Write-Host "已存在虚拟环境 .venv，跳过创建。"
} else {
    Write-Host "创建虚拟环境 .venv ..."
    Invoke-Expression "$BasePython -m venv `"$VenvDir`""
}

# ---------------------------------------------------------------- 3. 装依赖
Write-Host "升级 pip ..."
& $PythonExe -m pip install --upgrade pip -i $Mirror

Write-Host "安装基础依赖（numpy / scipy / opencv / scikit-learn / pandas / matplotlib ...）..."
& $PythonExe -m pip install --retries 10 --timeout 60 -r (Join-Path $ProjectRoot "requirements.txt") -i $Mirror

$answer = Read-Host "是否安装深度学习依赖 PyTorch（约 200MB，可选，输入 y 安装）"
if ($answer -eq "y" -or $answer -eq "Y") {
    Write-Host "安装 PyTorch（CPU 版）..."
    & $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements-cnn.txt") -i $Mirror
}

# ---------------------------------------------------------------- 4. 自检
Write-Host "`n=== 依赖自检 ===" -ForegroundColor Cyan
& $PythonExe -c @"
mods = ['numpy', 'scipy', 'cv2', 'sklearn', 'pandas', 'matplotlib', 'PIL', 'joblib', 'tqdm']
for m in mods:
    try:
        mod = __import__(m)
        print(f'  [OK]   {m:<12} {getattr(mod, "__version__", "")}')
    except Exception as e:
        print(f'  [FAIL] {m:<12} {e}')
try:
    import torch, torchvision
    print(f'  [OK]   torch        {torch.__version__} (可选，深度学习路线)')
except Exception:
    print('  [--]   torch        未安装（仅影响 --backend cnn，手工特征路线不受影响）')
"@

Write-Host "`n环境配置完成！下一步：" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m leaves.cli download"
Write-Host "  python -m leaves.cli train"
Write-Host "  python -m leaves.cli predict --input D:\some\leaf\photos"
