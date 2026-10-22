param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
if (-not $Python) {
    $localVenv = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $localVenv) { $Python = $localVenv }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $Python = 'python' }
    elseif (Test-Path -LiteralPath $bundledPython) { $Python = $bundledPython }
    else { throw '未找到 Python。请通过 -Python 指定 Python 3.12 可执行文件。' }
}
$workspaceRoot = Split-Path (Split-Path $projectRoot -Parent) -Parent
$runtimePackages = Join-Path $workspaceRoot '.runtime_packages'
if (Test-Path -LiteralPath $runtimePackages) { $env:PYTHONPATH = $runtimePackages }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $workspaceRoot '.matplotlib_cache'
& $Python (Join-Path $projectRoot 'Code\33_refresh_supplementary_experiments.py') --full
if ($LASTEXITCODE -ne 0) { throw "实验流水线失败，退出码 $LASTEXITCODE。" }
