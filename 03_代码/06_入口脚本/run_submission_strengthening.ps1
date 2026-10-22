param([string]$Python = '')
$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
if (-not $Python) {
    $Python=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {throw '请用 -Python 指定具备 pandas/numpy/scipy/matplotlib 的 Python 3.12。'}
}
$workspaceRoot=Split-Path (Split-Path $projectRoot -Parent) -Parent
$env:PYTHONPATH=Join-Path $workspaceRoot '.runtime_packages'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLCONFIGDIR=Join-Path $workspaceRoot '.matplotlib_cache'
& $Python (Join-Path $projectRoot 'Code\45_run_submission_strengthening.py') --workers 4
if ($LASTEXITCODE -ne 0) {throw "补强流程失败，退出码 $LASTEXITCODE。"}
