param(
    [string]$PythonExecutable = 'C:\Users\11828\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path $workspaceRoot '.runtime_packages'
$env:MPLCONFIGDIR = Join-Path $workspaceRoot '.matplotlib_cache'
$codePath = Join-Path $projectRoot '数据及代码\Code'
$backupPath = Join-Path $projectRoot '历史存档\pass_data\V4机制修正前快照_20261016_154642'
& $PythonExecutable (Join-Path $codePath '56_validate_and_scan_v4.py') --backup $backupPath
if ($LASTEXITCODE -ne 0) { throw 'V4 calculation or validation failed' }
& $PythonExecutable (Join-Path $codePath '57_plot_and_report_v4.py')
if ($LASTEXITCODE -ne 0) { throw 'V4 figure/report failed' }
Write-Output 'V4 complete. Inspect project results and the mechanism report; no final threshold has been calibrated.'
