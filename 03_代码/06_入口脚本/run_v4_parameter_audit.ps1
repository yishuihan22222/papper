param(
    [string]$PythonExecutable = 'C:\Users\11828\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path $workspaceRoot '.runtime_packages'
$env:MPLCONFIGDIR = Join-Path $workspaceRoot '.matplotlib_cache'
$codePath = Join-Path $PSScriptRoot '数据及代码\Code'
$backupPath = Join-Path $PSScriptRoot '历史存档\pass_data\V4参数检查前快照_20261016_162728'
& $PythonExecutable (Join-Path $codePath '58_run_v4_parameter_audit.py') --backup $backupPath
if ($LASTEXITCODE -ne 0) { throw 'Parameter audit calculation failed' }
& $PythonExecutable (Join-Path $codePath '59_summarize_v4_parameter_audit.py')
if ($LASTEXITCODE -ne 0) { throw 'Independent summary/validation failed' }
& $PythonExecutable (Join-Path $codePath '60_report_and_plot_v4_parameter_audit.py')
if ($LASTEXITCODE -ne 0) { throw 'Report/figure generation failed' }
Write-Output 'Parameter audit complete. Final parameter values have not been selected or calibrated.'
