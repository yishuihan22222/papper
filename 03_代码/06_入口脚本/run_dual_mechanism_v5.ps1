param(
    [string]$PythonExecutable = 'C:\Users\11828\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference='Stop'
$workspaceRoot=Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=Join-Path $workspaceRoot '.runtime_packages'
$env:MPLCONFIGDIR=Join-Path $workspaceRoot '.matplotlib_cache'
$codePath=Join-Path $PSScriptRoot '数据及代码\Code'
$backupPath=Join-Path $PSScriptRoot '历史存档\pass_data\V5双机制实施前快照_20261016_170237'
& $PythonExecutable (Join-Path $codePath '61_validate_dual_mechanism_v5.py')
if ($LASTEXITCODE -ne 0) { throw 'V5 analytic validation failed' }
& $PythonExecutable (Join-Path $codePath '62_run_dual_mechanism_v5.py') --backup $backupPath --phase all
if ($LASTEXITCODE -ne 0) { throw 'V5 mechanism/recovery experiment failed' }
& $PythonExecutable (Join-Path $codePath '63_validate_and_summarize_v5.py')
if ($LASTEXITCODE -ne 0) { throw 'V5 independent validation failed' }
& $PythonExecutable (Join-Path $codePath '64_plot_and_report_dual_mechanism_v5.py')
if ($LASTEXITCODE -ne 0) { throw 'V5 report/figure failed' }
Write-Output 'V5 calculations complete. Inspect assumptions, model limits and final figure QA before using in a manuscript.'
