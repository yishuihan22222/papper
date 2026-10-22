param(
    [string]$PythonExe = 'C:\Users\11828\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$codeDirectory = Join-Path $projectRoot '数据及代码\Code'
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python unavailable: $PythonExe" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path $workspaceRoot '.runtime_packages'
& $PythonExe (Join-Path $codeDirectory '84_run_capacity_policy_mainline.py') --resume
if ($LASTEXITCODE -ne 0) { throw 'Capacity study failed.' }
& $PythonExe (Join-Path $codeDirectory '85c_generate_constrained_nulls_fixed_steps.py') --resume
if ($LASTEXITCODE -ne 0) { throw 'Constrained-null generation failed.' }
& $PythonExe (Join-Path $codeDirectory '86c_run_null_policy_fixed_steps.py') --resume
if ($LASTEXITCODE -ne 0) { throw 'Null-policy study failed.' }
& $PythonExe (Join-Path $codeDirectory '84b_no_damage_controls.py')
if ($LASTEXITCODE -ne 0) { throw 'No-damage controls failed.' }
& $PythonExe (Join-Path $codeDirectory '87_validate_mainline.py')
if ($LASTEXITCODE -ne 0) { throw 'Independent validation failed.' }
& $PythonExe (Join-Path $codeDirectory '88_summarize_and_plot_mainline.py')
if ($LASTEXITCODE -ne 0) { throw 'Figure or summary generation failed.' }
& $PythonExe (Join-Path $codeDirectory '89_write_mainline_report.py')
if ($LASTEXITCODE -ne 0) { throw 'Report finalization failed; inspect validation or figure-review diagnostics.' }
