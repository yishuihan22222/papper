param([string]$PythonExe = '', [switch]$Smoke)
$ErrorActionPreference = 'Stop'
$codeDir = Join-Path $PSScriptRoot '数据及代码\Code'
$workspaceDir = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $PythonExe = $bundledPython }
    else { $PythonExe = (Get-Command python -ErrorAction Stop).Source }
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$packageDir = Join-Path $workspaceDir '.runtime_packages'
if (Test-Path -LiteralPath $packageDir) {
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$packageDir;$env:PYTHONPATH" } else { $packageDir }
}
& $PythonExe (Join-Path $codeDir '52_validate_closed_loop_recovery_v3.py')
if ($LASTEXITCODE -ne 0) { throw 'V3独立核验失败。' }
$runnerArgs = @()
if ($Smoke) { $runnerArgs += '--smoke' }
& $PythonExe (Join-Path $codeDir '51_run_closed_loop_recovery_v3.py') @runnerArgs
if ($LASTEXITCODE -ne 0) { throw 'V3闭环实验失败。' }
& $PythonExe (Join-Path $codeDir '53_write_closed_loop_v3_report.py') @runnerArgs
if ($LASTEXITCODE -ne 0) { throw 'V3报告汇总失败。' }
