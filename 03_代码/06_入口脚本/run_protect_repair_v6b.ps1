param(
    [string]$PythonExe = 'C:\Users\11828\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$codeDirectory = Join-Path $projectRoot '数据及代码\Code'
$originalManifest = Join-Path $projectRoot '项目结果\protect_repair_v6_20261017\72_final_manifest.json'
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python unavailable: $PythonExe" }
if (-not (Test-Path -LiteralPath $originalManifest -PathType Leaf)) { throw 'Complete and preserve the original V6 pilot first.' }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path $workspaceRoot '.runtime_packages'
& $PythonExe (Join-Path $codeDirectory '75_training_aware_protection_v6b.py') --resume
if ($LASTEXITCODE -ne 0) { throw 'V6b main experiment failed; inspect output before continuing.' }
& $PythonExe (Join-Path $codeDirectory '77_validate_training_aware_v6b.py')
if ($LASTEXITCODE -ne 0) { throw 'V6b independent validation failed.' }
& $PythonExe (Join-Path $codeDirectory '78_audit_large_protection_responses_v6b.py')
if ($LASTEXITCODE -ne 0) { throw 'V6b large-response cold replay failed.' }
& $PythonExe (Join-Path $codeDirectory '76_report_training_aware_v6b.py')
if ($LASTEXITCODE -ne 0) { throw 'V6b report or preservation check failed.' }
