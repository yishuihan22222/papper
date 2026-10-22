param(
    [string]$PythonExecutable = 'C:\Users\11828\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference='Stop'
$taskWorkspaceRoot=Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=Join-Path $taskWorkspaceRoot '.runtime_packages'
$taskCodePath=Join-Path $PSScriptRoot '数据及代码\Code'
& $PythonExecutable (Join-Path $taskCodePath '67_validate_recovery_budget_v5.py') --phase toy
if ($LASTEXITCODE -ne 0) { throw 'V5 recovery diagnostic toy verification failed' }
foreach ($taskPhase in @('diagnose','grid','pairs')) {
    & $PythonExecutable (Join-Path $taskCodePath '66_run_recovery_budget_v5.py') --phase $taskPhase
    if ($LASTEXITCODE -ne 0) { throw "V5 recovery phase failed: $taskPhase" }
}
& $PythonExecutable (Join-Path $taskCodePath '67_validate_recovery_budget_v5.py') --phase outputs
if ($LASTEXITCODE -ne 0) { throw 'V5 resource output verification failed' }
& $PythonExecutable (Join-Path $taskCodePath '69_full_pair_stop_diagnosis_v5.py')
if ($LASTEXITCODE -ne 0) { throw 'V5 supplementary pair diagnosis failed' }
& $PythonExecutable (Join-Path $taskCodePath '67_validate_recovery_budget_v5.py') --phase supplement
if ($LASTEXITCODE -ne 0) { throw 'V5 supplementary output verification failed' }
& $PythonExecutable (Join-Path $taskCodePath '68_report_recovery_budget_v5.py')
if ($LASTEXITCODE -ne 0) { throw 'V5 recovery report/final inventory failed' }
Write-Output 'Resource sensitivity and stopping diagnosis complete. Read the report before interpreting policy differences.'
