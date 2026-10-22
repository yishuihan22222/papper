$ErrorActionPreference = 'Stop'
$taskPython = 'C:\Users\11828\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$taskScripts = @('pilot.py','structural_checks.py','analytic_rule_check.py','uncertain_capacity.py','uncertainty_verification.py','finalize.py')
foreach ($taskScript in $taskScripts) {
    & $taskPython (Join-Path $PSScriptRoot ('code\' + $taskScript))
    if ($LASTEXITCODE -ne 0) { throw ('Pilot verification failed: ' + $taskScript) }
}
