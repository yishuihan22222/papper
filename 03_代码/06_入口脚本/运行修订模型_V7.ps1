param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$taskPython = 'C:\Users\11828\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$taskRoot = 'E:\sci23\数据及代码-TJJM20260523014641'
$taskResults = Join-Path $taskRoot '项目结果\processing_unit_rectification_v7_20261021'
$taskDesign = Get-Content -LiteralPath (Join-Path $taskResults '00_design.json') -Raw -Encoding utf8 | ConvertFrom-Json
$taskLock = Get-Content -LiteralPath (Join-Path $taskResults 'design_lock.json') -Raw -Encoding utf8 | ConvertFrom-Json
$taskHash = (Get-FileHash -LiteralPath (Join-Path $taskResults '00_design.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskHash -ne $taskLock.design_sha256) { throw 'Frozen V7 design changed.' }
Write-Output "V7 candidate nodes: $($taskDesign.candidate_nodes), edges: $($taskDesign.candidate_edges)"
Write-Output "Inputs: $(Join-Path $taskResults 'model_inputs')"
Write-Output 'These are processing-site candidates with proxy capacities; not all independently certified.'
if ($CheckOnly) { return }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'E:\sci23\.runtime_packages'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $taskPython (Join-Path $taskRoot '数据及代码\Code\137_run_processing_unit_rectification_v7.py')
if ($LASTEXITCODE -ne 0) { throw 'V7 simulation verification failed.' }
