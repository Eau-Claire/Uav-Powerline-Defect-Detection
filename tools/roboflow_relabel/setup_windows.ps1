$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Python = Get-Command py -ErrorAction SilentlyContinue
if ($Python) {
    & py -3.11 -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'"
    & py -3.11 -m venv .venv-roboflow
} else {
    & python -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'"
    & python -m venv .venv-roboflow
}
$VenvPython = Join-Path $Root ".venv-roboflow\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r tools\roboflow_relabel\requirements-handoff.txt
if (-not (Test-Path .env)) {
    Copy-Item tools\roboflow_relabel\.env.example .env
    Write-Host "Created .env. Insert a newly issued Roboflow key before running."
}
& $VenvPython tools\roboflow_relabel\doctor.py
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Setup completed, but preflight still has FAIL items. Fix them before running."
}
