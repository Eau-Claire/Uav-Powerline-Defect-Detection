$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $RepoRoot ".venv-roboflow\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv-roboflow. Create the environment before launching the reviewer."
}

Set-Location $RepoRoot
& $Python -m streamlit run "tools\roboflow_relabel\app.py"

