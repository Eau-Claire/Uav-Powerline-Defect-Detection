param(
    [string]$Device = "0",
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$BenchmarkRoot = Split-Path -Parent $PSScriptRoot
$Evaluator = Join-Path $PSScriptRoot "evaluate_nemo_selected_checkpoints.py"
$ValOut = Join-Path $BenchmarkRoot "results\finding_nemo\selected_checkpoints_eval_val"
$RawOut = Join-Path $BenchmarkRoot "results\finding_nemo\selected_checkpoints_eval_test_main_v1"

Write-Host "Checking Python environment..."
python -c "import torch, ultralytics, transformers, pandas, yaml, PIL; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) {
    throw "Missing Python dependencies. Activate the environment used for the earlier evaluation first."
}

Write-Host ""
Write-Host "[1/2] Resume Val evaluation. Existing YOLO26n/26m/11n results will be reused."
python $Evaluator `
    --dataset finding_nemo `
    --split val `
    --device $Device `
    --batch-size $BatchSize `
    --output-dir $ValOut
if ($LASTEXITCODE -ne 0) {
    throw "Val evaluation failed. Re-run this script to resume from completed models."
}

Write-Host ""
Write-Host "[2/2] Evaluate all six selected checkpoints on raw frozen test_main_v1."
python $Evaluator `
    --dataset test_main_v1 `
    --split test `
    --device $Device `
    --batch-size $BatchSize `
    --output-dir $RawOut
if ($LASTEXITCODE -ne 0) {
    throw "test_main_v1 evaluation failed. Re-run this script to resume from completed models."
}

Write-Host ""
Write-Host "Done."
Write-Host "Val summary: $ValOut\benchmark_summary.csv"
Write-Host "Raw summary: $RawOut\benchmark_summary.csv"
Write-Host "Raw per-class: $RawOut\per_class_summary.csv"
