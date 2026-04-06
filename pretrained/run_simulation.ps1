$ErrorActionPreference = "Stop"

. "$PSScriptRoot\bootstrap.ps1"

$projectRoot = Get-ProjectRoot
Set-PretrainedEnvironment -ProjectRoot $projectRoot
$env:MODEL_OUTPUT_BASE = "models/output"
$env:ROWS_PER_AGE_GROUP = "NONE"

$requiredModules = @(
    "src\simulation",
    "src\automl",
    "src\constrained_xgboost",
    "src\constrained_catboost"
)

foreach ($module in $requiredModules) {
    Sync-ModuleEnvironment -ProjectRoot $projectRoot -RelativeModulePath $module
}

$python = Get-VenvPython -ProjectRoot $projectRoot -RelativeModulePath "src\simulation"
Write-Host "[INFO] Running simulation with pretrained artifacts from $env:MODEL_OUTPUT_BASE"
Push-Location $projectRoot
try {
    & $python -m src.simulation.simulate_data @args
}
finally {
    Pop-Location
}
