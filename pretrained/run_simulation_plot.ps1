$ErrorActionPreference = "Stop"

. "$PSScriptRoot\bootstrap.ps1"

$projectRoot = Get-ProjectRoot
Set-PretrainedEnvironment -ProjectRoot $projectRoot
Sync-ModuleEnvironment -ProjectRoot $projectRoot -RelativeModulePath "src\simulation"
$python = Get-VenvPython -ProjectRoot $projectRoot -RelativeModulePath "src\simulation"

Write-Host "[INFO] Rendering simulation plots"
Push-Location $projectRoot
try {
    & $python -m src.simulation.simulation_plot @args
}
finally {
    Pop-Location
}
