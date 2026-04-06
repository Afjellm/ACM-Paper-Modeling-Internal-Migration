$ErrorActionPreference = "Stop"

. "$PSScriptRoot\bootstrap.ps1"

$projectRoot = Get-ProjectRoot
Set-PretrainedEnvironment -ProjectRoot $projectRoot
Sync-ModuleEnvironment -ProjectRoot $projectRoot -RelativeModulePath "src\constrained_xgboost"
$python = Get-VenvPython -ProjectRoot $projectRoot -RelativeModulePath "src\constrained_xgboost"

& $python -m pretrained.xgboost_runner @args
