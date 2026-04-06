$ErrorActionPreference = "Stop"

. "$PSScriptRoot\bootstrap.ps1"

$projectRoot = Get-ProjectRoot
Set-PretrainedEnvironment -ProjectRoot $projectRoot
Sync-ModuleEnvironment -ProjectRoot $projectRoot -RelativeModulePath "src\constrained_catboost"
$python = Get-VenvPython -ProjectRoot $projectRoot -RelativeModulePath "src\constrained_catboost"

& $python -m pretrained.catboost_runner @args
