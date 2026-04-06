$ErrorActionPreference = "Stop"

. "$PSScriptRoot\bootstrap.ps1"

$projectRoot = Get-ProjectRoot
Set-PretrainedEnvironment -ProjectRoot $projectRoot
Sync-ModuleEnvironment -ProjectRoot $projectRoot -RelativeModulePath "src\automl"
$python = Get-VenvPython -ProjectRoot $projectRoot -RelativeModulePath "src\automl"

& $python -m pretrained.autogluon_runner @args
