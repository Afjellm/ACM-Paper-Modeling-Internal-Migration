$ErrorActionPreference = "Stop"

. "$PSScriptRoot\bootstrap.ps1"

$projectRoot = Get-ProjectRoot
Set-PretrainedEnvironment -ProjectRoot $projectRoot
Sync-ModuleEnvironment -ProjectRoot $projectRoot -RelativeModulePath "src\gravity_model"
$python = Get-VenvPython -ProjectRoot $projectRoot -RelativeModulePath "src\gravity_model"

& $python -m pretrained.gravity_runner @args
