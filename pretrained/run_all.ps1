$ErrorActionPreference = "Stop"

. "$PSScriptRoot\bootstrap.ps1"

$projectRoot = Get-ProjectRoot
Set-PretrainedEnvironment -ProjectRoot $projectRoot

$models = @(
    @{ Module = "src\constrained_xgboost"; Runner = "pretrained.xgboost_runner" },
    @{ Module = "src\constrained_catboost"; Runner = "pretrained.catboost_runner" },
    @{ Module = "src\gravity_model"; Runner = "pretrained.gravity_runner" },
    @{ Module = "src\automl"; Runner = "pretrained.autogluon_runner" }
)

foreach ($model in $models) {
    Sync-ModuleEnvironment -ProjectRoot $projectRoot -RelativeModulePath $model.Module
    $python = Get-VenvPython -ProjectRoot $projectRoot -RelativeModulePath $model.Module
    & $python -m $model.Runner @args
}
