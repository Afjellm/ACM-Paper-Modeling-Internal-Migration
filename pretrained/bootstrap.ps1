$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Set-PretrainedEnvironment {
    param(
        [string]$ProjectRoot
    )

    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
    $env:PYTHONPATH = if ($env:PYTHONPATH) {
        "$ProjectRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
    }
    else {
        $ProjectRoot
    }
}

function Sync-ModuleEnvironment {
    param(
        [string]$ProjectRoot,
        [string]$RelativeModulePath
    )

    $moduleDir = Join-Path $ProjectRoot $RelativeModulePath
    Write-Host "[INFO] Setting up environment for $moduleDir"
    Push-Location $moduleDir
    try {
        uv sync
    }
    finally {
        Pop-Location
    }
}

function Get-VenvPython {
    param(
        [string]$ProjectRoot,
        [string]$RelativeModulePath
    )

    $pythonPath = Join-Path $ProjectRoot "$RelativeModulePath\.venv\Scripts\python.exe"
    if (-not (Test-Path $pythonPath)) {
        throw "Virtual environment Python not found at $pythonPath"
    }
    return $pythonPath
}
