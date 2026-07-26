#!/usr/bin/env pwsh
<#
.SYNOPSIS
    One-shot local environment setup for ForecastForge (Windows / PowerShell).

.DESCRIPTION
    Responsibilities (and only these):
      1. Create a virtual environment.
      2. Install the package (editable, with dev dependencies).
      3. Install the pre-commit git hook.
      4. Copy .env.example to .env if .env does not already exist.
      5. Print a success message.

.EXAMPLE
    ./scripts/bootstrap.ps1
#>

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvDir = ".venv"
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

Write-Host "==> Using Python interpreter: $PythonBin"

if (-not (Test-Path $VenvDir)) {
    Write-Host "==> Creating virtual environment in $VenvDir"
    & $PythonBin -m venv $VenvDir
} else {
    Write-Host "==> Virtual environment already exists at $VenvDir, reusing it"
}

$ActivateScript = Join-Path $VenvDir "Scripts/Activate.ps1"
Write-Host "==> Activating virtual environment"
. $ActivateScript

Write-Host "==> Upgrading pip"
python -m pip install --upgrade pip

Write-Host "==> Installing ForecastForge (editable) with dev dependencies"
pip install -e ".[dev,docs]"

Write-Host "==> Installing pre-commit git hook"
pre-commit install

$EnvFile = ".env"
$EnvExample = ".env.example"

if (-not (Test-Path $EnvFile)) {
    Write-Host "==> Copying $EnvExample to $EnvFile"
    Copy-Item $EnvExample $EnvFile
} else {
    Write-Host "==> $EnvFile already exists, leaving it unchanged"
}

Write-Host ""
Write-Host "✅ ForecastForge development environment is ready."
Write-Host "   Activate it in new shells with: $ActivateScript"
