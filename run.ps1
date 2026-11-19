# Starts Rhino Fixer and opens it in your browser.
# First run creates the Python environment (needs Python 3.13: https://www.python.org/downloads/).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Setting up Python environment (first run only)..."
    py -3.13 -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
}
Start-Process "http://localhost:8765"
.venv\Scripts\python.exe server.py
