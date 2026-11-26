# start_demo.ps1 - one command for the defense demo
#   Starts Docker Desktop (if needed), the Docker Compose stack (Airflow, MLflow,
#   PostgreSQL, object storage, email), the MLflow dashboard for the project's
#   models (mlruns/, port 5001), the live-model web app, then opens Airflow,
#   MLflow and http://127.0.0.1:8050 in the browser.
#
#   Run:  double-click start_demo.bat   (or: powershell -ExecutionPolicy Bypass -File start_demo.ps1)
#   Airflow login, if the browser asks: admin / admin (local test login; the session lasts 30 days)
param([switch]$NoBrowser)

$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
Set-Location $root

function Test-Url($url) {
    try { (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { $false }
}
function Wait-Until($what, $seconds, [scriptblock]$ok) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while (-not (& $ok)) {
        if ((Get-Date) -gt $deadline) { Write-Host "   $what did not come up within $seconds s." -ForegroundColor Red; return $false }
        Start-Sleep -Seconds 3
    }
    return $true
}
function Test-Docker { & docker info *> $null; return ($LASTEXITCODE -eq 0) }

Write-Host "1/5  Docker" -ForegroundColor Cyan
if (-not (Test-Docker)) {
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host "   starting Docker Desktop..."
    if (-not (Wait-Until "Docker" 180 { Test-Docker })) { exit 1 }
}
Write-Host "   Docker is running."

Write-Host "2/5  Airflow, MLflow, PostgreSQL, storage and email (docker compose)" -ForegroundColor Cyan
& docker compose up -d
if (-not (Wait-Until "Airflow" 240 { Test-Url "http://localhost:8080/health" })) { exit 1 }
Write-Host "   Airflow is up: http://localhost:8080"

Write-Host "3/5  MLflow dashboard for the project's models (mlruns/)" -ForegroundColor Cyan
if (-not (Test-Url "http://127.0.0.1:5001/")) {
    Start-Process mlflow -ArgumentList "ui", "--backend-store-uri", "./mlruns", "--port", "5001" -WorkingDirectory $root -WindowStyle Minimized
    Write-Host "   starting (about 15 s)..."
    if (-not (Wait-Until "The MLflow dashboard" 90 { Test-Url "http://127.0.0.1:5001/" })) { exit 1 }
}
Write-Host "   MLflow is up: http://localhost:5001"

Write-Host "4/5  Live-model app" -ForegroundColor Cyan
if (-not (Test-Url "http://127.0.0.1:8050/api/state")) {
    Start-Process python -ArgumentList "demo/app.py", "--no-browser" -WorkingDirectory $root -WindowStyle Minimized
    Write-Host "   starting (about 20 s)..."
    if (-not (Wait-Until "The live-model app" 120 { Test-Url "http://127.0.0.1:8050/api/state" })) { exit 1 }
}
Write-Host "   Live model is up: http://127.0.0.1:8050"

Write-Host "5/5  Opening the browser" -ForegroundColor Cyan
if (-not $NoBrowser) {
    Start-Process "http://localhost:8080/dags/energy_trading_mlops_pipeline/grid"
    Start-Sleep -Seconds 1
    Start-Process "http://localhost:5001/#/models/energy_trading_ppo"
    Start-Sleep -Seconds 1
    Start-Process "http://127.0.0.1:8050/"
}
Write-Host ""
Write-Host "Ready." -ForegroundColor Green
Write-Host "  Airflow     http://localhost:8080   (login if asked: admin / admin)"
Write-Host "  MLflow      http://localhost:5001   (models: energy_trading_ppo, champion = v5; experiments on the left)"
Write-Host "  Live model  http://127.0.0.1:8050   (press Reset demo first if it already shows the new days)"
Write-Host "  Email inbox http://localhost:8026"
