$ErrorActionPreference = "Stop"

Write-Host "== TradeOS Windows Health Check ==" -ForegroundColor Cyan


function Exit-IfFailed($stepDescription) {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$stepDescription failed. Fix the error above and re-run ./scripts/windows-health-check.ps1"
        exit $LASTEXITCODE
    }
}



if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "Run this script from the tradeos folder (where docker-compose.yml exists)."
}

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

$corsLine = Select-String -Path ".env" -Pattern "^CORS_ORIGINS=" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $corsLine) {
    Add-Content ".env" "`nCORS_ORIGINS=[\"http://localhost:3000\",\"http://localhost:8080\",\"http://localhost\"]"
    Write-Host "Added CORS_ORIGINS to .env" -ForegroundColor Yellow
} elseif ($corsLine.Line -notmatch "^CORS_ORIGINS=\[") {
    (Get-Content ".env") -replace '^CORS_ORIGINS=.*$', 'CORS_ORIGINS=["http://localhost:3000","http://localhost:8080","http://localhost"]' | Set-Content ".env"
    Write-Host "Updated CORS_ORIGINS to JSON array format" -ForegroundColor Yellow
}



codex/ensure-docker-compose-runs-cleanly-872v1p

$grafanaPortLine = Select-String -Path ".env" -Pattern "^GRAFANA_PORT=" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $grafanaPortLine) {
    Add-Content ".env" "`nGRAFANA_PORT=3001"
    Write-Host "Added GRAFANA_PORT=3001 to .env to avoid frontend port collisions" -ForegroundColor Yellow
} elseif ($grafanaPortLine.Line -match "^GRAFANA_PORT=3000$") {
    (Get-Content ".env") -replace '^GRAFANA_PORT=3000$', 'GRAFANA_PORT=3001' | Set-Content ".env"
    Write-Host "Updated GRAFANA_PORT from 3000 to 3001 to avoid frontend conflicts" -ForegroundColor Yellow
}

Write-Host "\n[1/6] Validating compose configuration..." -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml config | Out-Null

Exit-IfFailed "Compose config validation"

Write-Host "[2/6] Resetting old containers (down --remove-orphans)..." -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml down --remove-orphans
Exit-IfFailed "Compose down --remove-orphans"

Write-Host "[3/6] Starting services..." -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build
Exit-IfFailed "Compose up -d --build"

Write-Host "[4/6] Service status:" -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml ps
Exit-IfFailed "Compose ps"


Write-Host "[2/6] Resetting old containers (down --remove-orphans)..." -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml down --remove-orphans

Write-Host "[3/6] Starting services..." -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build

Write-Host "[4/6] Service status:" -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml ps


Write-Host "[5/6] Waiting for backend warmup (15s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host "[6/6] Backend health checks:" -ForegroundColor Yellow



Write-Host "\n[1/5] Validating compose configuration..." -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml config | Out-Null

Write-Host "[2/5] Starting services..." -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build

Write-Host "[3/5] Service status:" -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.override.yml ps

Write-Host "[4/5] Waiting for backend warmup (15s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host "[5/5] Backend health checks:" -ForegroundColor Yellow
main

$urls = @(
    "http://localhost:8000/health",
    "http://localhost:8000/ready",
    "http://localhost:8000/live"
)

foreach ($url in $urls) {
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 20
        Write-Host "PASS $url -> $($response.StatusCode)" -ForegroundColor Green
    } catch {
        Write-Host "FAIL $url -> $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "\nOpen these URLs in your browser:" -ForegroundColor Cyan
Write-Host "- Frontend: http://localhost:3000"
Write-Host "- API docs: http://localhost:8000/docs"

Write-Host "- Grafana: http://localhost:3001"

codex/ensure-docker-compose-runs-cleanly-872v1p
Write-Host "- Grafana: http://localhost:3001"
main

