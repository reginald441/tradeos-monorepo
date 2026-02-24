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


$overridePath = "docker-compose.override.yml"
if (Test-Path $overridePath) {
    $overrideText = Get-Content $overridePath -Raw
    if ($overrideText -match 'nginx\.dev\.conf') {
        $overrideText = $overrideText -replace 'nginx\.dev\.conf', 'nginx.conf'
        Set-Content $overridePath $overrideText
        Write-Host "Patched legacy nginx override entries (nginx.dev.conf -> nginx.conf)" -ForegroundColor Yellow
    }
}

$legacyMount = Select-String -Path $overridePath -Pattern "nginx.dev.conf" -ErrorAction SilentlyContinue
if ($legacyMount) {
    Write-Error "docker-compose.override.yml still references nginx.dev.conf. Open the file and replace nginx.dev.conf with nginx.conf, then re-run the script."
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

Write-Host "[5/6] Waiting for backend readiness (up to 60s)..." -ForegroundColor Yellow
$healthUrl = "http://localhost:8000/health"
$healthy = $false
for ($i = 1; $i -le 12; $i++) {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            $healthy = $true
            Write-Host "Backend became healthy on attempt $i" -ForegroundColor Green
            break
        }
    } catch {
        Start-Sleep -Seconds 5
    }
}

Write-Host "[6/6] Backend health checks:" -ForegroundColor Yellow
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

if (-not $healthy) {
    Write-Host "Backend did not become healthy in time; showing last 120 backend log lines:" -ForegroundColor Yellow
    docker compose -f docker-compose.yml -f docker-compose.override.yml logs --tail=120 backend
}

Write-Host "\nOpen these URLs in your browser:" -ForegroundColor Cyan
Write-Host "- Frontend: http://localhost:3000"
Write-Host "- API docs: http://localhost:8000/docs"
Write-Host "- Grafana: http://localhost:3001"
