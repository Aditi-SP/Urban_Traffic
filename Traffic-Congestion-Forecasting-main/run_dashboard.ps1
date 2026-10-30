<#
.SYNOPSIS
  Starts the Traffic Congestion dashboard using the project's existing virtual environment.
  No manual activation is needed.

.EXAMPLE
  .\run_dashboard.ps1                 # quick preflight, then open the dashboard in your browser
  .\run_dashboard.ps1 -FullCheck      # full preflight (loads the models, makes a real sample prediction), then launch
  .\run_dashboard.ps1 -CheckOnly      # run the preflight and exit (does not start the dashboard)
  .\run_dashboard.ps1 -Port 8600 -NoBrowser
  .\run_dashboard.ps1 -SkipPreflight  # launch immediately

.NOTES
  The virtual environment is looked for in this folder (.venv) and in the parent folder (..\.venv).
  The preflight separates REQUIRED components (blocks the launch if broken) from OPTIONAL ones (Java/PySpark, TomTom key),
  which never stop the dashboard from starting.
#>
param(
    [int]$Port = 8501,
    [switch]$NoBrowser,
    [switch]$SkipPreflight,
    [switch]$FullCheck,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot

# --- find the project's virtual environment (no activation needed: we call its python.exe directly) ---
$candidates = @(
    (Join-Path $project '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path $project -Parent) '.venv\Scripts\python.exe')
)
$python = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    Write-Host ''
    Write-Host 'ERROR: the project virtual environment was not found. Looked for:' -ForegroundColor Red
    $candidates | ForEach-Object { Write-Host "  $_" }
    Write-Host ''
    Write-Host 'Create it once (Python 3.11 recommended), then install the dependencies:' -ForegroundColor Yellow
    Write-Host '  py -3.11 -m venv ..\.venv'
    Write-Host '  ..\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    exit 1
}

Set-Location $project
$env:TF_CPP_MIN_LOG_LEVEL = '2'   # hides TensorFlow INFO logs only; numerical results are unchanged
$env:PYTHONUTF8 = '1'

# --- settings saved for your Windows user ---
# Windows copies environment variables into a process when it STARTS. A terminal that was opened before a variable was saved
# (for example a terminal inside VS Code, or another PowerShell window) does not have it, so the dashboard would start without your
# TomTom key or without LIVE_CACHE_TTL_S. This loads the SAVED user/machine value into this launcher (and so the dashboard) whenever the
# current terminal has none. A value already present in this terminal always wins. Values are never printed.
function Import-MissingEnvironment {
    param(
        [string[]]$Names,
        [scriptblock]$Getter = { param($n, $s) [Environment]::GetEnvironmentVariable($n, $s) }
    )
    foreach ($n in $Names) {
        if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($n, 'Process'))) { continue }
        foreach ($scope in 'User', 'Machine') {
            $v = & $Getter $n $scope
            if (-not [string]::IsNullOrWhiteSpace($v)) {
                [Environment]::SetEnvironmentVariable($n, $v, 'Process')
                [pscustomobject]@{ Name = $n; Scope = $scope }
                break
            }
        }
    }
}
foreach ($l in @(Import-MissingEnvironment -Names 'TOMTOM_API_KEY', 'LIVE_CACHE_TTL_S', 'LIVE_MAX_REQUESTS')) {
    if ($l.Name -eq 'TOMTOM_API_KEY') {
        Write-Host "TomTom key: loaded from your Windows $($l.Scope.ToLower()) environment (this terminal did not have it; value not shown)." -ForegroundColor Cyan
    } else {
        Write-Host "$($l.Name) = $([Environment]::GetEnvironmentVariable($l.Name, 'Process')) (loaded from your Windows $($l.Scope.ToLower()) environment)." -ForegroundColor Cyan
    }
}
if (-not [string]::IsNullOrWhiteSpace($env:TOMTOM_API_KEY)) {
    Write-Host 'TomTom key: found in the environment (value not shown).' -ForegroundColor Green
} elseif (Test-Path (Join-Path $project '.streamlit\secrets.toml')) {
    Write-Host 'TomTom key: no environment variable; using .streamlit\secrets.toml if it is valid (checked when the dashboard starts).' -ForegroundColor Green
} else {
    Write-Host 'TomTom key: NOT found. The map and place search work, but there will be no live traffic. See README: Setting up TOMTOM_API_KEY.' -ForegroundColor Yellow
}

# --- preflight ---
if (-not $SkipPreflight -or $CheckOnly) {
    Write-Host 'Running preflight check...' -ForegroundColor Cyan
    $pfArgs = @('-m', 'src.preflight')
    if (-not $FullCheck -and -not $CheckOnly) { $pfArgs += '--fast' }
    if (-not $FullCheck -and -not $CheckOnly) { $pfArgs += '--quiet' }
    & $python @pfArgs
    $pfExit = $LASTEXITCODE
    if ($CheckOnly) { exit $pfExit }
    if ($pfExit -ne 0) {
        Write-Host ''
        Write-Host 'Required components are missing or broken (see [FAIL] above), so the dashboard was NOT started.' -ForegroundColor Red
        Write-Host 'Fix those items, or run  .\run_dashboard.ps1 -SkipPreflight  to start anyway (affected features will be disabled).' -ForegroundColor Yellow
        exit $pfExit
    }
}

# --- pick a free port (8501 may be taken by another Streamlit app) ---
function Test-PortFree([int]$p) {
    try {
        $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
        $l.Start(); $l.Stop(); return $true
    } catch { return $false }
}
$chosen = $Port
while (-not (Test-PortFree $chosen) -and $chosen -lt ($Port + 20)) { $chosen++ }
if ($chosen -ne $Port) {
    Write-Host "Port $Port is busy; using $chosen instead." -ForegroundColor Yellow
    Write-Host "Another dashboard may already be running on port $Port (started before your settings changed). Open http://localhost:$chosen for THIS one," -ForegroundColor Yellow
    Write-Host "or stop the old one first (README: Restart the dashboard)." -ForegroundColor Yellow
}

$streamlitArgs = @('-m', 'streamlit', 'run', 'app.py', '--server.port', "$chosen", '--browser.gatherUsageStats', 'false')
if ($NoBrowser) { $streamlitArgs += @('--server.headless', 'true') }

Write-Host ''
Write-Host "Starting the dashboard on http://localhost:$chosen  (press Ctrl+C in this window to stop it)" -ForegroundColor Green
& $python @streamlitArgs
exit $LASTEXITCODE
