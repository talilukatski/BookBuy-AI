Write-Host "Starting Book Buying Agent Services..."

# --- Auto-Setup Checks ---

# 1. Backend Setup
if (-not (Test-Path ".\.venv")) {
    Write-Host "First time run detected for Backend. Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Syncing Python dependencies..." -ForegroundColor Cyan
# Using --no-deps for problematic packages if necessary, but try standard install first
.\.venv\Scripts\python -m pip install -r requirements.txt --disable-pip-version-check

# Activate the virtual environment in the current script scope
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    . .\.venv\Scripts\Activate.ps1
}

# 2. Frontend Setup
if (-not (Test-Path ".\frontend\node_modules")) {
    Write-Host "First time run detected for Frontend. Installing npm packages..." -ForegroundColor Cyan
    Push-Location frontend
    npm install
    Pop-Location
}

# Start Backend in a new window
Start-Process powershell -ArgumentList '-NoExit', '-Command', '
    $host.UI.RawUI.WindowTitle = ''Backend Retailer Service'';
    Write-Host "Activating venv and starting Backend...";
    .\.venv\Scripts\Activate.ps1;
    python -m uvicorn mock_retailer.main:app --reload --port 8000
'

# Start Frontend in a new window
Start-Process powershell -ArgumentList '-NoExit', '-Command', '
    $host.UI.RawUI.WindowTitle = ''Frontend (React UI)'';
    Write-Host "Starting Frontend...";
    Set-Location frontend;
    npm run dev
'

# Start Agent Server in a new window
Start-Process powershell -ArgumentList '-NoExit', '-Command', '
    $host.UI.RawUI.WindowTitle = ''Agent AI Server'';
    Write-Host "Activating venv and starting Agent Server...";
    .\.venv\Scripts\Activate.ps1;
    python agent_server.py
'

Write-Host "Services launched in separate windows."
Write-Host "NOTE: To keep the virtual environment active in this terminal, you must dot-source the script:" -ForegroundColor Yellow
Write-Host ". .\start_all.ps1" -ForegroundColor Yellow
