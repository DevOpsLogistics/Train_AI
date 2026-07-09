# Cài môi trường train GPU trên Windows (PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== 1. Tao virtualenv ===" -ForegroundColor Cyan
if (Test-Path .venv) {
    Write-Host "Xoa .venv cu (neu loi truoc do)..."
    Remove-Item -Recurse -Force .venv
}

# --without-pip neu ensurepip loi tren Windows
python -m venv .venv --without-pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "venv that bai, thu lai khong co --without-pip..."
    python -m venv .venv
}

Write-Host "=== 2. Kich hoat venv ===" -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# Cai pip neu venv tao khong co pip
python -c "import pip" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Cai pip thu cong..."
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
    python get-pip.py
    Remove-Item get-pip.py -ErrorAction SilentlyContinue
}

Write-Host "=== 3. Cai PyTorch + CUDA ===" -ForegroundColor Cyan
python -m pip install --upgrade pip
# CUDA 12.4 — phu hop driver NVIDIA moi
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "cu124 that bai, thu cu121..." -ForegroundColor Yellow
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

Write-Host "=== 4. Cai thu vien train ===" -ForegroundColor Cyan
python -m pip install -r requirements-train.txt

Write-Host "=== 5. Kiem tra GPU ===" -ForegroundColor Cyan
python -c @"
import torch
ok = torch.cuda.is_available()
print('CUDA available:', ok)
if ok:
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('LOI: PyTorch chua co CUDA. Kiem tra driver NVIDIA va cai lai torch cu124.')
"@

Write-Host "`nXong! Chay train: .\train_all.ps1" -ForegroundColor Green
