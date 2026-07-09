# Cài môi trường train GPU trên Windows (PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== 1. Tao virtualenv ===" -ForegroundColor Cyan
if (Test-Path .venv) {
    Write-Host "Xoa .venv cu..."
    Remove-Item -Recurse -Force .venv
}

python -m venv .venv --without-pip
if ($LASTEXITCODE -ne 0) {
    python -m venv .venv
}

Write-Host "=== 2. Kich hoat venv ===" -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

python -c "import pip" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Cai pip thu cong..."
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
    python get-pip.py
    Remove-Item get-pip.py -ErrorAction SilentlyContinue
}

Write-Host "=== 3. Cai PyTorch + CUDA ===" -ForegroundColor Cyan
python -m pip install --upgrade pip

# RTX 50xx (Blackwell sm_120) CAN cu128 — cu124 se crash (exit -1073741819)
Write-Host "Thu cu128 truoc (RTX 5060/5070 Blackwell)..." -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    Write-Host "cu128 that bai, thu cu124 (GPU cu hon)..." -ForegroundColor Yellow
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) {
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
}

Write-Host "=== 4. Cai thu vien train ===" -ForegroundColor Cyan
python -m pip install -r requirements-train.txt

Write-Host "=== 5. Kiem tra GPU ===" -ForegroundColor Cyan
python -c @"
import torch
print('torch', torch.__version__)
print('cuda runtime', torch.version.cuda)
ok = torch.cuda.is_available()
print('CUDA available:', ok)
if ok:
    print('GPU:', torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print('Compute capability:', cap)
    x = torch.zeros(1, device='cuda')
    print('CUDA tensor test: OK')
else:
    print('LOI: CUDA khong hoat dong')
"@

Write-Host "`nXong! Chay: .\debug_train.ps1 hoac .\train_all.ps1" -ForegroundColor Green
