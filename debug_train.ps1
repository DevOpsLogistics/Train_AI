# Chay 1 buoc train de xem loi chi tiet
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
.venv\Scripts\Activate.ps1

Write-Host "CUDA check:" -ForegroundColor Cyan
python -c "import torch; print('CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "`nTrain NER (1 epoch test)..." -ForegroundColor Cyan
python training/train_benh_an_ner.py --epochs 1

Write-Host "`nExit code:" $LASTEXITCODE
if (Test-Path models/benh_an_ner/final) { Get-ChildItem models/benh_an_ner/final | Select-Object Name }
