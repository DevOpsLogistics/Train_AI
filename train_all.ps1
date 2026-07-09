# Train NER + Relation + dong goi model
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "Python: $py" -ForegroundColor Gray
Write-Host "Kiem tra dependencies..." -ForegroundColor Yellow
& $py -m pip install -r requirements-train.txt
if ($LASTEXITCODE -ne 0) { throw "Khong cai duoc requirements-train.txt" }

& $py -c "import yaml, torch, transformers, datasets; print('deps OK'); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Thieu dependency. Chay: pip install pyyaml transformers datasets seqeval accelerate sentencepiece scikit-learn"
}

Write-Host "=== Train NER ===" -ForegroundColor Cyan
& $py training/train_benh_an_ner.py --epochs 15
if ($LASTEXITCODE -ne 0) { throw "Train NER that bai (exit $LASTEXITCODE)" }
if (-not (Test-Path "models/benh_an_ner/final/config.json")) {
    throw "Khong thay models/benh_an_ner/final/"
}

Write-Host "=== Train Relation ===" -ForegroundColor Cyan
& $py training/train_relation.py --epochs 10
if ($LASTEXITCODE -ne 0) { throw "Train Relation that bai (exit $LASTEXITCODE)" }
if (-not (Test-Path "models/benh_an_relation/final/config.json")) {
    throw "Khong thay models/benh_an_relation/final/"
}

Write-Host "=== Package models ===" -ForegroundColor Cyan
& $py scripts/package_trained_models.py
if ($LASTEXITCODE -ne 0) { throw "Package that bai (exit $LASTEXITCODE)" }
if (-not (Test-Path "trained_models.zip")) {
    throw "Khong tao duoc trained_models.zip"
}

Write-Host "Done. Copy trained_models.zip ve may chinh." -ForegroundColor Green
