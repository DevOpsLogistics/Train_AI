# Train NER + Relation + đóng gói model
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run-PythonStep {
    param(
        [string]$Label,
        [string[]]$Args
    )
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    & python @Args
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "Buoc '$Label' that bai (exit $code). Xem loi o tren."
    }
}

Write-Host "Kiem tra dependencies..." -ForegroundColor Yellow
& python -m pip install -r requirements-train.txt
if ($LASTEXITCODE -ne 0) { throw "Khong cai duoc requirements-train.txt" }

& python -c "import yaml, torch, transformers, datasets; print('deps OK'); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Thieu dependency. Chay: pip install pyyaml transformers datasets seqeval accelerate sentencepiece scikit-learn"
}

Run-PythonStep "Train NER" @("training/train_benh_an_ner.py", "--epochs", "15")
if (-not (Test-Path "models/benh_an_ner/final/config.json")) {
    throw "NER train xong nhung khong thay models/benh_an_ner/final/"
}

Run-PythonStep "Train Relation" @("training/train_relation.py", "--epochs", "10")
if (-not (Test-Path "models/benh_an_relation/final/config.json")) {
    throw "Relation train xong nhung khong thay models/benh_an_relation/final/"
}

Run-PythonStep "Package models" @("scripts/package_trained_models.py")
if (-not (Test-Path "trained_models.zip")) {
    throw "Khong tao duoc trained_models.zip"
}

Write-Host "Done. Copy trained_models.zip ve may chinh." -ForegroundColor Green
