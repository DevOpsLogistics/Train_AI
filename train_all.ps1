# Train NER + Relation + đóng gói model
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run-Step($label, $cmd) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Buoc '$label' that bai (exit $LASTEXITCODE)"
    }
}

Write-Host "Kiem tra dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements-train.txt
if ($LASTEXITCODE -ne 0) { throw "Khong cai duoc requirements-train.txt" }

python -c "import yaml, torch, transformers, datasets; print('deps OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Thieu thu vien. Chay: pip install pyyaml transformers datasets seqeval accelerate sentencepiece scikit-learn" -ForegroundColor Red
    throw "Thieu dependency"
}

Run-Step "Train NER" "python training/train_benh_an_ner.py --epochs 15"
Run-Step "Train Relation" "python training/train_relation.py --epochs 10"
Run-Step "Package models" "python scripts/package_trained_models.py"
Write-Host "Done. Copy trained_models.zip ve may chinh." -ForegroundColor Green
