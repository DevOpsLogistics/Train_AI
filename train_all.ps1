# Train NER + Relation + đóng gói model
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "=== Train NER ===" -ForegroundColor Cyan
python training/train_benh_an_ner.py --epochs 15
Write-Host "=== Train Relation ===" -ForegroundColor Cyan
python training/train_relation.py --epochs 10
Write-Host "=== Package models ===" -ForegroundColor Cyan
python scripts/package_trained_models.py
Write-Host "Done. Copy trained_models.zip ve may chinh." -ForegroundColor Green
