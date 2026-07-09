#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pip install -r requirements-train.txt
python -c "import yaml, torch, transformers, datasets; print('deps OK')"

echo "=== Train NER ==="
python training/train_benh_an_ner.py --epochs 15
echo "=== Train Relation ==="
python training/train_relation.py --epochs 10
echo "=== Package models ==="
python scripts/package_trained_models.py
echo "Done. Download: trained_models.zip"
