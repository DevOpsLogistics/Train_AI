#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== 1. Tao virtualenv ==="
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

echo "=== 2. Cai PyTorch + CUDA ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
  || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== 3. Cai thu vien train ==="
pip install -r requirements-train.txt

echo "=== 4. Kiem tra GPU ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo "Xong! Chay: ./train_all.sh"
