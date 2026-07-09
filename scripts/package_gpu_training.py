#!/usr/bin/env python3
"""Đóng gói toàn bộ code + data để train trên máy có GPU."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.convert_benh_an import convert, DEFAULT_ANNOTATIONS, DEFAULT_SAMPLES, NER_OUT, REL_OUT

README_GPU = """# Viettel AI — Train trên máy GPU

Gói này gồm data + script train NER và Relation (PhoBERT).

## Yêu cầu

- Python 3.10+
- NVIDIA GPU + CUDA (khuyến nghị ≥ 8GB VRAM)
- Internet lần đầu (tải `vinai/phobert-base` từ HuggingFace)

## Cài đặt (1 lần)

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements-train.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Train

### Cách 1 — Train cả hai (khuyến nghị)
```powershell
# Windows
.\\train_all.ps1

# Linux
chmod +x train_all.sh && ./train_all.sh
```

### Cách 2 — Train từng phần
```powershell
python training/train_benh_an_ner.py --epochs 15
python training/train_relation.py --epochs 10
```

Thời gian ước tính (GPU RTX 3060+): NER ~15–30 phút, Relation ~5–15 phút.

## Đóng gói model mang về máy chính

```powershell
python scripts/package_trained_models.py
# → trained_models.zip
```

Copy `trained_models.zip` về máy Windows chính, giải nén vào thư mục project:

```
models/benh_an_ner/final/
models/benh_an_relation/final/
```

## Bật model trên máy chính

Sửa `config/models.yaml`:
```yaml
ner:
  use_transformer: true
  model_path: ./models/benh_an_ner/final

relation_extraction:
  use_transformer: true
  model_path: ./models/benh_an_relation/final
```

Chạy lại export:
```powershell
python scripts/export_competition_output.py -i path/to/input -o output_v3.zip
```

## Cấu trúc output

| Model | Thư mục |
|-------|---------|
| NER | `models/benh_an_ner/final/` |
| Relation | `models/benh_an_relation/final/` |
"""


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def package(out_zip: Path) -> None:
    print("Chuyển đổi annotations → CoNLL + JSONL...")
    convert(DEFAULT_ANNOTATIONS, DEFAULT_SAMPLES, NER_OUT, REL_OUT)

    staging = ROOT / "data" / "training" / "_gpu_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Data
    for src in [NER_OUT, REL_OUT]:
        _copy_tree(src, staging / src.relative_to(ROOT))

    # Training code
    train_files = [
        "training/train_benh_an_ner.py",
        "training/train_relation.py",
        "training/convert_benh_an.py",
        "training/config_benh_an_ner.yaml",
        "training/config_relation.yaml",
        "training/label_mapping.yaml",
    ]
    for rel in train_files:
        _copy_tree(ROOT / rel, staging / rel)

    utils_dir = ROOT / "training" / "utils"
    _copy_tree(utils_dir, staging / "training" / "utils")

    # Scripts
    for rel in [
        "scripts/package_trained_models.py",
        "requirements-train.txt",
        "setup_gpu.ps1",
        "setup_gpu.sh",
    ]:
        _copy_tree(ROOT / rel, staging / rel)

    # Reference annotations (optional regenerate)
    _copy_tree(DEFAULT_ANNOTATIONS, staging / DEFAULT_ANNOTATIONS.relative_to(ROOT))
    samples_dst = staging / "data" / "samples" / "benh_an"
    if DEFAULT_SAMPLES.exists():
        _copy_tree(DEFAULT_SAMPLES, samples_dst)

    # README + train helpers
    (staging / "README_GPU_TRAIN.md").write_text(README_GPU, encoding="utf-8")

    train_all_sh = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== Train NER ==="
python training/train_benh_an_ner.py --epochs 15
echo "=== Train Relation ==="
python training/train_relation.py --epochs 10
echo "=== Package models ==="
python scripts/package_trained_models.py
echo "Done. Download: trained_models.zip"
"""
    (staging / "train_all.sh").write_text(train_all_sh, encoding="utf-8")

    train_all_ps1 = """# Train NER + Relation + đóng gói model
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "=== Train NER ===" -ForegroundColor Cyan
python training/train_benh_an_ner.py --epochs 15
Write-Host "=== Train Relation ===" -ForegroundColor Cyan
python training/train_relation.py --epochs 10
Write-Host "=== Package models ===" -ForegroundColor Cyan
python scripts/package_trained_models.py
Write-Host "Done. Copy trained_models.zip ve may chinh." -ForegroundColor Green
"""
    (staging / "train_all.ps1").write_text(train_all_ps1, encoding="utf-8")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(staging).as_posix())

    shutil.rmtree(staging)
    size_mb = out_zip.stat().st_size / 1024 / 1024
    print(f"GPU training package: {out_zip} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package GPU training bundle")
    parser.add_argument(
        "--output",
        "-o",
        default=str(ROOT / "data" / "training" / "viettel_gpu_train.zip"),
    )
    args = parser.parse_args()
    package(Path(args.output))


if __name__ == "__main__":
    main()
