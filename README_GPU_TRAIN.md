# Viettel AI — Train trên máy GPU

Gói này gồm data + script train NER và Relation (PhoBERT).

## Yêu cầu

- Python 3.10+
- NVIDIA GPU + CUDA (khuyến nghị ≥ 8GB VRAM)
- Internet lần đầu (tải `vinai/phobert-base` từ HuggingFace)

## Cài đặt (1 lần)

### Windows (PowerShell) — khuyến nghị

```powershell
# Cho phep chay script (1 lan)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

git clone https://github.com/DevOpsLogistics/Train_AI.git
cd Train_AI

# Script tu dong: venv + pip + PyTorch CUDA + dependencies
.\setup_gpu.ps1
```

**Nếu `python -m venv` lỗi ensurepip**, chạy thủ công:

```powershell
python -m venv .venv --without-pip
.venv\Scripts\Activate.ps1
Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
python get-pip.py

# QUAN TRONG: cai torch ban CUDA (khong dung pip install torch thuong)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Lỗi `Torch not compiled with CUDA enabled`:** bạn đã cài nhầm bản CPU. Gỡ và cài lại:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Linux

```bash
chmod +x setup_gpu.sh train_all.sh
./setup_gpu.sh
```

## Train

### Cách 1 — Train cả hai (khuyến nghị)
```powershell
# Windows
.\train_all.ps1

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
