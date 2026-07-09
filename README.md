# Train AI — Viettel Medical NLP

Repo train **NER + Relation** (PhoBERT) trên 110 bệnh án gán nhãn.

## Quick start (máy GPU)

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\setup_gpu.ps1
.\train_all.ps1
```

Sau train: copy `trained_models.zip` về project [Viettel_AI](https://github.com/DevOpsLogistics/Viettel_AI).

Chi tiết: [README_GPU_TRAIN.md](README_GPU_TRAIN.md)
