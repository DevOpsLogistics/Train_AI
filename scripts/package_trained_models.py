#!/usr/bin/env python3
"""Đóng gói model đã train để copy về máy chính."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NER_FINAL = ROOT / "models" / "benh_an_ner" / "final"
REL_FINAL = ROOT / "models" / "benh_an_relation" / "final"


def _read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def package(out_zip: Path) -> None:
    missing = [p for p in (NER_FINAL, REL_FINAL) if not p.exists()]
    if missing:
        print("Thiếu model — train trước:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)

    staging = ROOT / "data" / "training" / "_models_staging"
    if staging.exists():
        shutil.rmtree(staging)

    ner_dst = staging / "models" / "benh_an_ner" / "final"
    rel_dst = staging / "models" / "benh_an_relation" / "final"
    shutil.copytree(NER_FINAL, ner_dst)
    shutil.copytree(REL_FINAL, rel_dst)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ner_path": str(NER_FINAL.relative_to(ROOT)),
        "relation_path": str(REL_FINAL.relative_to(ROOT)),
        "install_hint": "Giai nen vao thu muc goc Viettel_AI/",
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    install_ps1 = """# Giai nen trained_models.zip vao thu muc goc Viettel_AI (cung cap voi src/)
# Sau do sua config/models.yaml:
#   ner.model_path: ./models/benh_an_ner/final
#   relation_extraction.model_path: ./models/benh_an_relation/final
"""
    (staging / "INSTALL.txt").write_text(install_ps1, encoding="utf-8")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())

    shutil.rmtree(staging)
    size_mb = out_zip.stat().st_size / 1024 / 1024
    print(f"Trained models package: {out_zip} ({size_mb:.1f} MB)")
    print("Copy file nay ve may chinh, giai nen vao thu muc Viettel_AI/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package trained NER + Relation models")
    parser.add_argument(
        "--output",
        "-o",
        default=str(ROOT / "trained_models.zip"),
    )
    args = parser.parse_args()
    package(Path(args.output))


if __name__ == "__main__":
    main()
