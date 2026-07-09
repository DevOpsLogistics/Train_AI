#!/usr/bin/env python3
"""Fine-tune PhoBERT phân loại quan hệ y khoa (marked entity pair)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "training" / "config_relation.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def prepare_datasets(cfg: dict, tokenizer, label2id: dict):
    from datasets import Dataset

    raw_dir = ROOT / cfg["dataset"]["raw_dir"].lstrip("./")
    max_length = cfg["model"]["max_length"]

    train_rows = load_jsonl(raw_dir / "train.jsonl")
    dev_rows = load_jsonl(raw_dir / "dev.jsonl")

    def encode(rows: list[dict]) -> Dataset:
        texts = [r["marked_text"] for r in rows]
        labels = [label2id[r["relation"]] for r in rows]
        enc = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        enc["labels"] = labels
        return Dataset.from_dict(enc)

    return encode(train_rows), encode(dev_rows)


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def train(cfg: dict | None = None, epochs: int | None = None, config_path: Path | None = None):
    cfg = cfg or load_config(config_path or DEFAULT_CONFIG)
    tcfg = cfg["training"]
    mcfg = cfg["model"]

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    relation_types = cfg["dataset"]["relation_types"]
    label2id = {l: i for i, l in enumerate(relation_types)}
    id2label = {i: l for l, i in label2id.items()}

    output_dir = ROOT / tcfg["output_dir"].lstrip("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {mcfg['base']}")
    print(f"Relation types: {relation_types}")

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base"], use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["base"],
        num_labels=len(relation_types),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds, eval_ds = prepare_datasets(cfg, tokenizer, label2id)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs or tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["eval_batch_size"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=True,
        save_total_limit=tcfg["save_total_limit"],
        seed=tcfg["seed"],
        report_to="none",
        logging_steps=10,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting relation training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = trainer.evaluate()
    print(f"Eval metrics: {metrics}")
    print(f"Model saved -> {final_dir}")
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="Train relation classifier on benh_an labels")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    train(epochs=args.epochs, config_path=Path(args.config))


if __name__ == "__main__":
    main()
