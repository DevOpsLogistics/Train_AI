#!/usr/bin/env python3
"""Fine-tune PhoBERT NER trên 110 bệnh án (CoNLL từ convert_benh_an.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "training" / "config_benh_an_ner.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_datasets(cfg: dict, tokenizer, label2id: dict):
    from datasets import Dataset

    from training.utils.conll_parser import read_conll
    from training.utils.tokenize import tokenize_and_align_batch

    raw_dir = ROOT / cfg["dataset"]["raw_dir"].lstrip("./")
    max_length = cfg["model"]["max_length"]

    def tokenize_and_align(examples):
        return tokenize_and_align_batch(examples, tokenizer, label2id, max_length)

    def load_split(name: str) -> Dataset:
        sents = read_conll(raw_dir / f"{name}.txt")
        return Dataset.from_dict(
            {"words": [s.words for s in sents], "labels": [s.labels for s in sents]}
        )

    train_ds = load_split("train").map(tokenize_and_align, batched=True, remove_columns=["words", "labels"])
    eval_ds = load_split("dev").map(tokenize_and_align, batched=True, remove_columns=["words", "labels"])
    return train_ds, eval_ds


def train(cfg: dict | None = None, epochs: int | None = None, config_path: Path | None = None):
    cfg = cfg or load_config(config_path or DEFAULT_CONFIG)
    tcfg = cfg["training"]
    mcfg = cfg["model"]

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    from training.utils.conll_parser import load_label_list
    from training.utils.metrics import compute_ner_metrics

    labels_path = ROOT / cfg["dataset"]["labels_file"].lstrip("./")
    label_list = load_label_list(labels_path)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    output_dir = ROOT / tcfg["output_dir"].lstrip("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {mcfg['base']}")
    print(f"Labels ({len(label_list)}): {label_list[:5]}...")

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base"], use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(
        mcfg["base"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds, eval_ds = prepare_datasets(cfg, tokenizer, label2id)
    compute_ner_metrics.id2label = id2label  # type: ignore

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs or tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["eval_batch_size"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        warmup_steps=100,
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
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_ner_metrics,
    )

    print("Starting NER training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (final_dir / "pipeline_label_map.json").write_text(
        json.dumps(
            {
                "CONDITION": "CONDITION",
                "SYMPTOM": "SYMPTOM",
                "LAB_TEST": "LAB_TEST",
                "MEDICATION": "MEDICATION",
                "PROCEDURE": "PROCEDURE",
                "DOSAGE": "DOSAGE",
                "OTHER": "OTHER",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = trainer.evaluate()
    print(f"Eval metrics: {metrics}")
    print(f"Model saved -> {final_dir}")
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="Train PhoBERT NER on benh_an labels")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    train(epochs=args.epochs, config_path=Path(args.config))


if __name__ == "__main__":
    main()
