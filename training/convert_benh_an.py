#!/usr/bin/env python3
"""Chuyển benh_an_annotations.json → CoNLL (NER) + JSONL (relation)."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ANNOTATIONS = ROOT / "data" / "golden" / "benh_an_annotations.json"
DEFAULT_SAMPLES = ROOT / "data" / "samples" / "benh_an"
NER_OUT = ROOT / "data" / "training" / "benh_an_ner"
REL_OUT = ROOT / "data" / "training" / "benh_an_relation"

ENTITY_TYPES = (
    "CONDITION",
    "SYMPTOM",
    "LAB_TEST",
    "MEDICATION",
    "PROCEDURE",
    "DOSAGE",
    "OTHER",
)
RELATION_TYPES = (
    "TREATS",
    "DOSAGE_OF",
    "ASSOCIATED_WITH",
    "INDICATES",
)

E1_START, E1_END = "<e1>", "</e1>"
E2_START, E2_END = "<e2>", "</e2>"


@dataclass
class CharSpan:
    start: int
    end: int
    label: str
    text: str


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _word_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words: list[str] = []
    offsets: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        words.append(m.group())
        offsets.append((m.start(), m.end()))
    return words, offsets


def _locate_span(
    text: str,
    entity_text: str,
    evidence: str,
    occupied: list[tuple[int, int]],
) -> tuple[int, int] | None:
    needle = entity_text.strip()
    if not needle:
        return None

    candidates: list[tuple[int, int, int]] = []
    lower_text = text.lower()
    lower_needle = needle.lower()
    start = 0
    while True:
        idx = lower_text.find(lower_needle, start)
        if idx == -1:
            break
        span = (idx, idx + len(needle))
        if not any(_overlaps(span, occ) for occ in occupied):
            score = 0
            if evidence:
                ev = evidence.lower()
                ev_idx = lower_text.find(ev[: min(40, len(ev))])
                if ev_idx >= 0:
                    score -= abs(idx - ev_idx)
            candidates.append((score, idx, idx + len(needle)))
        start = idx + 1

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def _spans_to_bio(words: list[str], offsets: list[tuple[int, int]], spans: list[CharSpan]) -> list[str]:
    labels = ["O"] * len(words)
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        word_idxs = [
            i
            for i, (ws, we) in enumerate(offsets)
            if ws < span.end and we > span.start
        ]
        if not word_idxs:
            continue
        labels[word_idxs[0]] = f"B-{span.label}"
        for wi in word_idxs[1:]:
            labels[wi] = f"I-{span.label}"
    return labels


def _write_conll(path: Path, words: list[str], labels: list[str]) -> None:
    lines = [f"{w} {l}" for w, l in zip(words, labels)]
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")


def _mark_pair(text: str, src: tuple[int, int], tgt: tuple[int, int]) -> str:
    spans = [
        (src[0], src[1], E1_START, E1_END),
        (tgt[0], tgt[1], E2_START, E2_END),
    ]
    spans.sort(key=lambda x: x[0], reverse=True)
    out = text
    for s, e, left, right in spans:
        out = out[:s] + left + out[s:e] + right + out[e:]
    return out


def convert(
    annotations_path: Path,
    samples_dir: Path,
    ner_out: Path,
    rel_out: Path,
    seed: int = 42,
) -> dict:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    case_ids = sorted(data["cases"].keys())
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    n = len(case_ids)
    n_test = max(1, round(n * 0.1))
    n_dev = max(1, round(n * 0.1))
    n_train = n - n_dev - n_test
    splits = {
        "train": case_ids[:n_train],
        "dev": case_ids[n_train : n_train + n_dev],
        "test": case_ids[n_train + n_dev :],
    }

    ner_out.mkdir(parents=True, exist_ok=True)
    rel_out.mkdir(parents=True, exist_ok=True)

    labels = ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]
    (ner_out / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    stats = {"cases": n, "entities": 0, "relations": 0, "missed_entities": 0}

    for split_name, ids in splits.items():
        ner_parts: list[str] = []
        rel_rows: list[dict] = []

        for case_id in ids:
            case = data["cases"][case_id]
            text_path = ROOT / case["text_file"]
            if not text_path.exists():
                text_path = samples_dir / f"{case_id}.txt"
            text = text_path.read_text(encoding="utf-8").strip()

            words, offsets = _word_offsets(text)
            occupied: list[tuple[int, int]] = []
            char_spans: list[CharSpan] = []
            ent_span_map: dict[str, tuple[int, int]] = {}

            entities_sorted = sorted(
                case["entities"],
                key=lambda e: len(e.get("text", "")),
                reverse=True,
            )
            for ent in entities_sorted:
                etype = ent.get("type", "OTHER")
                if etype not in ENTITY_TYPES:
                    etype = "OTHER"
                located = _locate_span(text, ent["text"], ent.get("evidence", ""), occupied)
                if not located:
                    stats["missed_entities"] += 1
                    continue
                occupied.append(located)
                char_spans.append(CharSpan(located[0], located[1], etype, ent["text"]))
                ent_span_map[ent["text"].strip().lower()] = located
                stats["entities"] += 1

            bio = _spans_to_bio(words, offsets, char_spans)
            ner_parts.append("\n".join(f"{w} {l}" for w, l in zip(words, bio)) + "\n")

            for rel in case["relations"]:
                src = rel["source"].strip()
                tgt = rel["target"].strip()
                src_span = ent_span_map.get(src.lower())
                tgt_span = ent_span_map.get(tgt.lower())
                if not src_span or not tgt_span:
                    continue
                rel_type = rel.get("type", "ASSOCIATED_WITH")
                if rel_type not in RELATION_TYPES:
                    continue
                rel_rows.append(
                    {
                        "id": rel.get("relation_id", ""),
                        "case_id": case_id,
                        "relation": rel_type,
                        "source_text": src,
                        "target_text": tgt,
                        "source_type": next(
                            (e["type"] for e in case["entities"] if e["text"].strip().lower() == src.lower()),
                            "",
                        ),
                        "target_type": next(
                            (e["type"] for e in case["entities"] if e["text"].strip().lower() == tgt.lower()),
                            "",
                        ),
                        "text": text,
                        "marked_text": _mark_pair(text, src_span, tgt_span),
                    }
                )
                stats["relations"] += 1

        (ner_out / f"{split_name}.txt").write_text("\n".join(ner_parts) + "\n", encoding="utf-8")
        rel_path = rel_out / f"{split_name}.jsonl"
        rel_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rel_rows) + ("\n" if rel_rows else ""),
            encoding="utf-8",
        )

    meta = {
        "splits": {k: len(v) for k, v in splits.items()},
        "labels": labels,
        "entity_types": list(ENTITY_TYPES),
        "relation_types": list(RELATION_TYPES),
        **stats,
    }
    (ner_out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (rel_out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert benh_an annotations to training data")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--ner-out", default=str(NER_OUT))
    parser.add_argument("--rel-out", default=str(REL_OUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = convert(
        Path(args.annotations),
        Path(args.samples),
        Path(args.ner_out),
        Path(args.rel_out),
        seed=args.seed,
    )
    print("Converted training data:")
    print(f"  NER  → {args.ner_out}")
    print(f"  REL  → {args.rel_out}")
    print(f"  Cases: {meta['cases']} | entities: {meta['entities']} | relations: {meta['relations']}")
    print(f"  Missed entity spans: {meta['missed_entities']}")
    print(f"  Splits: {meta['splits']}")


if __name__ == "__main__":
    main()
