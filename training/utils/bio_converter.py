"""Convert BIO tags to entity spans."""

from __future__ import annotations

from dataclasses import dataclass

from training.utils.conll_parser import ConLLSentence


@dataclass
class EntitySpan:
    text: str
    start: int
    end: int
    label: str  # raw BIO type without B-/I-


def _char_offsets(words: list[str]) -> list[tuple[int, int]]:
    offsets = []
    pos = 0
    for i, word in enumerate(words):
        if i > 0:
            pos += 1  # space
        start = pos
        pos += len(word)
        offsets.append((start, pos))
    return offsets


def bio_to_entities(sentence: ConLLSentence) -> list[EntitySpan]:
    words = sentence.words
    labels = sentence.labels
    offsets = _char_offsets(words)
    text = sentence.text
    entities: list[EntitySpan] = []
    i = 0

    while i < len(labels):
        label = labels[i]
        if label == "O" or label == "":
            i += 1
            continue

        if label.startswith("B-"):
            etype = label[2:]
            start_idx = i
            i += 1
            while i < len(labels) and labels[i] == f"I-{etype}":
                i += 1
            char_start = offsets[start_idx][0]
            char_end = offsets[i - 1][1]
            entities.append(
                EntitySpan(
                    text=text[char_start:char_end],
                    start=char_start,
                    end=char_end,
                    label=etype,
                )
            )
        elif label.startswith("I-"):
            # Malformed — treat as B-
            etype = label[2:]
            start_idx = i
            i += 1
            while i < len(labels) and labels[i] == f"I-{etype}":
                i += 1
            char_start = offsets[start_idx][0]
            char_end = offsets[i - 1][1]
            entities.append(
                EntitySpan(
                    text=text[char_start:char_end],
                    start=char_start,
                    end=char_end,
                    label=etype,
                )
            )
        else:
            i += 1

    return entities
