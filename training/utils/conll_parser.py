"""Parse CoNLL format (ViMedNER)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConLLSentence:
    words: list[str]
    labels: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.words)


def read_conll(path: Path) -> list[ConLLSentence]:
    sentences: list[ConLLSentence] = []
    words: list[str] = []
    labels: list[str] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(ConLLSentence(words=words, labels=labels))
                    words, labels = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # CoNLL: last column is label, rest is word (ViMedNER uses single token per line)
            words.append(parts[0])
            labels.append(parts[-1])

    if words:
        sentences.append(ConLLSentence(words=words, labels=labels))

    return sentences


def load_label_list(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if "O" not in labels:
        labels.append("O")
    return labels
