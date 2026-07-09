"""Tokenize + align labels cho PhoBERT (slow tokenizer)."""

from __future__ import annotations


def tokenize_and_align_batch(
    examples: dict,
    tokenizer,
    label2id: dict[str, int],
    max_length: int,
) -> dict:
    all_input_ids: list[list[int]] = []
    all_attention_mask: list[list[int]] = []
    all_labels: list[list[int]] = []

    for words, labels in zip(examples["words"], examples["labels"]):
        tokens: list[str] = []
        label_ids: list[int] = []

        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word, add_prefix_space=len(tokens) > 0)
            if not word_tokens:
                continue
            tokens.extend(word_tokens)
            lid = label2id[label]
            label_ids.extend([lid] + [-100] * (len(word_tokens) - 1))

        max_tokens = max_length - 2
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            label_ids = label_ids[:max_tokens]

        input_ids = tokenizer.build_inputs_with_special_tokens(
            tokenizer.convert_tokens_to_ids(tokens)
        )
        aligned_labels = [-100] + label_ids + [-100]

        if len(aligned_labels) != len(input_ids):
            # safety trim
            aligned_labels = aligned_labels[: len(input_ids)]
            while len(aligned_labels) < len(input_ids):
                aligned_labels.append(-100)

        all_input_ids.append(input_ids)
        all_attention_mask.append([1] * len(input_ids))
        all_labels.append(aligned_labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }
