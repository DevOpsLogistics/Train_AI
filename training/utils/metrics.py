"""seqeval metrics for Trainer."""

from __future__ import annotations

import numpy as np


def compute_ner_metrics(eval_pred):
    from seqeval.metrics import f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    pred_labels = []

    # label names injected at runtime via partial
    id2label = compute_ner_metrics.id2label  # type: ignore

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_out = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_seq.append(id2label[int(l)])
            pred_seq_out.append(id2label[int(p)])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_out)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }
