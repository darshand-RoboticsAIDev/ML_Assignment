"""Shared metric helper used by both the training script and the Streamlit app,
so the numbers in the README and on the app come from the same code."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba, labels=None):
    """Return the six assignment metrics for a multi-class problem.

    Precision / Recall / F1 use macro averaging (each bean variety counts
    equally, so the small BOMBAY class is not drowned out by DERMASON).
    AUC is the macro one-vs-rest ROC AUC computed from predicted probabilities.
    """
    if labels is None:
        labels = np.arange(y_proba.shape[1])

    present = np.unique(y_true)
    if len(present) < 2:
        auc = float("nan")  # AUC undefined when only one class is present
    elif len(present) == len(labels):
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
    else:
        # Uploaded file holds only some classes: score the classes that appear
        cols = [list(labels).index(c) for c in present]
        sub = y_proba[:, cols]
        sub = sub / sub.sum(axis=1, keepdims=True)
        auc = (roc_auc_score(y_true, sub, multi_class="ovr", average="macro", labels=present)
               if len(present) > 2 else roc_auc_score(y_true == present[1], sub[:, 1]))

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
