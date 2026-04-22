from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABEL_NAMES = ["Non-hate", "Hate"]


def metric_bundle(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, object]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(4.3, 3.6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(LABEL_NAMES))
    plt.xticks(tick_marks, LABEL_NAMES)
    plt.yticks(tick_marks, LABEL_NAMES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
