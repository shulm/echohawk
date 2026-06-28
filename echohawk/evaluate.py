"""Evaluation helpers: detection metrics and plotting."""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (roc_curve, auc, confusion_matrix,  # noqa: E402
                             ConfusionMatrixDisplay)


def pd_at_pfa(y_true, scores, pfa=0.01):
    """Probability of detection at a fixed probability of false alarm."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(pfa, fpr, tpr))


def plot_roc(y_true, scores, path, title="Drone detection ROC"):
    fpr, tpr, _ = roc_curve(y_true, scores)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {a:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Detection rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return a


def plot_confusion(y_true, y_pred, path, labels=("background", "drone")):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
