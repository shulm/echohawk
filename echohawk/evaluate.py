"""Evaluation helpers: detection metrics and plotting."""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc,  # noqa: E402
                             confusion_matrix, ConfusionMatrixDisplay)


def pd_at_pfa(y_true, scores, pfa=0.01, method="interp", return_threshold=False):
    """Probability of detection at a fixed probability of false alarm.

    ``method="interp"`` returns the linearly interpolated ROC operating point.
    ``method="empirical"`` returns the best observed TPR with FPR <= ``pfa``.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    if method == "interp":
        value = float(np.interp(pfa, fpr, tpr))
        if return_threshold:
            return value, None
        return value
    if method == "empirical":
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        mask = fpr <= pfa
        if not np.any(mask):
            value, threshold = 0.0, float("inf")
        else:
            candidates = np.where(mask)[0]
            best = candidates[int(np.argmax(tpr[candidates]))]
            value = float(tpr[best])
            threshold = float(thresholds[best])
        if return_threshold:
            return value, threshold
        return value
    raise ValueError(f"unknown Pd@Pfa method: {method}")


def binary_detection_metrics(y_true, scores, threshold=0.5, pd_method="interp"):
    """Return pooled binary detection metrics for one held-out split."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, scores)),
        "pd_at_1pct_fa": pd_at_pfa(y_true, scores, pfa=0.01, method=pd_method),
        "pd_at_5pct_fa": pd_at_pfa(y_true, scores, pfa=0.05, method=pd_method),
        "pd_at_10pct_fa": pd_at_pfa(y_true, scores, pfa=0.10, method=pd_method),
    }


def mean_ci(values):
    """Return mean and normal-approximate 95% CI half-width."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    ci = float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))
    return mean, ci


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
