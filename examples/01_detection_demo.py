"""Track A -- drone vs. background detection on synthetic data.

Builds a balanced dataset, extracts summary features, trains a Random Forest
baseline, and writes ROC + confusion-matrix figures to ``results/``.
"""
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echohawk import simulate, features, models, evaluate  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    print("Generating synthetic dataset ...")
    X, y, fs = simulate.make_detection_dataset(n_per_class=200)
    print(f"  {len(X)} clips @ {fs} Hz")

    print("Extracting features ...")
    F = models.features_matrix(X, fs, features.feature_vector)

    print("Training Random Forest baseline with 5-fold StratifiedKFold ...")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_metrics = []
    last_yte = last_scores = last_yhat = None
    for fold, (train_idx, test_idx) in enumerate(splitter.split(F, y), start=1):
        clf = models.build_baseline("rf", seed=fold - 1)
        clf.fit(F[train_idx], y[train_idx])
        scores = clf.predict_proba(F[test_idx])[:, 1]
        metrics = evaluate.binary_detection_metrics(y[test_idx], scores)
        fold_metrics.append(metrics)
        last_yte = y[test_idx]
        last_scores = scores
        last_yhat = (scores >= 0.5).astype(int)
        print(
            f"  fold {fold}: accuracy={metrics['accuracy']:.3f}  "
            f"AUC={metrics['auc']:.3f}  "
            f"Pd@1%FA={metrics['pd_at_1pct_fa']:.3f}"
        )

    auc_values = np.array([m["auc"] for m in fold_metrics])
    acc_values = np.array([m["accuracy"] for m in fold_metrics])
    pd01_values = np.array([m["pd_at_1pct_fa"] for m in fold_metrics])

    # Keep the figures lightweight by plotting the final held-out fold.
    evaluate.plot_roc(last_yte, last_scores, os.path.join(RESULTS, "roc.png"))
    evaluate.plot_confusion(last_yte, last_yhat, os.path.join(RESULTS, "confusion.png"))

    print(
        "\n5-fold results:  "
        f"accuracy={acc_values.mean():.3f}±{acc_values.std(ddof=1):.3f}  "
        f"AUC={auc_values.mean():.3f}±{auc_values.std(ddof=1):.3f}  "
        f"Pd@1%FA={pd01_values.mean():.3f}±{pd01_values.std(ddof=1):.3f}"
    )
    print(f"Figures -> {RESULTS}/roc.png, confusion.png")


if __name__ == "__main__":
    main()
