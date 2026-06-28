"""Track A -- drone vs. background detection on synthetic data.

Builds a balanced dataset, extracts summary features, trains a Random Forest
baseline, and writes ROC + confusion-matrix figures to ``results/``.
"""
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

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
    Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.3,
                                          stratify=y, random_state=0)

    print("Training Random Forest baseline ...")
    clf = models.build_baseline("rf")
    clf.fit(Xtr, ytr)
    scores = clf.predict_proba(Xte)[:, 1]
    yhat = (scores >= 0.5).astype(int)

    auc = evaluate.plot_roc(yte, scores, os.path.join(RESULTS, "roc.png"))
    evaluate.plot_confusion(yte, yhat, os.path.join(RESULTS, "confusion.png"))
    pd01 = evaluate.pd_at_pfa(yte, scores, pfa=0.01)
    acc = (yhat == yte).mean()

    print(f"\nResults:  accuracy={acc:.3f}  AUC={auc:.3f}  "
          f"Pd@1%FA={pd01:.3f}")
    print(f"Figures -> {RESULTS}/roc.png, confusion.png")


if __name__ == "__main__":
    main()
