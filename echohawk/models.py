"""Classifiers for drone detection.

Two paths:

    * ``build_baseline()`` -- a scikit-learn pipeline (standardise + Random
      Forest / SVM) over the summary feature vectors. Light, fast, always
      available; this is what generates the figures in ``results/``.

    * ``DroneCNN`` -- a small PyTorch CNN over log-mel spectrograms, for the
      richer model described in the project plan. Imported lazily so the rest
      of the package runs without torch installed. Train it once you have real
      audio (see HANDOFF.md).
"""
from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline(kind="rf", seed=0):
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                     n_jobs=-1)
    elif kind == "svm":
        clf = SVC(C=10.0, gamma="scale", probability=True, random_state=seed)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), clf)


def features_matrix(X, fs, feature_fn):
    """Stack ``feature_fn(x, fs)`` for every waveform into a 2-D array."""
    return np.vstack([feature_fn(x, fs) for x in X])


# --------------------------------------------------------------------------- #
# Optional deep model (requires torch)
# --------------------------------------------------------------------------- #
def build_cnn(n_mels=40, n_classes=2):
    """Return an untrained DroneCNN. Raises a helpful error if torch is absent."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the CNN. `pip install torch`, then see "
            "HANDOFF.md for the training recipe."
        ) from e

    class DroneCNN(nn.Module):
        def __init__(self, n_mels=40, n_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):           # x: (B, 1, n_mels, T)
            z = self.net(x).flatten(1)
            return self.head(z)

    return DroneCNN(n_mels, n_classes)
