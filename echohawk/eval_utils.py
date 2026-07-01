"""Shared experiment helpers for reproducible grouped evaluation."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from . import evaluate


METRIC_KEYS = [
    "auc",
    "accuracy",
    "pd_at_1pct_fa",
    "pd_at_5pct_fa",
    "pd_at_10pct_fa",
]


def keys_to_group_ids(keys):
    """Map stable string group keys to integer ids in first-seen order."""
    mapping = {}
    ids = []
    for key in keys:
        if key not in mapping:
            mapping[key] = len(mapping)
        ids.append(mapping[key])
    return np.asarray(ids, dtype=np.int64), mapping


def make_group_kfold(y, group_keys, n_splits=5):
    """Return deterministic GroupKFold train/test index pairs."""
    group_ids, _ = keys_to_group_ids(group_keys)
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(y)), y, group_ids))


def make_validation_split(train_idx, y, group_keys, seed=0, val_size=0.20):
    """Create a group-disjoint validation split inside an outer train fold."""
    train_idx = np.asarray(train_idx)
    group_ids, _ = keys_to_group_ids(group_keys[train_idx])
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    inner_train, val = next(splitter.split(np.zeros(len(train_idx)), y[train_idx], group_ids))
    return train_idx[inner_train], train_idx[val]


def assert_group_disjoint(y, group_keys, assignments, labels=(0, 1)):
    """Raise if any group key appears in more than one split for either class.

    ``assignments`` is a dict such as ``{"train": idx, "val": idx, "test": idx}``.
    Empty or omitted splits are ignored.
    """
    messages = []
    for label in labels:
        split_sets = {}
        for split, idx in assignments.items():
            idx = np.asarray(idx, dtype=np.int64)
            if idx.size == 0:
                continue
            split_sets[split] = set(group_keys[idx][y[idx] == label])
        names = sorted(split_sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = split_sets[a] & split_sets[b]
                if overlap:
                    sample = sorted(overlap)[:5]
                    raise AssertionError(
                        f"group leakage label={label} splits={a}/{b} "
                        f"count={len(overlap)} sample={sample}"
                    )
        counts = ", ".join(f"{name}={len(split_sets[name])}" for name in names)
        messages.append(f"label={label}: {counts}; pairwise disjoint=PASS")
    return messages


def group_summary(y, group_keys):
    out = {}
    for label, name in [(0, "negative"), (1, "drone")]:
        out[name] = int(len(set(group_keys[y == label])))
    out["total"] = int(len(set(group_keys)))
    return out


def metrics_summary(fold_metrics):
    summary = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in fold_metrics]
        mean, ci = evaluate.mean_ci(vals)
        summary[key] = {
            "mean": mean,
            "ci95": ci,
            "values": [float(v) for v in vals],
        }
    return summary


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_predictions_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "group_mode",
        "fold",
        "sample_index",
        "label",
        "score",
        "group_key",
        "file_key",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pip_freeze(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        capture_output=True,
        check=False,
    )
    text = proc.stdout if proc.returncode == 0 else proc.stderr
    path.write_text(text, encoding="utf-8")


def split_manifest(y, group_keys, file_keys, folds, val_splits=None):
    """Build a compact group-level split manifest."""
    val_splits = val_splits or {}
    manifest = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        fold_item = {"fold": fold, "splits": {}}
        split_indices = {
            "train": np.asarray(train_idx, dtype=np.int64),
            "test": np.asarray(test_idx, dtype=np.int64),
        }
        if fold in val_splits:
            split_indices["train"] = np.asarray(val_splits[fold][0], dtype=np.int64)
            split_indices["val"] = np.asarray(val_splits[fold][1], dtype=np.int64)
        for split, idx in split_indices.items():
            groups = sorted(set(group_keys[idx]))
            fold_item["splits"][split] = {
                "n_samples": int(idx.size),
                "n_drone": int((y[idx] == 1).sum()),
                "n_negative": int((y[idx] == 0).sum()),
                "n_groups": int(len(groups)),
                "groups": groups,
                "n_files": int(len(set(file_keys[idx]))),
            }
        manifest.append(fold_item)
    return manifest
