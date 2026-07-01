"""Real-data Random-Forest detection with corrected grouped CV.

This script evaluates the RF baseline on DroneAudioDataset with corrected
recording/source grouping. It also supports a leakage-pathway study comparing
clip-level, file-level, and corrected recording-level grouping.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from echohawk import features, models, evaluate, eval_utils  # noqa: E402
from echohawk.data_io import load_drone_audio_dataset  # noqa: E402

RESULTS = os.path.join(_REPO, "results")
MANIFESTS = os.path.join(RESULTS, "manifests")
FS = 16_000
CLIP_S = 1.0
SEED = 0
DATA_ROOT = os.path.join(_REPO, "data", "drone_audio_dataset")


def _group_keys_for_mode(mode: str, corrected_keys, file_keys, n_samples: int):
    if mode == "corrected":
        return corrected_keys
    if mode == "file":
        return file_keys
    if mode == "clip":
        return np.asarray([f"clip:{i}" for i in range(n_samples)], dtype=object)
    raise ValueError(mode)


def _format_summary(summary):
    return {
        key: f"{item['mean']:.4f} ± {item['ci95']:.4f}"
        for key, item in summary.items()
    }


def run_rf_cv(X_list, y, corrected_keys, file_keys, folds, group_mode, save_prefix):
    group_keys = _group_keys_for_mode(group_mode, corrected_keys, file_keys, len(y))
    group_ids, _ = eval_utils.keys_to_group_ids(group_keys)
    cv_folds = eval_utils.make_group_kfold(y, group_keys, n_splits=folds)

    print(f"\nRF grouped CV: mode={group_mode}, folds={folds}")
    print("Group summary:", eval_utils.group_summary(y, group_keys))
    print("Extracting features ...")
    t0 = time.time()
    F = models.features_matrix(X_list, FS, features.feature_vector)
    print(f"  Feature matrix: {F.shape}  ({time.time() - t0:.1f}s)")

    fold_metrics = []
    pred_rows = []
    assertion_messages = []

    for fold, (train_idx, test_idx) in enumerate(cv_folds):
        messages = eval_utils.assert_group_disjoint(
            y,
            group_keys,
            {"train": train_idx, "test": test_idx},
        )
        assertion_messages.extend([f"fold={fold} {m}" for m in messages])

        clf = models.build_baseline("rf", seed=SEED + fold)
        clf.fit(F[train_idx], y[train_idx])
        scores = clf.predict_proba(F[test_idx])[:, 1]
        metrics = evaluate.binary_detection_metrics(y[test_idx], scores)
        fold_metrics.append(metrics)

        for sample_idx, label, score in zip(test_idx, y[test_idx], scores):
            pred_rows.append(
                {
                    "model": "rf",
                    "group_mode": group_mode,
                    "fold": fold,
                    "sample_index": int(sample_idx),
                    "label": int(label),
                    "score": float(score),
                    "group_key": str(group_keys[sample_idx]),
                    "file_key": str(file_keys[sample_idx]),
                }
            )

        print(
            f"  fold {fold}: "
            f"AUC={metrics['auc']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"Pd@1%FA={metrics['pd_at_1pct_fa']:.4f}  "
            f"Pd@5%FA={metrics['pd_at_5pct_fa']:.4f}  "
            f"Pd@10%FA={metrics['pd_at_10pct_fa']:.4f}"
        )

    summary = eval_utils.metrics_summary(fold_metrics)
    print("Summary:", _format_summary(summary))

    out_base = os.path.join(MANIFESTS, save_prefix)
    eval_utils.write_predictions_csv(out_base + "_predictions.csv", pred_rows)
    eval_utils.save_json(
        out_base + "_summary.json",
        {
            "model": "rf",
            "seed": SEED,
            "folds": folds,
            "group_mode": group_mode,
            "pd_at_pfa_method": "interp",
            "group_summary": eval_utils.group_summary(y, group_keys),
            "fold_metrics": fold_metrics,
            "summary": summary,
            "assertions": assertion_messages,
        },
    )
    eval_utils.save_json(
        out_base + "_split_manifest.json",
        {
            "model": "rf",
            "group_mode": group_mode,
            "folds": eval_utils.split_manifest(y, group_keys, file_keys, cv_folds),
        },
    )
    eval_utils.save_pip_freeze(os.path.join(MANIFESTS, "pip_freeze_rf.txt"))

    # Retain familiar figures using the final fold only.
    last_fold = cv_folds[-1]
    last_test = last_fold[1]
    last_scores = np.asarray([r["score"] for r in pred_rows if r["fold"] == folds - 1])
    evaluate.plot_roc(y[last_test], last_scores, os.path.join(RESULTS, "roc_real.png"))
    evaluate.plot_confusion(
        y[last_test],
        (last_scores >= 0.5).astype(int),
        os.path.join(RESULTS, "confusion_real.png"),
    )

    return summary, assertion_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--group-modes",
        nargs="+",
        default=["corrected"],
        choices=["clip", "file", "corrected"],
    )
    parser.add_argument("--leakage-study", action="store_true")
    args = parser.parse_args()

    os.makedirs(MANIFESTS, exist_ok=True)
    group_modes = ["clip", "file", "corrected"] if args.leakage_study else args.group_modes

    print(f"Loading DroneAudioDataset from local path: {DATA_ROOT} ...")
    t0 = time.time()
    X_list, y, groups, fs, metadata = load_drone_audio_dataset(
        DATA_ROOT,
        fs=FS,
        clip_s=CLIP_S,
        return_metadata=True,
    )
    if fs != FS:
        raise RuntimeError(f"unexpected fs={fs}")
    corrected_keys = metadata["group_keys"]
    file_keys = metadata["file_keys"]
    print(f"Loaded {len(y)} clips in {time.time() - t0:.1f}s")
    print(f"Class counts: negative={(y == 0).sum()}, drone={(y == 1).sum()}")
    print("Corrected group summary:", eval_utils.group_summary(y, corrected_keys))

    all_summaries = {}
    all_assertions = {}
    for mode in group_modes:
        prefix = f"real_rf_{mode}_{args.folds}fold"
        summary, messages = run_rf_cv(
            X_list,
            y,
            corrected_keys,
            file_keys,
            folds=args.folds,
            group_mode=mode,
            save_prefix=prefix,
        )
        all_summaries[mode] = summary
        all_assertions[mode] = messages

    eval_utils.save_json(
        os.path.join(MANIFESTS, f"real_rf_all_modes_{args.folds}fold_summary.json"),
        {
            "seed": SEED,
            "folds": args.folds,
            "pd_at_pfa_method": "interp",
            "summaries": all_summaries,
            "assertions": all_assertions,
        },
    )


if __name__ == "__main__":
    main()
