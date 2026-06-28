"""Track A (real data) -- drone detection on real datasets.

Supports multiple datasets via the REAL_SOURCE environment variable:
    REAL_SOURCE=dad_github : DroneAudioDataset (GitHub)
    REAL_SOURCE=dads       : drone-audio-detection-samples (HuggingFace)

If REAL_SOURCE is not set, it defaults to whichever dataset is found locally.

Workflow
--------
1. Load dataset clips.
2. Extract ``features.feature_vector`` for every clip.
3. Grouped split 70 / 30 train / test split  (seed=0 — fixed). Clips from the same 
   source recording stay on the same side.
4. Fit ``models.build_baseline("rf")`` on the *training* split only.
5. Evaluate on the held-out test split; write ROC + confusion figures.
6. Print accuracy, ROC-AUC, and Pd@1%, 5%, 10% FA.

Hard rules obeyed
-----------------
* All random seeds fixed at 0.
* No test-set information used during feature extraction or training.
* Metrics come from the held-out split — no cross-validation leak.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Make sure the repo root is on sys.path so `import echohawk` works when
# the script is run from any directory.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from echohawk import features, models, evaluate  # noqa: E402
from echohawk.data_io import load_dads, load_dads_hf, load_drone_audio_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — edit here or use env vars
# ---------------------------------------------------------------------------
RESULTS       = os.path.join(_REPO, "results")
FS            = 16_000
CLIP_S        = 1.0
SEED          = 0
MAX_PER_CLASS = 2000    # clips per class in streaming mode

_DAD_GITHUB_ROOT = os.path.join(_REPO, "data", "drone_audio_dataset")
_DADS_LOCAL_ROOT = os.path.join(_REPO, "data", "dads")

_has_dad_github = os.path.isdir(os.path.join(_DAD_GITHUB_ROOT, "Binary_Drone_Audio"))
_has_dads_local = (
    os.path.isdir(os.path.join(_DADS_LOCAL_ROOT, "0")) and
    os.path.isdir(os.path.join(_DADS_LOCAL_ROOT, "1"))
)

REAL_SOURCE = os.environ.get("REAL_SOURCE", "").lower()
if not REAL_SOURCE:
    if _has_dad_github:
        REAL_SOURCE = "dad_github"
    else:
        REAL_SOURCE = "dads"

USE_LOCAL_DADS = bool(int(os.environ.get("DADS_LOCAL", "0"))) or _has_dads_local

os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    dataset_name = ""
    t0 = time.time()
    
    if REAL_SOURCE == "dad_github":
        print(f"Loading DroneAudioDataset from local path: {_DAD_GITHUB_ROOT} ...")
        X_list, y, groups, fs = load_drone_audio_dataset(_DAD_GITHUB_ROOT, fs=FS, clip_s=CLIP_S)
        dataset_name = "DroneAudioDataset (GitHub)"
    else:
        if USE_LOCAL_DADS:
            print(f"Loading DADS from local path: {_DADS_LOCAL_ROOT} ...")
            X_list, y, groups, fs = load_dads(_DADS_LOCAL_ROOT, fs=FS, clip_s=CLIP_S)
        else:
            print("Loading DADS via HuggingFace streaming ...")
            X_list, y, groups, fs = load_dads_hf(
                repo_id="geronimobasso/drone-audio-detection-samples",
                fs=FS,
                clip_s=CLIP_S,
                max_per_class=MAX_PER_CLASS,
                seed=SEED,
            )
        dataset_name = "DADS"

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    
    unique_groups = np.unique(groups)
    print(f"  {len(X_list)} clips total  "
          f"(class 0 / no-drone: {n0},  class 1 / drone: {n1})  "
          f"fs={fs} Hz  clip={CLIP_S}s")
    print(f"  {len(unique_groups)} unique source recordings.")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # 2. Feature extraction (train + test clips together; labels not used)
    # -----------------------------------------------------------------------
    print("Extracting features ...")
    t0 = time.time()
    F = models.features_matrix(X_list, fs, features.feature_vector)
    print(f"  Feature matrix: {F.shape}  ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 3. Grouped train / test split  (seed fixed → reproducible)
    # -----------------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, test_idx = next(gss.split(F, y, groups))
    
    F_tr, F_te = F[train_idx], F[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"  Train: {len(y_tr)} clips  |  Test: {len(y_te)} clips")

    # -----------------------------------------------------------------------
    # 4. Train classifier on TRAIN split only
    # -----------------------------------------------------------------------
    print("Training Random Forest baseline ...")
    clf = models.build_baseline("rf", seed=SEED)
    clf.fit(F_tr, y_tr)

    # -----------------------------------------------------------------------
    # 5. Evaluate on held-out TEST split
    # -----------------------------------------------------------------------
    scores = clf.predict_proba(F_te)[:, 1]
    y_hat  = (scores >= 0.5).astype(int)

    acc   = float((y_hat == y_te).mean())
    auc   = evaluate.plot_roc(y_te, scores,
                              os.path.join(RESULTS, "roc_real.png"),
                              title=f"Drone detection ROC ({dataset_name})")
    evaluate.plot_confusion(y_te, y_hat,
                            os.path.join(RESULTS, "confusion_real.png"))
    
    pd01  = evaluate.pd_at_pfa(y_te, scores, pfa=0.01)
    pd05  = evaluate.pd_at_pfa(y_te, scores, pfa=0.05)
    pd10  = evaluate.pd_at_pfa(y_te, scores, pfa=0.10)

    # -----------------------------------------------------------------------
    # 6. Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 58)
    print(f"  Real-data results  ({dataset_name} — held-out test split)")
    print("=" * 58)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  Pd@1%FA    : {pd01:.4f}")
    print(f"  Pd@5%FA    : {pd05:.4f}")
    print(f"  Pd@10%FA   : {pd10:.4f}")
    print(f"  Train size : {len(y_tr)} clips  "
          f"(drone={int((y_tr==1).sum())}, no-drone={int((y_tr==0).sum())})")
    print(f"  Test  size : {len(y_te)} clips  "
          f"(drone={int((y_te==1).sum())}, no-drone={int((y_te==0).sum())})")
    print(f"  Clips/class: no-drone={n0}, drone={n1}")
    
    unique_train_groups = len(np.unique(groups[train_idx]))
    unique_test_groups = len(np.unique(groups[test_idx]))
    print(f"  Recordings : train={unique_train_groups}, test={unique_test_groups}")
    print("=" * 58)
    print(f"\nFigures -> {RESULTS}/roc_real.png, confusion_real.png")


if __name__ == "__main__":
    main()
