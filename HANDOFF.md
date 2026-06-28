# EchoHawk — Build & Extension Spec (for the implementation agent)

**Working model:** the research lead has designed the architecture, written and
**validated** the reference implementation on synthetic data, and defined the
tasks below. The implementation agent ("doer") executes the tasks on the user's
machine and reports back; the lead reviews against the acceptance criteria.

**Environment (use this for everything):**
```bash
conda activate rl        # python, numpy, scipy, scikit-learn, matplotlib, PyTorch + CUDA
```
Install extras if missing: `pip install soundfile librosa huggingface_hub`.

**Hard rules**
- Never commit raw audio (already in `.gitignore`). Keep datasets under `data/`.
- Keep all random seeds; results must be reproducible.
- Claims in the README must match measured numbers. Synthetic = "synthetic".
  Real-data results get their own clearly-labelled table.
- Do not invent capabilities. If something underperforms, report it.

---

## Status (done & validated by the lead — do not redo)
- Library: `simulate`, `features`, `beamforming` (Bartlett/MVDR/MUSIC/GCC-PHAT/
  SRP-PHAT, vectorised), `models` (RF baseline + optional PyTorch CNN), `evaluate`.
- All three `examples/*.py` run end-to-end on synthetic data.
- Validated numbers: detection ROC-AUC ≈ 0.93; DOA RMSE ~0.3° (≥+5 dB) rising to
  ~2–4° at −10 dB; HPS BPF 110→109 Hz.

---

## Tasks (in order, each with acceptance criteria)

### T1 — Regenerate baseline figures on the `rl` env
- Delete `results/*.png`, run the three example scripts, confirm five figures.
- **Accept:** five figures present; numbers within ±10% of the values above.

### T2 — Real detection on DADS
1. Add `scripts/download_data.sh` (content below) and fetch DADS into `data/dads/`.
2. New module `echohawk/data_io.py`: `load_dads(root, fs=16000, clip_s=1.0)` →
   returns `(X_list, y, fs)`, segmenting/zero-padding clips to `clip_s`.
3. New script `examples/04_detection_real.py`: reuse `features.feature_vector` +
   `models.build_baseline`, stratified train/test, write
   `results/roc_real.png`, `results/confusion_real.png`; print AUC, accuracy,
   Pd@1%FA.
- **Accept:** runs on real DADS; a real-data results table added to README;
  AUC reported honestly (do not tune on the test split).

### T3 — Real DOA on DREGON
1. `load_dregon_flight(path)` in `data_io.py`: read the 8-channel WAV + the
   array geometry (from the DREGON docs) + ground-truth source direction.
2. `examples/05_doa_real.py`: run `music_spectrum` and `srp_phat`, compare to
   ground truth over time, write `results/doa_real.png`, print azimuth RMSE.
- **Accept:** azimuth error reported vs. ground truth on ≥1 flight; note the
  ego-noise caveat (DREGON's array is on the UAV).

### T4 — Train the CNN (GPU)
- `examples/06_train_cnn.py`: log-mel tensors → `models.build_cnn` → train with
  Adam, early stopping; compare to the RF baseline on the same split.
- Recipe: `lr=1e-3`, `batch=32`, `epochs<=40`, `weight_decay=1e-4`, augment
  (add noise, time/pitch shift). Save `results/cnn_training.png` (loss/acc) and
  a confusion matrix.
- **Accept:** CNN trains on GPU (`conda activate rl`); report whether it beats
  the baseline; commit the curve.

### T5 — Optional polish (only if T1–T4 are clean)
- Kalman tracker over DOA (`echohawk/track.py`) + a tracking figure.
- `Dockerfile` (CPU) and a minimal GitHub Action running `python -m py_compile`
  + the synthetic detection demo. These satisfy the role's CI/CD + Docker
  "nice-to-haves".

### T6 — Publish to GitHub (account: https://github.com/shulm)
- Add `scripts/push_to_github.sh` (content below).
- **The `git push` step needs the user's GitHub credentials — the user runs it /
  authenticates; the agent must not enter or store the user's password/token.**
- **Accept:** repo live at `github.com/shulm/echohawk`, README renders with
  figures, clean history, no data committed.

---

## File contents to add

`scripts/download_data.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/dads data/dregon
# DADS (Hugging Face) — requires: pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="geronimobasso/drone-audio-detection-samples",
                  repo_type="dataset", local_dir="data/dads")
PY
echo "DREGON: download flights manually from https://dregon.inria.fr/ into data/dregon/"
```

`scripts/push_to_github.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
git init -b main
git add .
git commit -m "EchoHawk: acoustic drone detection, classification & DOA"
# Create the empty repo first at https://github.com/new (name: echohawk), then:
git remote add origin https://github.com/shulm/echohawk.git
git push -u origin main     # <-- user authenticates here (browser / PAT / gh auth login)
```
Alternative one-liner if the GitHub CLI is authenticated by the user:
`gh repo create shulm/echohawk --public --source=. --remote=origin --push`

---

## Lead review checklist
- [ ] Repo runs from a clean `conda activate rl` with only `requirements.txt`.
- [ ] Synthetic and real results are in separate, correctly-labelled tables.
- [ ] No raw audio or large binaries in git history.
- [ ] DOA error and detection AUC reported against held-out / ground-truth data.
- [ ] README figures render on GitHub; claims match measured numbers.
- [ ] Seeds fixed; `python -m py_compile echohawk/*.py examples/*.py` passes.
