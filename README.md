# EchoHawk — Passive Acoustic Drone Detection, Classification & Direction-Finding

A compact, readable reference implementation of the core algorithms behind
acoustic counter-UAS systems: detect a drone from its sound, classify it
against confusers (vehicles, machinery, birds), estimate the **blade-passing
frequency**, and find its **direction of arrival** with a microphone array.

The repository is fully runnable **without downloading any data** — a physically
transparent simulator generates drone harmonic signatures and a microphone
array, so every algorithm can be exercised and verified end-to-end. Hooks for
real public datasets (DREGON, DADS) are included.

> Built as an open, methods-focused demonstrator. It is not affiliated with any
> company and uses only public/synthetic data.

## Paper

[**EchoHawk: A Reproducible Acoustic Pipeline for Drone Detection, Classification, and Direction-Finding, with a Cautionary Study of Session-Level Data Leakage**](report/echohawk-arxiv.pdf)

[Read the paper (PDF)](report/echohawk-arxiv.pdf) | [arXiv:2606.29589](https://arxiv.org/abs/2606.29589)

**Citation:** Shulman, D., 2026. EchoHawk: A Reproducible Acoustic Pipeline for Drone Detection, Classification, and Direction-Finding, with a Cautionary Study of Session-Level Data Leakage. arXiv preprint [arXiv:2606.29589](https://arxiv.org/abs/2606.29589).

## What's inside

| Capability | Method | Module |
|---|---|---|
| Drone vs. non-drone detection | Log-mel / MFCC + harmonic features → Random Forest (baseline) and a CNN | `features.py`, `models.py` |
| Blade-passing-frequency estimation | Harmonic Product Spectrum | `features.py` |
| Direction of arrival (single array) | Delay-and-Sum, MVDR/Capon, MUSIC | `beamforming.py` |
| Time-delay localisation | GCC-PHAT, SRP-PHAT | `beamforming.py` |
| Synthetic data + array simulation | Far-field plane-wave model | `simulate.py` |

## Results (synthetic benchmark, reproducible with the example scripts)

**Detection** (5-fold CV, balanced drone vs. vehicle/confuser/noise, SNR −15…+3 dB):
ROC-AUC ≈ **0.93**, accuracy ≈ **0.86**. Classes overlap by design (drones and
ground vehicles are both low-frequency harmonic sources), so this is a realistic
rather than a trivially separable task.

**Direction of arrival** (8-mic uniform circular array, r = 10 cm):

| Array SNR | Bartlett | MVDR | MUSIC |
|---|---|---|---|
| −10 dB | ~4° | ~1.6° | ~2.3° |
| 0 dB | ~0.6° | ~0.8° | ~0.9° |
| +10 dB | ~0.3° | ~0.3° | ~0.3° |

**Blade-passing frequency**: 110 Hz (true) → 109 Hz (estimated via HPS).

Figures: `results/roc.png`, `results/confusion.png`, `results/doa_spectra.png`,
`results/doa_error_vs_snr.png`, `results/features_bpf.png`, `results/doa_tracking.png`.

## Results — Real data (DroneAudioDataset)

> [!NOTE]
> Run `python examples/04_detection_real.py` (RF Baseline) and `python examples/06_train_cnn.py` (CNN) to reproduce. The scripts use the local `DroneAudioDataset` clone by default.
> DADS (HuggingFace) is also supported as an alternative for unrestricted networks.

**Dataset**: [DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset) (GitHub)  
*Caveat: DroneAudioDataset negatives are general environmental/speech sounds (ESC-50 / Speech Commands), so this is an "easy-negative" benchmark; operationally hard confusers (ground vehicles / other harmonic sources) are covered by the synthetic benchmark. Furthermore, leakage is tightly controlled here at the SESSION level. Drone clips are grouped by recording session (257 total sessions), and negatives are grouped per-file, ensuring no individual recording crosses between train and test splits.*

**Split**: `GroupShuffleSplit` 70 / 30 train / test, seed=0. Split grouped by session recording.
**Classifiers**: 
- **RF Baseline**: Random Forest (300 trees, `sklearn`) over `features.feature_vector` (MFCC statistics + spectral + HPS).
- **CNN**: PyTorch CNN over log-mel spectrograms.

### Detection — model comparison
*Note: These numbers reflect strict session-level grouping to prevent leakage. Previous file-grouped results artificially inflated RF AUC to 0.987 and CNN AUC to 0.997 because adjacent clips from the same continuous drone session were split across train and test sets.*

| Metric | RF Baseline | DroneCNN |
|--------|-------------|----------|
| ROC-AUC | 0.9815 | **0.9941** |
| Accuracy | 0.9638 | **0.9840** |
| Pd @ 1 % FA | 0.7453 | **0.9384** |
| Pd @ 5 % FA | 0.9185 | **0.9784** |
| Pd @ 10 % FA | 0.9499 | **0.9834** |

**Conclusion**: The CNN strongly outperforms the Random Forest baseline under strict session-level cross-validation, especially in the critical low false-alarm regime (Pd @ 1% FA jumps from 74.5% to 93.8%). The CNN's ability to learn spatial-temporal patterns over the full log-mel spectrogram provides much richer discriminative power than the hand-crafted MFCC summary statistics used by the RF baseline.

### Dataset Details
| Split | Total Clips | Drone | No-Drone | Unique Sessions |
|-------|-------------|-------|----------|-----------------|
| Train+Val | 13 583 | 1 819 | 11 764 | 7 440 |
| Test | 5 692 | 601 | 5 091 | 3 189 |
| **Total** | **19 275** | **2 420** | **16 855** | **10 629** |

Figures: `results/roc_cnn_real.png`, `results/confusion_cnn.png`, `results/cnn_training.png`.

## Quickstart

```bash
# uses an env with python, numpy, scipy, scikit-learn, matplotlib
conda activate rl          # or: pip install -r requirements.txt

python examples/01_detection_demo.py        # detection ROC + confusion matrix
python examples/02_array_doa_demo.py        # DOA spectra + error-vs-SNR
python examples/03_features_bpf_demo.py     # spectrogram + BPF estimate
python examples/04_detection_real.py        # real-data RF detection
python examples/05_doa_real.py              # real-data DOA (requires DREGON)
python examples/06_train_cnn.py             # real-data CNN training
python examples/07_doa_tracking_demo.py     # DOA tracking over time
```

All figures are written to `results/`.

## Repository layout

```
echohawk/
├── echohawk/            # library: simulate, features, beamforming, models, evaluate
├── examples/            # runnable demos that generate the figures in results/
├── results/             # output figures
├── data/                # (gitignored) place DREGON / DADS here; see data/README.md
├── requirements.txt
├── HANDOFF.md           # build/extension spec
└── README.md
```

## Methods in brief

- **Detection** turns each clip into a log-mel spectrogram / MFCCs plus harmonic
  cues (spectral flatness, HPS fundamental) and classifies it. A small CNN over
  log-mel spectrograms (`models.build_cnn`, PyTorch) is included for the richer
  model; train it on real audio with a GPU.
- **DOA** forms a wideband spatial spectrum by averaging narrowband covariance
  spectra across the drone harmonic band, then peak-picks the azimuth. MVDR and
  MUSIC give higher resolution than delay-and-sum at low SNR.
- **Localisation** uses GCC-PHAT time-delays across microphone pairs; SRP-PHAT
  aggregates them into a robust azimuth map — the basis for multi-sensor
  triangulation to a position.

## Limitations

- **Easy Negatives in Real Data**: The negatives in DroneAudioDataset are general sounds (ESC-50). Operationally hard confusers (lawnmowers, weed-whackers) are only evaluated on the synthetic benchmark.
- **Synthetic DOA**: The DOA benchmark primarily uses a synthetic array, with a script for real DREGON data, pending physical dataset access.

## Roadmap / extensions

- Train the CNN on DroneAudioDataset (session-grouped) — [DONE]; benchmark on **DADS** (Hugging Face) — pending (unblocked network).
- Run DOA on the **DREGON** 8-mic array recordings — code ready, pending physical dataset access.
- Add a Kalman tracker over DOA; add Docker + CI. [DONE]

## License

MIT — see `LICENSE`.

*Author: David Shulman, Ph.D.*
