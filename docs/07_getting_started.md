# 7. Getting started

You do **not** need any dataset to see EchoHawk work — a built-in simulator
drives most demos.

## 7.1 Install

```bash
# Option A: conda (recommended; the CNN needs PyTorch)
conda activate rl            # an env with python, numpy, scipy, scikit-learn, matplotlib, pytorch

# Option B: pip, core only (everything except the CNN demo)
pip install -r requirements.txt
```

Core demos need only `numpy`, `scipy`, `scikit-learn`, `matplotlib`. The CNN
(`06_train_cnn.py`) additionally needs `torch`; real-audio loaders use
`soundfile`.

## 7.2 Run the demos (no data required)

```bash
python examples/03_features_bpf_demo.py     # spectrogram + blade-passing frequency
python examples/01_detection_demo.py        # synthetic detection: ROC + confusion
python examples/02_array_doa_demo.py        # direction-finding + error-vs-SNR
python examples/07_doa_tracking_demo.py     # bearing tracking over time
```

Each writes PNGs to `results/`. Expected ballpark: detection AUC ≈ 0.93; DOA RMSE
sub-degree at high SNR; BPF within ~2 Hz of truth.

## 7.3 Run on real data (optional)

```bash
# Recorded audio (drone vs everyday sound)
git clone https://github.com/saraalemadi/DroneAudioDataset data/drone_audio_dataset
python examples/04_detection_real.py        # Random-Forest detection on real audio
python examples/06_train_cnn.py             # train + evaluate the CNN (GPU)

# Microphone-array direction-finding (download DREGON flights into data/dregon/)
python examples/05_doa_real.py
```

If a dataset is missing, the script prints guidance and exits cleanly.

## 7.4 What each figure means

| Figure | How to read it |
|---|---|
| `features_bpf.png` | Left: the drone's spectrogram (bright harmonic lines). Right: the Harmonic Product Spectrum peaking at the estimated blade-passing frequency vs the true one. |
| `roc.png` / `roc_real.png` | Detection rate vs false-alarm rate; closer to the top-left and higher AUC is better. |
| `confusion*.png` | Counts of correct/incorrect calls; strong diagonal = good. |
| `doa_spectra.png` | Estimated "energy vs angle" for each method; all should peak at the true bearing, MUSIC sharpest. |
| `doa_error_vs_snr.png` | Bearing error shrinking as signal strength rises. |
| `doa_tracking.png` | True bearing vs raw vs smoothed — the tracked line is cleaner. |
| `cnn_training.png` | Training/validation loss and validation AUC per epoch. |

## 7.5 Run the tests

```bash
pytest -q          # fast synthetic smoke tests (no GPU, no data)
```

The tests assert that detection clears a minimum AUC, that the direction-finder
recovers a known angle within a few degrees, and that the blade-passing-frequency
estimate is close — a quick guarantee the pipeline is intact.

## 7.6 Troubleshooting

- **A real-data script says "place the dataset here"** — that's expected; it
  means the dataset folder is empty. Follow the printed path.
- **A corporate content filter blocks a download** — fetch the dataset on an
  unrestricted machine and copy it into the named folder. (Do not disable
  certificate verification.)
- **The CNN demo can't find `torch`** — install PyTorch in your environment, or
  just run the non-CNN demos.

➡ Next: [API reference](08_api_reference.md).
