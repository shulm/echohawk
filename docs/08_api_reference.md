# 8. API reference

A guided tour of the public functions. Signatures are shown in simplified form;
the authoritative details are the docstrings in each module. Conventions:
sample rate `fs` is in hertz; azimuths are in degrees unless noted; microphone
positions are an `(M, 2)` array of metres.

## `simulate.py` — synthetic data

- **`drone_signal(fs=16000, dur=1.0, bpf=120, n_harmonics=8, harmonic_decay=0.7, fm_hz=3.0, fm_depth=0.02, snr_db=10, seed=None)`**
  A drone-like harmonic stack at blade-passing frequency `bpf`, with rotor
  turbulence and a small RPM wobble, at the requested SNR. → 1-D array.
- **`vehicle_signal(...)`** A low-frequency harmonic source (ground
  vehicle/generator) — the *hard* negative.
- **`background_signal(..., color="pink")`** Coloured background noise (easy
  negative).
- **`confuser_signal(...)`** Tonal distractors (sirens, machinery, bird-like).
- **`make_detection_dataset(n_per_class=200, snr_range=(-15, 3), seed=0)`**
  A balanced drone/non-drone set with realistic overlap. → `(X_list, y, fs)`.
- **`uca_positions(n_mics=8, radius=0.10)`** Uniform circular array geometry. →
  `(M, 2)`.
- **`simulate_array(signal, doa_deg, positions, fs=16000, snr_db=10, seed=None)`**
  Propagate one far-field source across the array. → `(M, N)` channels × samples.

## `features.py` — feature extraction

- **`stft_mag(x, fs, n_fft=512, hop=256)`** → `(freqs, times, magnitude)`.
- **`log_mel_spectrogram(x, fs, n_mels=40)`** The CNN's input picture. → `(n_mels, T)`.
- **`mfcc(x, fs, n_mfcc=13)`** Compact cepstral features. → `(n_mfcc, T)`.
- **`harmonic_product_spectrum(x, fs, ...)`** → `(f0_hz, freqs, hps)`; `f0_hz` is
  the estimated blade-passing frequency.
- **`feature_vector(x, fs)`** Fixed-length summary (MFCC stats + spectral
  centroid/spread/flatness + BPF) for the baseline classifier. → 1-D array.

## `models.py` — classifiers

- **`build_baseline(kind="rf", seed=0)`** A scikit-learn pipeline
  (standardise → Random Forest or SVM). → estimator with `.fit/.predict_proba`.
- **`features_matrix(X, fs, feature_fn)`** Stack `feature_fn` over many clips. →
  2-D array.
- **`build_cnn(n_mels=40, n_classes=2)`** A small PyTorch CNN over log-mel
  spectrograms (raises a helpful error if PyTorch is absent).

## `beamforming.py` — direction of arrival

- **`steering_vector(positions, theta_rad, freq, c=343)`** Array manifold for one
  angle/frequency. → length-`M` complex vector.
- **`steering_matrix(positions, az_rad, freq, c=343)`** Same for many angles at
  once (vectorised). → `(M, G)`.
- **`bartlett_spectrum(X, fs, positions, az_grid_deg, fmin, fmax)`** Delay-and-sum
  power vs azimuth. → `(az_grid_deg, power)`.
- **`mvdr_spectrum(...)`** Capon/MVDR spectrum (sharper). → `(az, power)`.
- **`music_spectrum(..., n_sources=1)`** MUSIC pseudospectrum (high resolution).
  → `(az, power)`.
- **`estimate_doa(az_grid_deg, P)`** Peak of a spectrum. → azimuth in degrees.
- **`gcc_phat(x1, x2, fs, max_tau=None)`** Time-delay estimate between two
  channels. → `(tau, cc, lags)`.
- **`srp_phat(X, fs, positions, az_grid_deg)`** Robust azimuth power map from all
  pairs. → `(az, power)`.

## `track.py` — temporal smoothing

- **`AzimuthTracker(...)`** A 1-D Kalman/alpha-beta smoother. Call
  `.update(measurement_deg)` per frame; returns the smoothed azimuth.

## `evaluate.py` — metrics & plots

- **`pd_at_pfa(y_true, scores, pfa=0.01)`** Probability of detection at a fixed
  false-alarm rate. → float.
- **`plot_roc(y_true, scores, path, title=...)`** Save an ROC figure. → AUC.
- **`plot_confusion(y_true, y_pred, path, labels=...)`** Save a confusion matrix.

## `data_io.py` — real datasets (leakage-safe)

- **`load_drone_audio_dataset(root, fs=16000, clip_s=1.0)`** Load the
  DroneAudioDataset, returning a per-clip **`groups`** array keyed on the
  recording **session** so splits can't leak. → `(X_list, y, groups, fs)`.
- **`load_dads(root, ...)`** Load DADS from a local folder-of-WAVs.
- **`load_dregon_flight(path)`** Load a DREGON 8-channel flight (+ geometry, and
  ground-truth azimuth when available) for real direction-finding.

> Tip: the leakage-safe `groups` array is the detail that makes the real-data
> numbers trustworthy — see [chapter 6](06_evaluation_rigor_and_leakage.md).

➡ Next: [Glossary](09_glossary.md).
