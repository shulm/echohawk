# 4. Detection & classification

This chapter explains how EchoHawk decides "drone or not a drone", and what the
numbers mean. For the underlying ideas, see the [concepts primer](02_concepts_primer.md).

## 4.1 From sound to features

Every clip is resampled to 16 kHz and cut into ~1-second windows. Two feature
representations are produced (`features.py`):

- **Log-mel spectrogram** — the time-frequency "picture" (used by the CNN).
- **`feature_vector`** — a fixed-length summary used by the Random Forest:
  the mean and spread of the first 13 **MFCCs**, plus the spectral **centroid**
  (where the energy sits), **spread**, **flatness** (tonal vs noisy), and the
  estimated **blade-passing frequency** from the Harmonic Product Spectrum.

These features capture exactly what separates a drone from background: a tonal,
harmonically-structured, low-fundamental sound.

## 4.2 Two classifiers

- **Random Forest baseline** (`build_baseline`): 300 decision trees over the
  summary feature vector. Fast, transparent, and a fair yardstick.
- **DroneCNN** (`build_cnn`): a compact convolutional network reading the
  log-mel spectrogram directly, learning the harmonic pattern end-to-end. Trained
  on a GPU (`conda activate rl`) with class-weighting to handle the fact that
  drones are the minority class.

## 4.3 Two benchmarks, on purpose

EchoHawk is measured on **two** datasets that test different things:

**(a) Synthetic benchmark — the hard case.** The simulator pits drones against
**ground-vehicle confusers** that also have low-frequency harmonics, across a
wide SNR range (−15 to +3 dB). Because the classes genuinely overlap, the
baseline scores a realistic **ROC-AUC ≈ 0.93** — deliberately *not* a perfect
score. This is where the operationally hard "drone vs vehicle" question is tested.

**(b) Real audio — the DroneAudioDataset.** Real recorded drone audio versus
everyday sounds (ESC-50 / speech). These negatives are "easy" (very unlike a
drone), so scores are high. The value here is showing the pipeline works on real
recordings — *and* surfacing a subtle evaluation pitfall (next section).

## 4.4 Results on real audio (honest, session-grouped)

Measured on a held-out test split where **no recording session appears in both
train and test** (see [chapter 6](06_evaluation_rigor_and_leakage.md) for why
this matters):

| Metric | Random Forest | DroneCNN |
|---|---|---|
| ROC-AUC | 0.9815 | **0.9941** |
| Accuracy | 0.9638 | **0.9840** |
| Pd @ 1% FA | 0.7453 | **0.9384** |
| Pd @ 5% FA | 0.9185 | **0.9784** |
| Pd @ 10% FA | 0.9499 | **0.9834** |

**Reading this:** at a strict 1% false-alarm rate, the CNN catches ~94% of
drones versus ~75% for the baseline — the kind of low-false-alarm performance
that matters operationally. The CNN wins because it sees the full spectro-temporal
pattern, not just a hand-made summary.

**Dataset at a glance:** 19,275 one-second clips (2,420 drone / 16,855 non-drone)
from 10,629 source recordings (257 of them drone sessions), split 70/30 by
recording.

## 4.5 Why accuracy alone is misleading

Because only ~12.5% of clips are drones, a lazy "always say no" would already be
~87% accurate. That's why EchoHawk leads with **AUC** and **Pd at a fixed false
alarm rate**, not raw accuracy — they reflect how well drones are actually caught
without drowning operators in false alarms.

➡ Next: [Direction-finding & tracking](05_direction_finding_and_tracking.md).
