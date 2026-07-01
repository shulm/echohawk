# 5. Direction-finding & tracking

This chapter covers how EchoHawk estimates **where** a drone is — the bearing, or
**direction of arrival (DOA)** — and how it follows that bearing over time. The
plain-language version is in the [concepts primer](02_concepts_primer.md §2.5–2.7).

## 5.1 The array model

EchoHawk assumes a **far-field, planar** setup: the drone is far enough away that
its sound arrives as a flat wave, and the microphones lie in a plane (the demos
use an 8-microphone **uniform circular array**, 10 cm radius, from
`simulate.uca_positions`). A wave arriving from azimuth θ reaches each microphone
at a slightly different time; the pattern of those delays is the array's
"signature" for that direction, captured by a **steering vector**
(`beamforming.steering_vector`).

The simulator (`simulate.simulate_array`) produces realistic multi-channel
captures by applying exactly those delays plus sensor noise, so the
direction-finders can be tested against a known ground-truth angle.

## 5.2 The estimators

All operate **wideband**: they compute a short-time Fourier transform per
channel, build a spatial **covariance matrix** per frequency bin across the drone
harmonic band, then combine bins. They differ in how they turn that into an
angle:

- **Bartlett / Delay-and-Sum** (`bartlett_spectrum`) — steered power; simple and
  robust, but a wide main lobe (lower resolution).
- **MVDR / Capon** (`mvdr_spectrum`) — minimises interference while holding unit
  gain on the look direction; sharper.
- **MUSIC** (`music_spectrum`) — splits the covariance into "signal" and "noise"
  subspaces and exploits their orthogonality for very sharp peaks; needs to know
  roughly how many sources there are (here, one).
- **GCC-PHAT** (`gcc_phat`) — estimates the precise time delay between a pair of
  microphones, robust to reverberation.
- **SRP-PHAT** (`srp_phat`) — sums GCC-PHAT across all microphone pairs into a
  robust azimuth power map; closest in spirit to how networked multi-sensor
  systems triangulate.

`estimate_doa` simply reads off the peak of any of these spectra.

## 5.3 Results — accuracy vs SNR

On the simulated array, with a known true bearing, the **root-mean-square error**
of the estimate behaves exactly as theory predicts — excellent at high SNR,
degrading gracefully as noise rises, with the adaptive/subspace methods (MVDR,
MUSIC) beating plain delay-and-sum when the signal is weak:

| Array SNR | Bartlett | MVDR | MUSIC |
|---|---|---|---|
| −10 dB | ~4° | ~1.6° | ~2.3° |
| 0 dB | ~0.6° | ~0.8° | ~0.9° |
| +10 dB | ~0.3° | ~0.3° | ~0.3° |

The companion figure `results/doa_spectra.png` shows all four methods peaking at
the same true bearing, with MUSIC the sharpest and Bartlett the broadest — a
textbook illustration of the resolution trade-off.

## 5.4 Tracking a moving drone

A single-frame bearing is jittery. `track.AzimuthTracker` is a small
Kalman/alpha-beta smoother: it predicts where the bearing should be next and
blends that with the new noisy measurement. In `examples/07_doa_tracking_demo.py`
a source moves across azimuth over ~3 seconds; tracking cuts the bearing error
from **~5.7° (raw MUSIC)** to **~1.9° (tracked)** — see `results/doa_tracking.png`.

## 5.5 Real-array data (DREGON) — ready, pending access

`data_io.load_dregon_flight` and `examples/05_doa_real.py` are written to run the
same MUSIC / SRP-PHAT pipeline on the public **DREGON** 8-microphone dataset and
report error against its ground-truth angles. The script exits cleanly with
guidance if the data isn't present. DREGON's array is mounted *on* a UAV (so its
own rotor noise is present), which is a useful stress test once the data is in
place. This is the main pending item for the project.

➡ Next: [Evaluation, rigor & the leakage story](06_evaluation_rigor_and_leakage.md).
