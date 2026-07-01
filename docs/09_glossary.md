# 9. Glossary

Plain-language definitions of every term used in these docs.

**Acoustic** — relating to sound. EchoHawk is an *acoustic* sensor: it works from
sound alone.

**Array (microphone array)** — several microphones in a known geometry, used
together to work out the direction a sound came from.

**AUC (Area Under the ROC Curve)** — a single score for a detector, from 0.5
(useless, coin-flip) to 1.0 (perfect).

**Azimuth** — a compass-like horizontal angle (the bearing) to the source.

**Bartlett beamformer** — see *Delay-and-sum*. The simplest beamformer.

**Beamforming** — combining microphone signals so the array listens most
sensitively in a chosen direction.

**Blade-Passing Frequency (BPF)** — the rate at which propeller blades slice the
air; the fundamental tone of a drone's sound. `BPF = blades × revolutions-per-second`.

**CNN (Convolutional Neural Network)** — a neural network that learns patterns
from images; here it reads spectrogram "pictures".

**Confuser** — a non-drone sound that resembles a drone (e.g. a ground vehicle's
low-frequency harmonics); the hard case for detection.

**Covariance matrix** — a table summarising how the microphone signals vary
together; the mathematical raw material for MVDR and MUSIC.

**Decibel (dB)** — a logarithmic ratio used for loudness and signal-to-noise.

**Delay-and-sum** — line up the microphone signals for a chosen direction and add
them; energy peaks at the true direction. Robust but low resolution.

**Direction of Arrival (DOA)** — the bearing from the array to the sound source.

**False alarm (Pfa)** — wrongly flagging a non-drone as a drone. Security systems
demand a *low* false-alarm rate.

**Far-field** — the source is far enough away that its sound arrives as a flat
(plane) wave; simplifies the array maths.

**Fourier transform / FFT** — the maths that converts a signal into "how much of
each frequency is present".

**GCC-PHAT** — a robust way to measure the precise time delay between two
microphones.

**Grouped split (GroupShuffleSplit)** — a train/test split that keeps all clips
from one recording on the same side, preventing *leakage*.

**Harmonic** — a tone at an exact integer multiple of a fundamental frequency.

**Harmonic Product Spectrum (HPS)** — a method that finds a fundamental frequency
by multiplying frequency-squashed copies of the spectrum; used to estimate the BPF.

**Harmonic stack** — a fundamental tone plus its harmonics; the shape of a
drone's (or voice's) sound.

**Leakage (data leakage)** — when test information sneaks into training, making a
model look better than it is. EchoHawk's key cautionary tale (see chapter 6).

**Log-mel spectrogram** — a spectrogram with a hearing-matched frequency scale
and log brightness; the CNN's input.

**MFCC (Mel-Frequency Cepstral Coefficients)** — a compact numerical summary of a
spectrogram, common in audio classification.

**MUSIC** — a high-resolution direction-finding method using the signal's
mathematical "subspaces"; produces very sharp angle estimates.

**MVDR / Capon** — an adaptive beamformer that suppresses other directions while
preserving the chosen one; sharper than delay-and-sum.

**Pd (Probability of Detection)** — the fraction of real drones that are caught.

**Pd@1%FA** — detection rate when the false-alarm rate is held at 1%; the
operationally important number.

**Random Forest** — a classifier that averages many decision trees; EchoHawk's
simple, transparent baseline.

**ROC curve** — a plot of detection rate vs false-alarm rate across all decision
thresholds.

**SNR (Signal-to-Noise Ratio)** — how strong the signal is versus background
noise, in dB; the master dial for difficulty.

**Spectrogram** — a picture of sound: time across, frequency up, brightness =
loudness.

**SRP-PHAT** — combines GCC-PHAT across all microphone pairs into a robust map of
energy vs direction.

**Steering vector** — the array's expected response to a wave from a particular
direction; the building block of beamforming.

**Tracking** — smoothing a sequence of noisy direction estimates into a clean
trajectory, using the fact that drones move predictably.

**UAS / UAV** — Unmanned Aerial System / Vehicle; a drone. *Counter-UAS* =
defending against drones.

**Wideband** — using a broad range of frequencies (rather than a single tone);
EchoHawk's direction-finders combine many frequency bins.
