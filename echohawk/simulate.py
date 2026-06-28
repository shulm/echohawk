"""Synthetic signal and microphone-array simulation.

Everything here is deliberately simple and physically transparent so the
algorithms in the rest of the package can be exercised without any external
data. A drone is modelled as a *harmonic stack* (a fundamental blade-passing
frequency plus integer harmonics) embedded in noise. The realistic hard
negative is a ground vehicle / generator -- also a low-frequency harmonic
source, differing from a drone mainly in spectral envelope -- so the detection
task is non-trivial. The array model is far-field (plane-wave) with
per-microphone propagation delays.

Azimuth convention: theta from the +x axis (radians); the unit vector toward
the source is u = (cos theta, sin theta).
"""
from __future__ import annotations

import numpy as np

C_AIR = 343.0  # speed of sound, m/s


def _harmonic_stack(fs, dur, f0, n_harmonics, decay, fm_hz, fm_depth,
                    broadband, rng):
    t = np.arange(int(fs * dur)) / fs
    inst = f0 * (1.0 + fm_depth * np.sin(2 * np.pi * fm_hz * t))
    phase = 2 * np.pi * np.cumsum(inst) / fs
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        sig += (decay ** (k - 1)) * np.sin(k * phase + rng.uniform(0, 2*np.pi))
    sig += broadband * rng.standard_normal(t.shape)
    return sig


def drone_signal(fs=16000, dur=1.0, bpf=120.0, n_harmonics=8,
                 harmonic_decay=0.7, fm_hz=3.0, fm_depth=0.02,
                 snr_db=10.0, seed=None):
    """A drone-like harmonic stack at blade-passing frequency ``bpf``.

    Rich high-order harmonics (slow decay) and a faster RPM wobble are the
    subtle cues that separate it from a ground vehicle.
    """
    rng = np.random.default_rng(seed)
    sig = _harmonic_stack(fs, dur, bpf, n_harmonics, harmonic_decay,
                          fm_hz, fm_depth, 0.15, rng)
    return _set_snr(sig, snr_db, rng)


def vehicle_signal(fs=16000, dur=1.0, f0=90.0, n_harmonics=5,
                   harmonic_decay=0.5, snr_db=10.0, seed=None):
    """Ground vehicle / generator: low-freq harmonic source (hard negative)."""
    rng = np.random.default_rng(seed)
    sig = _harmonic_stack(fs, dur, f0, n_harmonics, harmonic_decay,
                          fm_hz=rng.uniform(0.3, 2.0), fm_depth=0.01,
                          broadband=0.25, rng=rng)
    return _set_snr(sig, snr_db, rng)


def background_signal(fs=16000, dur=1.0, color="pink", snr_db=10.0, seed=None):
    """Coloured background noise (easy negative)."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    white = rng.standard_normal(n)
    if color == "white":
        sig = white
    else:
        X = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, 1 / fs); f[0] = f[1]
        sig = np.fft.irfft(X / np.sqrt(f), n=n)
    return _set_snr(sig, snr_db, rng)


def confuser_signal(fs=16000, dur=1.0, snr_db=10.0, seed=None):
    """Non-harmonic tonal distractor (siren chirp / bird warble / machinery)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    sig = np.zeros_like(t)
    kind = rng.integers(0, 3)
    if kind == 0:
        for _ in range(rng.integers(1, 4)):
            sig += rng.uniform(0.4, 1.0) * np.sin(2*np.pi*rng.uniform(150,3000)*t)
    elif kind == 1:
        f0, f1 = rng.uniform(200, 800), rng.uniform(800, 3000)
        sig += np.sin(2*np.pi*(f0*t + 0.5*(f1-f0)/dur*t**2))
    else:
        fc = rng.uniform(1000, 4000)
        sig += np.sin(2*np.pi*fc*t + 6*np.sin(2*np.pi*rng.uniform(4,9)*t))
    sig += 0.3 * rng.standard_normal(t.shape)
    return _set_snr(sig, snr_db, rng)


def _set_snr(sig, snr_db, rng):
    sig = sig / (np.std(sig) + 1e-12)
    noise_power = 10 ** (-snr_db / 10.0)
    return (sig + np.sqrt(noise_power) * rng.standard_normal(sig.shape)).astype(np.float64)


def make_detection_dataset(n_per_class=200, fs=16000, dur=1.0,
                           snr_range=(-15, 3), seed=0):
    """Balanced drone/non-drone dataset with realistic class overlap.

    Negatives are dominated by ground-vehicle harmonic sources whose
    parameters overlap the drone's, so the classes are genuinely confusable.

    Returns X (list of waveforms), y (1=drone, 0=non-drone), fs.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_per_class):
        snr = rng.uniform(*snr_range)
        X.append(drone_signal(fs, dur, bpf=rng.uniform(90, 190),
                              n_harmonics=int(rng.integers(5, 11)),
                              harmonic_decay=rng.uniform(0.6, 0.8),
                              fm_hz=rng.uniform(2, 6), fm_depth=0.02,
                              snr_db=snr, seed=int(rng.integers(1e9))))
        y.append(1)
        snr = rng.uniform(*snr_range); r = rng.random()
        if r < 0.55:                       # hard: vehicle/generator
            neg = vehicle_signal(fs, dur, f0=rng.uniform(70, 170),
                                 n_harmonics=int(rng.integers(3, 8)),
                                 harmonic_decay=rng.uniform(0.45, 0.7),
                                 snr_db=snr, seed=int(rng.integers(1e9)))
        elif r < 0.8:                      # tonal distractors
            neg = confuser_signal(fs, dur, snr_db=snr, seed=int(rng.integers(1e9)))
        else:                              # noise
            col = "pink" if rng.random() < 0.5 else "white"
            neg = background_signal(fs, dur, color=col, snr_db=snr,
                                    seed=int(rng.integers(1e9)))
        X.append(neg); y.append(0)
    return X, np.asarray(y), fs


def uca_positions(n_mics=8, radius=0.10):
    """Uniform circular array in z=0 plane. Returns (n_mics, 2) metres."""
    ang = 2 * np.pi * np.arange(n_mics) / n_mics
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def simulate_array(signal, doa_deg, positions, fs=16000, c=C_AIR,
                   snr_db=10.0, seed=None):
    """Propagate one far-field source across the array. Returns (n_mics, N)."""
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(doa_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    tau = -(positions @ u) / c
    n = signal.shape[0]
    S = np.fft.rfft(signal); f = np.fft.rfftfreq(n, 1 / fs)
    X = np.empty((positions.shape[0], n))
    for m, tm in enumerate(tau):
        xm = np.fft.irfft(S * np.exp(-1j*2*np.pi*f*tm), n=n)
        p = np.mean(xm**2); npow = p * 10 ** (-snr_db/10.0)
        X[m] = xm + np.sqrt(npow) * rng.standard_normal(n)
    return X
