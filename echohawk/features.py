"""Acoustic feature extraction (NumPy/SciPy only -- no librosa dependency).

Provides:
    * STFT magnitude spectrogram
    * mel filterbank + log-mel spectrogram (CNN-style input)
    * MFCCs (classical-classifier input)
    * harmonic-product-spectrum and blade-passing-frequency estimation
    * a compact summary feature vector for the baseline classifier
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.fft import dct


# --------------------------------------------------------------------------- #
# Spectro-temporal representations
# --------------------------------------------------------------------------- #
def stft_mag(x, fs, n_fft=512, hop=256):
    f, t, Z = sps.stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                       window="hann", boundary=None, padded=False)
    return f, t, np.abs(Z)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(fs, n_fft=512, n_mels=40, fmin=50.0, fmax=None):
    fmax = fmax or fs / 2
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, fs / 2, n_bins)
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = min(l + 1, n_bins - 1)
        if r == c:
            r = min(c + 1, n_bins - 1)
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


def log_mel_spectrogram(x, fs, n_fft=512, hop=256, n_mels=40):
    _, _, mag = stft_mag(x, fs, n_fft, hop)
    fb = mel_filterbank(fs, n_fft, n_mels)
    mel = fb @ (mag ** 2)
    return np.log(mel + 1e-10)


def mfcc(x, fs, n_fft=512, hop=256, n_mels=40, n_mfcc=13):
    logmel = log_mel_spectrogram(x, fs, n_fft, hop, n_mels)
    coeffs = dct(logmel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return coeffs


# --------------------------------------------------------------------------- #
# Harmonic / blade-passing-frequency analysis
# --------------------------------------------------------------------------- #
def harmonic_product_spectrum(x, fs, n_fft=8192, n_harmonics=5,
                              fmin=40.0, fmax=400.0):
    """Estimate the fundamental (blade-passing) frequency via HPS.

    Returns ``(f0_hz, freqs, hps)`` where ``hps`` is the harmonic product
    spectrum over ``freqs``.
    """
    win = np.hanning(min(n_fft, len(x)))
    seg = x[:len(win)] * win
    spec = np.abs(np.fft.rfft(seg, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    hps = spec.copy()
    for h in range(2, n_harmonics + 1):
        dec = spec[::h]
        hps[:len(dec)] *= dec
    lo = np.searchsorted(freqs, fmin)
    hi = np.searchsorted(freqs, fmax)
    f0 = freqs[lo + np.argmax(hps[lo:hi])]
    return f0, freqs, hps


# --------------------------------------------------------------------------- #
# Summary feature vector for the classical baseline
# --------------------------------------------------------------------------- #
def feature_vector(x, fs):
    """Fixed-length feature vector: MFCC stats + spectral + harmonic cues."""
    m = mfcc(x, fs)
    feats = [m.mean(axis=1), m.std(axis=1)]

    f, _, mag = stft_mag(x, fs)
    psd = (mag ** 2).mean(axis=1) + 1e-12
    psd_n = psd / psd.sum()
    centroid = float((f * psd_n).sum())
    spread = float(np.sqrt(((f - centroid) ** 2 * psd_n).sum()))
    flatness = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))  # tonal->0
    f0, _, _ = harmonic_product_spectrum(x, fs)

    extra = np.array([centroid, spread, flatness, f0], dtype=np.float64)
    return np.concatenate([*feats, extra])
