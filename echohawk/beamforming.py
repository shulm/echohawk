"""Microphone-array direction-of-arrival (DOA) and localisation.

Wideband, far-field, azimuth-only estimators on a planar array:

    * delay-and-sum (Bartlett) spectrum
    * MVDR / Capon spectrum
    * MUSIC pseudospectrum
    * GCC-PHAT pairwise time-delay estimation
    * SRP-PHAT azimuth power map

The wideband spectra are formed by averaging narrowband spatial spectra over a
band of STFT frequency bins (incoherent averaging). Azimuth convention matches
``simulate``: theta from +x, source direction u = (cos theta, sin theta).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

C_AIR = 343.0


# --------------------------------------------------------------------------- #
# Steering vectors and spatial covariance
# --------------------------------------------------------------------------- #
def steering_vector(positions, theta_rad, freq, c=C_AIR):
    """Array manifold vector a(theta, f) for a planar array, shape (M,)."""
    u = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    delays = (positions @ u) / c           # matches simulate_array sign
    return np.exp(1j * 2 * np.pi * freq * delays)


def steering_matrix(positions, az_rad, freq, c=C_AIR):
    """Steering vectors for many azimuths at once. Shape (M, n_az)."""
    U = np.stack([np.cos(az_rad), np.sin(az_rad)], axis=0)   # (2, G)
    delays = (positions @ U) / c                             # (M, G)
    return np.exp(1j * 2 * np.pi * freq * delays)


def _stft_multi(X, fs, n_fft=512, hop=256):
    """STFT each channel. Returns freqs and Z of shape (M, F, T)."""
    Zs = []
    for ch in X:
        f, _, Z = sps.stft(ch, fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                           window="hann", boundary=None, padded=False)
        Zs.append(Z)
    return f, np.stack(Zs, axis=0)


def covariances(Z, fbin_idx):
    """Per-bin spatial covariance matrices estimated over time frames.

    Z : (M, F, T) complex STFT. Returns dict {bin: (M, M)}.
    """
    M = Z.shape[0]
    Rs = {}
    for k in fbin_idx:
        Xk = Z[:, k, :]                    # (M, T)
        T = Xk.shape[1]
        R = (Xk @ Xk.conj().T) / max(T, 1)
        R += 1e-6 * np.trace(R) / M * np.eye(M)   # diagonal loading
        Rs[k] = R
    return Rs


def _band_bins(freqs, fmin, fmax):
    return np.where((freqs >= fmin) & (freqs <= fmax))[0]


# --------------------------------------------------------------------------- #
# Wideband azimuth spectra
# --------------------------------------------------------------------------- #
def _azimuth_spectrum(X, fs, positions, kind, az_grid_deg,
                      fmin, fmax, c, n_sources, n_fft, hop):
    """Wideband azimuth spectrum, vectorised over the azimuth grid per bin."""
    freqs, Z = _stft_multi(X, fs, n_fft, hop)
    fbins = _band_bins(freqs, fmin, fmax)
    if len(fbins) == 0:
        raise ValueError("No STFT bins in the requested band.")
    Rs = covariances(Z, fbins)
    M = Z.shape[0]
    az = np.deg2rad(az_grid_deg)
    P = np.zeros(len(az))

    for k in fbins:
        A = steering_matrix(positions, az, freqs[k], c)        # (M, G)
        R = Rs[k]
        if kind == "bartlett":
            # a^H R a for every column, vectorised
            P += np.real(np.einsum("mg,mn,ng->g", A.conj(), R, A)) / M
        elif kind == "mvdr":
            Rinv = np.linalg.inv(R)
            denom = np.real(np.einsum("mg,mn,ng->g", A.conj(), Rinv, A))
            P += 1.0 / np.maximum(denom, 1e-12)
        elif kind == "music":
            w, V = np.linalg.eigh(R)                            # ascending
            En = V[:, : M - n_sources]                         # noise subspace
            proj = En.conj().T @ A                             # (noise, G)
            denom = np.sum(np.abs(proj) ** 2, axis=0)
            P += 1.0 / (denom + 1e-12)
        else:
            raise ValueError(kind)

    P /= len(fbins)
    P = P / P.max()
    return az_grid_deg, P


def bartlett_spectrum(X, fs, positions, az_grid_deg=np.arange(0, 360, 1.0),
                      fmin=80.0, fmax=4000.0, c=C_AIR, n_fft=512, hop=256):
    return _azimuth_spectrum(X, fs, positions, "bartlett", az_grid_deg,
                             fmin, fmax, c, 1, n_fft, hop)


def mvdr_spectrum(X, fs, positions, az_grid_deg=np.arange(0, 360, 1.0),
                  fmin=80.0, fmax=4000.0, c=C_AIR, n_fft=512, hop=256):
    return _azimuth_spectrum(X, fs, positions, "mvdr", az_grid_deg,
                             fmin, fmax, c, 1, n_fft, hop)


def music_spectrum(X, fs, positions, n_sources=1,
                   az_grid_deg=np.arange(0, 360, 1.0),
                   fmin=80.0, fmax=4000.0, c=C_AIR, n_fft=512, hop=256):
    return _azimuth_spectrum(X, fs, positions, "music", az_grid_deg,
                             fmin, fmax, c, n_sources, n_fft, hop)


def estimate_doa(az_grid_deg, P):
    """Peak of a spectrum -> estimated azimuth in degrees."""
    return float(az_grid_deg[int(np.argmax(P))])


# --------------------------------------------------------------------------- #
# Time-delay methods
# --------------------------------------------------------------------------- #
def gcc_phat(x1, x2, fs, max_tau=None, interp=16):
    """Generalised cross-correlation with phase transform.

    Returns ``(tau, cc, lags)``: estimated delay of x1 relative to x2 (s),
    the correlation function and its lag axis (s).
    """
    n = x1.shape[0] + x2.shape[0]
    X1 = np.fft.rfft(x1, n=n)
    X2 = np.fft.rfft(x2, n=n)
    R = X1 * np.conj(X2)
    R /= np.abs(R) + 1e-12                       # PHAT weighting
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    lags = np.arange(-max_shift, max_shift + 1) / float(interp * fs)
    tau = lags[int(np.argmax(np.abs(cc)))]
    return tau, cc, lags


def srp_phat(X, fs, positions, az_grid_deg=np.arange(0, 360, 1.0), c=C_AIR):
    """SRP-PHAT azimuth map: sum pairwise GCC-PHAT at the steered delays."""
    M = X.shape[0]
    az = np.deg2rad(az_grid_deg)
    # pre-compute each pair's GCC-PHAT curve once
    pairs, ccs, lagaxes = [], [], []
    for i in range(M):
        for j in range(i + 1, M):
            _, cc, lags = gcc_phat(X[i], X[j], fs)
            pairs.append((i, j))
            ccs.append(cc)
            lagaxes.append(lags)
    P = np.zeros(len(az))
    for gi, th in enumerate(az):
        u = np.array([np.cos(th), np.sin(th)])
        s = 0.0
        for (i, j), cc, lags in zip(pairs, ccs, lagaxes):
            tau_ij = -((positions[i] - positions[j]) @ u) / c
            s += np.interp(tau_ij, lags, cc)
        P[gi] = s
    P = (P - P.min()) / (np.ptp(P) + 1e-12)
    return az_grid_deg, P
