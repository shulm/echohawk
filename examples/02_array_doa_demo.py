"""Track B -- microphone-array direction-of-arrival on synthetic data.

Simulates an 8-mic uniform circular array receiving a drone harmonic stack
from a known azimuth, then compares Bartlett / MVDR / MUSIC / SRP-PHAT and
plots a DOA-error-vs-SNR curve. Figures go to ``results/``.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echohawk import simulate, beamforming as bf  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)

FS = 16000
POS = simulate.uca_positions(n_mics=8, radius=0.10)
GRID = np.arange(0, 360, 1.0)


def spectra_figure(true_doa=65.0, snr_db=5.0):
    sig = simulate.drone_signal(FS, dur=1.0, bpf=120, n_harmonics=8, snr_db=20)
    X = simulate.simulate_array(sig, true_doa, POS, FS, snr_db=snr_db, seed=1)

    _, Pb = bf.bartlett_spectrum(X, FS, POS, GRID)
    _, Pm = bf.mvdr_spectrum(X, FS, POS, GRID)
    _, Pu = bf.music_spectrum(X, FS, POS, n_sources=1, az_grid_deg=GRID)
    _, Ps = bf.srp_phat(X, FS, POS, GRID)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for P, name in [(Pb, "Bartlett"), (Pm, "MVDR"),
                    (Pu, "MUSIC"), (Ps, "SRP-PHAT")]:
        ax.plot(GRID, P, label=f"{name} (est {bf.estimate_doa(GRID, P):.0f}°)")
    ax.axvline(true_doa, color="k", ls="--", lw=1, label=f"true {true_doa:.0f}°")
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Normalised power")
    ax.set_title(f"DOA spectra, 8-mic UCA, SNR={snr_db:.0f} dB")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "doa_spectra.png"), dpi=130)
    plt.close(fig)
    print("  -> results/doa_spectra.png")


def error_vs_snr(snrs=(-10, -5, 0, 5, 10, 15, 20), trials=12):
    methods = {"Bartlett": bf.bartlett_spectrum,
               "MVDR": bf.mvdr_spectrum,
               "MUSIC": lambda X, fs, pos, grid: bf.music_spectrum(
                   X, fs, pos, n_sources=1, az_grid_deg=grid)}
    rmse = {m: [] for m in methods}
    rng = np.random.default_rng(0)
    for snr in snrs:
        errs = {m: [] for m in methods}
        for _ in range(trials):
            true = float(rng.uniform(0, 360))
            sig = simulate.drone_signal(FS, 1.0, bpf=120, snr_db=20,
                                        seed=int(rng.integers(1e9)))
            X = simulate.simulate_array(sig, true, POS, FS, snr_db=snr,
                                        seed=int(rng.integers(1e9)))
            for m, fn in methods.items():
                _, P = fn(X, FS, POS, GRID)
                est = bf.estimate_doa(GRID, P)
                d = abs((est - true + 180) % 360 - 180)
                errs[m].append(d)
        for m in methods:
            rmse[m].append(np.sqrt(np.mean(np.square(errs[m]))))
        print(f"  SNR {snr:+3d} dB  " +
              "  ".join(f"{m}={rmse[m][-1]:5.1f}°" for m in methods))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m in methods:
        ax.plot(snrs, rmse[m], marker="o", label=m)
    ax.set_xlabel("Array SNR (dB)")
    ax.set_ylabel("DOA RMSE (degrees)")
    ax.set_yscale("log")
    ax.set_title(f"DOA accuracy vs SNR ({trials} trials/point)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "doa_error_vs_snr.png"), dpi=130)
    plt.close(fig)
    print("  -> results/doa_error_vs_snr.png")


def main():
    print("DOA spectra figure ...")
    spectra_figure()
    print("DOA error vs SNR ...")
    error_vs_snr()


if __name__ == "__main__":
    main()
