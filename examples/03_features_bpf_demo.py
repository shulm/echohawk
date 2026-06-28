"""Track A (features) -- spectrogram + blade-passing-frequency estimation.

Shows the log-mel spectrogram of a synthetic drone clip and the
harmonic-product-spectrum used to recover its blade-passing frequency.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echohawk import simulate, features  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)

FS = 16000
TRUE_BPF = 110.0


def main():
    x = simulate.drone_signal(FS, dur=1.5, bpf=TRUE_BPF, n_harmonics=9,
                              snr_db=8, seed=3)
    logmel = features.log_mel_spectrogram(x, FS)
    f0, freqs, hps = features.harmonic_product_spectrum(x, FS)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    a1.imshow(logmel, origin="lower", aspect="auto", cmap="magma")
    a1.set_title("Log-mel spectrogram (drone)")
    a1.set_xlabel("Frame")
    a1.set_ylabel("Mel band")

    hi = np.searchsorted(freqs, 700)
    a2.plot(freqs[:hi], hps[:hi] / hps[:hi].max(), lw=1.2)
    a2.axvline(TRUE_BPF, color="g", ls="--", label=f"true BPF {TRUE_BPF:.0f} Hz")
    a2.axvline(f0, color="r", ls=":", label=f"estimated {f0:.0f} Hz")
    a2.set_title("Harmonic product spectrum")
    a2.set_xlabel("Frequency (Hz)")
    a2.set_ylabel("Normalised HPS")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "features_bpf.png"), dpi=130)
    plt.close(fig)
    print(f"true BPF={TRUE_BPF:.0f} Hz, estimated={f0:.0f} Hz")
    print("  -> results/features_bpf.png")


if __name__ == "__main__":
    main()
