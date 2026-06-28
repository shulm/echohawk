"""EchoHawk - passive acoustic drone detection, classification and direction-finding.

A compact, readable reference implementation of the core algorithms behind
acoustic counter-UAS systems:

    * drone vs. background detection / classification from acoustic signatures
    * blade-passing-frequency (BPF) / harmonic analysis
    * microphone-array direction-of-arrival (DOA): delay-and-sum, MVDR, MUSIC
    * time-delay localisation: GCC-PHAT and SRP-PHAT

The package ships with a synthetic data simulator so every algorithm is
runnable end-to-end without downloading any dataset. See ``examples/`` for
scripts that reproduce the figures in ``results/``.
"""

__version__ = "0.1.0"

from . import simulate, features, beamforming, evaluate  # noqa: F401
