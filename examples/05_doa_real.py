"""Real-data DOA using DREGON 8-channel array.

Workflow
--------
1. Load 8-channel DREGON flight recording.
2. If data is absent, prints a guidance message and exits.
3. Compute MUSIC spectrum and SRP-PHAT for the recording.
4. Plot the azimuth spectrum and overlay ground truth if available.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from echohawk import beamforming
from echohawk.data_io import load_dregon_flight

RESULTS = os.path.join(_REPO, "results")
DREGON_DIR = os.path.join(_REPO, "data", "dregon")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    
    # Check for DREGON data
    wav_files = []
    if os.path.isdir(DREGON_DIR):
        wav_files = [os.path.join(DREGON_DIR, f) for f in os.listdir(DREGON_DIR) if f.endswith(".wav")]
        
    if not wav_files:
        print(f"DREGON data not found in {DREGON_DIR}.")
        print("Please place a DREGON flight (e.g. flight1.wav) in this directory.")
        print("You can download it from http://dregon.inria.fr")
        sys.exit(0)
        
    # Use the first available flight
    flight_path = wav_files[0]
    print(f"Loading {flight_path} ...")
    
    try:
        X, positions, fs, gt_azimuth = load_dregon_flight(flight_path)
    except Exception as e:
        print(f"Failed to load DREGON flight: {e}")
        sys.exit(1)
        
    print(f"Data: {X.shape[0]} channels, {X.shape[1]} samples, fs={fs}Hz")
    
    # Process a 1-second snippet from the middle for a clean snapshot
    mid = X.shape[1] // 2
    n_samples = fs * 1  # 1 second
    start = max(0, mid - n_samples // 2)
    end = min(X.shape[1], start + n_samples)
    X_snip = X[:, start:end]
    
    print("Computing MUSIC spectrum ...")
    angles = np.arange(0, 360, 1.0)
    music_spec = beamforming.music_spectrum(X_snip, fs, positions)[1]
    srp_spec = beamforming.srp_phat(X_snip, fs, positions)[1]
    
    est_az_music = angles[np.argmax(music_spec)]
    est_az_srp = angles[np.argmax(srp_spec)]
    
    print(f"MUSIC peak : {est_az_music:.1f}°")
    print(f"SRP-PHAT peak: {est_az_srp:.1f}°")
    
    if gt_azimuth is not None:
        print(f"Ground Truth: {gt_azimuth:.1f}°")
        print(f"MUSIC Error: {abs(est_az_music - gt_azimuth):.1f}°")
        
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(angles, music_spec / np.max(music_spec), label="MUSIC", color="blue")
    plt.plot(angles, srp_spec / np.max(srp_spec), label="SRP-PHAT", color="orange")
    if gt_azimuth is not None:
        plt.axvline(gt_azimuth, color='r', linestyle='--', label=f"Truth ({gt_azimuth:.1f}°)")
    
    plt.title("DREGON Flight DOA Azimuth Spectrum")
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Normalized Spatial Power")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS, "doa_real_dregon.png")
    plt.savefig(out_path)
    plt.close()
    
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
