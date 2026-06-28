"""DOA Tracking Demo.

Simulates a drone moving across azimuth over ~3 seconds.
Computes noisy MUSIC bearings per frame, smooths with AzimuthTracker,
and plots the results.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from echohawk import simulate, beamforming, track

RESULTS = os.path.join(_REPO, "results")

def main():
    os.makedirs(RESULTS, exist_ok=True)
    
    fs = 16000
    total_dur = 3.0
    snapshot_dur = 0.2
    n_snapshots = int(total_dur / snapshot_dur)
    
    # Simulate a drone moving from 30° to 60°
    true_azimuths = np.linspace(30.0, 60.0, n_snapshots)
    
    # 8-mic array
    positions = simulate.uca_positions(8, radius=0.1)
    
    tracker = track.AzimuthTracker(alpha=0.3, beta=0.05, dt=snapshot_dur)
    
    time_pts = []
    raw_estimates = []
    tracked_estimates = []
    
    angles = np.arange(0, 360, 1.0)
    rng = np.random.default_rng(42)
    
    for i in range(n_snapshots):
        az = true_azimuths[i]
        
        # Generate a small drone snippet
        signal = simulate.drone_signal(fs=fs, dur=snapshot_dur, seed=rng.integers(1e9))
        
        # Simulate array with heavy noise to induce large raw errors (SNR=-8dB)
        X = simulate.simulate_array(signal, doa_deg=az, positions=positions, fs=fs, snr_db=-8.0, seed=rng.integers(1e9))
        
        # MUSIC
        spec = beamforming.music_spectrum(X, fs, positions)[1]
        meas_az = angles[np.argmax(spec)]
        
        # Update tracker
        est_az = tracker.update(meas_az)
        
        time_pts.append(i * snapshot_dur)
        raw_estimates.append(meas_az)
        tracked_estimates.append(est_az)
        
    true_azimuths = np.array(true_azimuths)
    raw_estimates = np.array(raw_estimates)
    tracked_estimates = np.array(tracked_estimates)
    
    raw_rmse = np.sqrt(np.mean((raw_estimates - true_azimuths)**2))
    tracked_rmse = np.sqrt(np.mean((tracked_estimates - true_azimuths)**2))
    
    print(f"Raw MUSIC RMSE: {raw_rmse:.2f}°")
    print(f"Tracked RMSE  : {tracked_rmse:.2f}°")
    
    plt.figure(figsize=(8, 4))
    plt.plot(time_pts, true_azimuths, 'k--', label="True Azimuth")
    plt.plot(time_pts, raw_estimates, 'ro', label=f"Raw MUSIC (RMSE={raw_rmse:.1f}°)")
    plt.plot(time_pts, tracked_estimates, 'b-', linewidth=2, label=f"Tracked (RMSE={tracked_rmse:.1f}°)")
    plt.xlabel("Time (s)")
    plt.ylabel("Azimuth (degrees)")
    plt.title("DOA Tracking over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    out_path = os.path.join(RESULTS, "doa_tracking.png")
    plt.savefig(out_path)
    plt.close()
    
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
