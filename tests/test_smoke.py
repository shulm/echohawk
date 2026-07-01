import os
import sys
import tempfile
import numpy as np

import pytest
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from echohawk import simulate, features, models, evaluate, beamforming
from echohawk.data_io import load_dregon_flight

def test_detection_synthetic():
    # Generate a small synthetic dataset
    X_list, y, fs = simulate.make_detection_dataset(n_per_class=50, seed=42)
    
    # Feature extraction
    F = models.features_matrix(X_list, fs, features.feature_vector)
    
    # Train test split (80/20)
    from sklearn.model_selection import train_test_split
    F_tr, F_te, y_tr, y_te = train_test_split(F, y, test_size=0.2, random_state=42)
    
    # Train RF
    clf = models.build_baseline("rf", seed=42)
    clf.fit(F_tr, y_tr)
    
    scores = clf.predict_proba(F_te)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_te, scores)
    
    assert auc > 0.80, f"Expected AUC > 0.8, got {auc}"

def test_doa_music():
    fs = 16000
    azimuth = 45.0
    signal = simulate.drone_signal(fs=fs, dur=0.5, seed=42)
    positions = simulate.uca_positions(8, radius=0.1)
    
    X = simulate.simulate_array(signal, azimuth, positions, fs=fs, snr_db=10.0, seed=42)
    
    spec = beamforming.music_spectrum(X, fs, positions)[1]
    angles = np.arange(0, 360, 1.0)
    est_az = angles[np.argmax(spec)]
    
    err = min(abs(est_az - azimuth), 360 - abs(est_az - azimuth))
    assert err <= 5.0, f"Expected error <= 5°, got {err}°"

def test_bpf():
    fs = 16000
    true_bpf = 120.0
    signal = simulate.drone_signal(fs=fs, dur=1.0, bpf=true_bpf, seed=42)
    
    est_f0, _, _ = features.harmonic_product_spectrum(signal, fs=fs, fmin=50.0, fmax=400.0)
    
    err = abs(est_f0 - true_bpf)
    assert err <= 5.0, f"Expected BPF error <= 5Hz, got {err}Hz"

def test_load_dregon_flight():
    sf = pytest.importorskip("soundfile")
    # Create dummy 8-channel WAV
    fs = 16000
    dur = 0.1
    # 8-channel random noise
    data = np.random.randn(int(fs * dur), 8).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    
    sf.write(path, data, fs)
    
    try:
        X, pos, fs_out, az = load_dregon_flight(path)
        assert X.shape == (8, int(fs * dur))
        assert pos.shape == (8, 2)
        assert fs_out == fs
        assert az is None
    finally:
        os.remove(path)
