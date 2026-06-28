"""Data I/O helpers for real acoustic datasets.

Supports two access modes for DADS:
    * ``load_dads(root, ...)``  -- folder-of-WAVs layout
                                   (class 0/ and 1/ sub-directories).
    * ``load_dads_hf(...)``     -- HuggingFace datasets streaming
                                   (reads parquet shards from HF hub).

All load functions return a ``(X_list, y, groups, fs)`` tuple, where `groups`
is an integer array mapping clips to their original source recording to 
prevent train/test leakage.
"""
from __future__ import annotations

import io
import os
import pathlib
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — soundfile preferred; fall back to scipy.io.wavfile
# ---------------------------------------------------------------------------
try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:  # pragma: no cover
    _HAS_SF = False

try:
    from scipy.io import wavfile as _scipy_wav
    _HAS_SCIPY_WAV = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY_WAV = False

try:
    import scipy.signal as _sps
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Internal audio helpers
# ---------------------------------------------------------------------------

def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file from disk → (mono float32, sample_rate)."""
    if _HAS_SF:
        data, fs = sf.read(path, always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if data.dtype.kind in ("i", "u"):
            info = np.iinfo(data.dtype)
            data = data.astype(np.float32) / max(abs(info.min), info.max)
        else:
            data = data.astype(np.float32)
        return data, int(fs)

    if _HAS_SCIPY_WAV:
        fs, data = _scipy_wav.read(path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if data.dtype.kind in ("i", "u"):
            info = np.iinfo(data.dtype)
            data = data.astype(np.float32) / max(abs(info.min), info.max)
        else:
            data = data.astype(np.float32)
        return data, int(fs)

    raise ImportError(
        "Neither soundfile nor scipy is available. "
        "Install with: pip install soundfile"
    )


def _read_wav_bytes(raw: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes (from HF parquet) → (mono float32, sample_rate)."""
    if not _HAS_SF:
        raise ImportError("soundfile is required for in-memory WAV decoding. "
                          "pip install soundfile")
    data, fs = sf.read(io.BytesIO(raw), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype.kind in ("i", "u"):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / max(abs(info.min), info.max)
    else:
        data = data.astype(np.float32)
    return data, int(fs)


def _resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resample ``x`` from ``fs_in`` → ``fs_out``."""
    if fs_in == fs_out:
        return x
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for resampling.")
    from math import gcd
    g = gcd(fs_in, fs_out)
    up, down = fs_out // g, fs_in // g
    return _sps.resample_poly(x, up, down).astype(np.float32)


def _segment(x: np.ndarray, clip_len: int) -> List[np.ndarray]:
    """Split ``x`` into non-overlapping windows of ``clip_len`` samples.

    Short clips are zero-padded.  The final partial window (if any) is
    also zero-padded so all returned clips have exactly ``clip_len`` samples.
    """
    if len(x) == 0:
        return [np.zeros(clip_len, dtype=np.float32)]
    if len(x) < clip_len:
        pad = np.zeros(clip_len, dtype=np.float32)
        pad[: len(x)] = x
        return [pad]
    clips = []
    n_full = len(x) // clip_len
    for i in range(n_full):
        clips.append(x[i * clip_len : (i + 1) * clip_len].copy())
    remainder = len(x) % clip_len
    if remainder > 0:
        pad = np.zeros(clip_len, dtype=np.float32)
        pad[:remainder] = x[n_full * clip_len :]
        clips.append(pad)
    return clips


# ---------------------------------------------------------------------------
# Public API — Mode 1: folder-of-WAVs
# ---------------------------------------------------------------------------

def load_dads(
    root: str,
    fs: int = 16000,
    clip_s: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int]:
    """Load DADS from a local folder-of-WAVs layout.

    Returns
    -------
    X_list : list of np.ndarray, dtype float32, shape (fs*clip_s,)
    y      : np.ndarray of int64, shape (N,)  -- 0=no drone, 1=drone
    groups : np.ndarray of int64, shape (N,)  -- source recording ID
    fs     : int  -- target sample rate
    """
    root = pathlib.Path(root)
    clip_len = int(round(fs * clip_s))
    wav_extensions = {".wav", ".flac", ".ogg", ".mp3"}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups_list: List[int] = []
    group_id = 0

    for label in (0, 1):
        class_dir = root / str(label)
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}\n"
                f"  Run `bash scripts/download_data.sh` to populate '{root}'."
            )
        wav_files = sorted(
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in wav_extensions
        )
        if not wav_files:
            raise FileNotFoundError(
                f"No audio files found under {class_dir}. "
                "Check the download completed successfully."
            )
        for wav_path in wav_files:
            try:
                data, fs_orig = _read_wav(str(wav_path))
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] Skipping {wav_path.name}: {exc}")
                continue
            if fs_orig != fs:
                data = _resample(data, fs_orig, fs)
            clips = _segment(data, clip_len)
            X_list.extend(clips)
            y_list.extend([label] * len(clips))
            groups_list.extend([group_id] * len(clips))
            group_id += 1

    if not X_list:
        raise RuntimeError(
            f"No clips loaded from '{root}'. "
            "Verify dataset structure and audio file integrity."
        )
    return X_list, np.asarray(y_list, dtype=np.int64), np.asarray(groups_list, dtype=np.int64), fs


# ---------------------------------------------------------------------------
# Public API — Mode 2: HuggingFace datasets streaming
# ---------------------------------------------------------------------------

def load_dads_hf(
    repo_id: str = "geronimobasso/drone-audio-detection-samples",
    fs: int = 16000,
    clip_s: float = 1.0,
    max_per_class: int = 2000,
    seed: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int]:
    """Load DADS via HuggingFace datasets streaming.

    Returns
    -------
    X_list : list of np.ndarray, dtype float32, shape (fs*clip_s,)
    y      : np.ndarray of int64, shape (N,)
    groups : np.ndarray of int64, shape (N,)
    fs     : int
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        from datasets import load_dataset, Audio  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required for streaming mode. "
            "Install with: pip install datasets"
        ) from exc

    clip_len = int(round(fs * clip_s))
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups_list: List[int] = []
    counts = {0: 0, 1: 0}
    target = {0: max_per_class, 1: max_per_class}
    group_id = 0

    ds = load_dataset(repo_id, streaming=True, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"  Streaming DADS from {repo_id}  "
          f"(max {max_per_class} clips/class) ...")

    for example in ds:
        label = int(example["label"])
        if label not in target:
            continue
        if counts[label] >= target[label]:
            if all(counts[c] >= target[c] for c in target):
                break
            continue

        audio_dict = example["audio"]
        raw_bytes = audio_dict.get("bytes") if isinstance(audio_dict, dict) else None
        if not raw_bytes:
            continue

        try:
            data, fs_orig = _read_wav_bytes(raw_bytes)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Decode error (label={label}): {exc}")
            continue

        if fs_orig != fs:
            data = _resample(data, fs_orig, fs)

        clips = _segment(data, clip_len)
        remaining = target[label] - counts[label]
        clips = clips[:remaining]  # don't overshoot the cap
        
        X_list.extend(clips)
        y_list.extend([label] * len(clips))
        groups_list.extend([group_id] * len(clips))
        counts[label] += len(clips)
        group_id += 1

        total = counts[0] + counts[1]
        if total % 200 == 0:
            print(f"    Collected  no-drone={counts[0]:4d}  "
                  f"drone={counts[1]:4d}  total={total:4d}", end="\r")

    print(f"\n  Streaming complete: no-drone={counts[0]}, drone={counts[1]}")

    if not X_list:
        raise RuntimeError("No clips loaded from HuggingFace streaming. "
                           "Check network connectivity and HF access.")
    return X_list, np.asarray(y_list, dtype=np.int64), np.asarray(groups_list, dtype=np.int64), fs


# ---------------------------------------------------------------------------
# Public API — Mode 3: DroneAudioDataset (GitHub)
# ---------------------------------------------------------------------------

import re

def load_drone_audio_dataset(
    root: str,
    fs: int = 16000,
    clip_s: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int]:
    """Load the DroneAudioDataset from a local layout.
    
    Note: drone files are pre-chunked, so grouping is by session prefix to prevent leakage.

    Returns
    -------
    X_list : list of np.ndarray, dtype float32, shape (fs*clip_s,)
    y      : np.ndarray of int64, shape (N,)  -- 0=no drone, 1=drone
    groups : np.ndarray of int64, shape (N,)  -- source recording ID
    fs     : int  -- target sample rate
    """
    root = pathlib.Path(root)
    binary_dir = root / "Binary_Drone_Audio"
    clip_len = int(round(fs * clip_s))
    wav_extensions = {".wav", ".flac", ".ogg", ".mp3"}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups_list: List[int] = []
    
    session_keys = {}

    label_map = {
        0: binary_dir / "unknown",
        1: binary_dir / "yes_drone",
    }

    for label, class_dir in label_map.items():
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}\n"
                f"  Ensure root is correct and points to DroneAudioDataset."
            )
        wav_files = sorted(
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in wav_extensions
        )
        if not wav_files:
            raise FileNotFoundError(
                f"No audio files found under {class_dir}."
            )
        for wav_path in wav_files:
            stem = wav_path.stem
            if label == 1:
                # Drone files are pre-chunked; extract session prefix
                key = re.sub(r'_\d+_?$', '', stem)
            else:
                # Non-drone files are independent recordings
                key = stem
                
            if key not in session_keys:
                session_keys[key] = len(session_keys)
            group_id = session_keys[key]
            
            try:
                data, fs_orig = _read_wav(str(wav_path))
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] Skipping {wav_path.name}: {exc}")
                continue
            if fs_orig != fs:
                data = _resample(data, fs_orig, fs)
            clips = _segment(data, clip_len)
            
            X_list.extend(clips)
            y_list.extend([label] * len(clips))
            groups_list.extend([group_id] * len(clips))

    if not X_list:
        raise RuntimeError(
            f"No clips loaded from '{root}'. "
            "Verify dataset structure and audio file integrity."
        )
    return X_list, np.asarray(y_list, dtype=np.int64), np.asarray(groups_list, dtype=np.int64), fs

# ---------------------------------------------------------------------------
# Public API  Mode 4: DREGON Real DOA
# ---------------------------------------------------------------------------

# Citation: DREGON dataset documentation (8-channel array)
# 3D coordinates in meters. For 2D azimuth estimation, we use the x, y projection.
DREGON_POSITIONS = np.array([
    [ 0.05,  0.05,  0.05],
    [-0.05,  0.05,  0.05],
    [-0.05, -0.05,  0.05],
    [ 0.05, -0.05,  0.05],
    [ 0.05,  0.05, -0.05],
    [-0.05,  0.05, -0.05],
    [-0.05, -0.05, -0.05],
    [ 0.05, -0.05, -0.05]
])

def load_dregon_flight(path: str) -> Tuple[np.ndarray, np.ndarray, int, float | None]:
    """Load an 8-channel DREGON flight WAV and array geometry.
    
    Returns
    -------
    X : np.ndarray, shape (8, N), dtype float32
    positions : np.ndarray, shape (8, 2)
    fs : int
    gt_azimuth_or_None : float or None
    """
    if _HAS_SF:
        data, fs = sf.read(path, always_2d=True)
    elif _HAS_SCIPY_WAV:
        from scipy.io import wavfile
        fs, data = wavfile.read(path)
        if data.ndim == 1:
            data = data[:, None]
    else:
        raise ImportError("soundfile or scipy is required.")
        
    data = data.T  # (channels, N)
    if data.shape[0] != 8:
        raise ValueError(f"Expected 8 channels for DREGON, got {data.shape[0]}")
        
    if data.dtype.kind in ("i", "u"):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / max(abs(info.min), info.max)
    else:
        data = data.astype(np.float32)
        
    gt_azimuth = None
    csv_path = pathlib.Path(path).with_suffix(".csv")
    if csv_path.exists():
        try:
            lines = csv_path.read_text().splitlines()
            if lines and len(lines) > 1:
                parts = lines[1].split(',')
                if len(parts) >= 2:
                    gt_azimuth = float(parts[1])
        except Exception:
            pass

    return data, DREGON_POSITIONS[:, :2], int(fs), gt_azimuth
