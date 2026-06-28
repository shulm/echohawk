# Datasets

Raw audio is **not** committed (see `.gitignore`). Download into this folder.

## DREGON — microphone-array + ground-truth DOA
- Site: https://dregon.inria.fr/
- 8-channel array on a UAV; ground-truth direction (<2° error); static + in-flight.
- Use for `beamforming` (Track B). Place flights under `data/dregon/`.

## DADS — Drone Audio Detection Samples (detection)
- Hugging Face: `geronimobasso/drone-audio-detection-samples`
- Binary 0 (no drone) / 1 (drone), 16 kHz mono.
- Use for detection (Track A). Place under `data/dads/`.

## Backgrounds / negatives
- ESC-50: https://github.com/karolpiczak/ESC-50
- UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html

See `scripts/download_data.sh` (created per HANDOFF.md) for fetch commands.
