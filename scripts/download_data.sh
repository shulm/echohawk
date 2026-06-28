#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/dads data/dregon
# DADS (Hugging Face) — requires: pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="geronimobasso/drone-audio-detection-samples",
                  repo_type="dataset", local_dir="data/dads")
PY
echo "DREGON: download flights manually from https://dregon.inria.fr/ into data/dregon/"
