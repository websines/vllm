#!/usr/bin/env bash
# Phase 0 model fetch for the vllm-port project.
# Downloads:
#   - cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4   (~26 GB target, AWQ INT4)
#   - z-lab/Qwen3.6-27B-DFlash             (~3.5 GB DFlash draft, BF16)
# Both go to ~/models/. Skips if config.json already present.

set -u
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv/bin/python not found."
    exit 1
fi
PY=.venv/bin/python

mkdir -p "$HOME/models"

# Make sure huggingface_hub is installed in the venv
if ! "$PY" -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    uv pip install -q huggingface_hub
fi

cat > /tmp/fetch_models.py <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

models = [
    ("cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4",
     os.path.expanduser("~/models/Qwen3.6-27B-AWQ-BF16-INT4"),
     "~26 GB"),
    ("z-lab/Qwen3.6-27B-DFlash",
     os.path.expanduser("~/models/Qwen3.6-27B-DFlash"),
     "~3.5 GB"),
]

for repo, target, size in models:
    cfg = os.path.join(target, "config.json")
    if os.path.exists(cfg):
        print(f"SKIP: {repo} already at {target}")
        continue
    print(f"DOWNLOAD: {repo} ({size}) -> {target}")
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        max_workers=8,
    )
    print(f"OK: {repo}")
print("All models present.")
PYEOF

echo "======== Fetching models ========"
"$PY" /tmp/fetch_models.py
RC=$?

echo
echo "======== Disk usage ========"
du -sh "$HOME/models"/* 2>/dev/null
echo
df -h "$HOME" | tail -2

echo
echo "======== Listings ========"
for d in "$HOME/models/Qwen3.6-27B-AWQ-BF16-INT4" "$HOME/models/Qwen3.6-27B-DFlash"; do
    echo "--- $d ---"
    ls -lah "$d" 2>/dev/null | head -20
    echo
done

echo "fetch exit: $RC"
echo 'DONE'
