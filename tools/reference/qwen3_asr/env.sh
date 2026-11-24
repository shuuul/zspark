#!/bin/bash
# The Python environment of spec 138's reference (Qwen3-ASR-1.7B, the `qwen-asr` package, transformers
# backend, bf16 on the GPU). Builds ~/.cache/zspark/qwen3-asr-env/venv on an aarch64 Spark with CUDA 13 and
# prints the python to use. No container: uv takes the PyPI wheels (torch 2.14.1+cu130 for aarch64 with its
# nvidia-* libraries, transformers 4.57.6 as the checkpoint was saved with, librosa 1.0.0, soundfile 0.14.0,
# soxr 1.1.0: what qwen-asr 0.0.6 resolved to on 2026-10-25; `freeze.txt` beside the venv lists all).
# About 6 GB; checked on spark-b (GB10, sm_121: a bf16 matmul runs).
#
#   PY=$(tools/reference/qwen3_asr/env.sh)
#   flock -s ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock "$PY" -I tools/reference/qwen3_asr/export.py --help
set -euo pipefail
ENV=${ZSPARK_QWEN3_ASR_ENV:-$HOME/.cache/zspark/qwen3-asr-env}
if [ ! -x "$ENV/venv/bin/python" ] || ! "$ENV/venv/bin/python" -I -c "import qwen_asr" 2>/dev/null; then
  uv venv --python /usr/bin/python3 "$ENV/venv" >&2
  uv pip install --python "$ENV/venv/bin/python" "qwen-asr==0.0.6" >&2
  uv pip freeze --python "$ENV/venv/bin/python" > "$ENV/freeze.txt"
fi
echo "$ENV/venv/bin/python"
