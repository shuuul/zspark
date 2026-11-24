#!/bin/bash
# The Python reference environment for checkpoints that run through their own HF code (spec 137 WS-00; Qwen3-ASR,
# spec 138, adds `qwen-asr` to it). Source it: `source tools/reference/hf_env.sh`. Model-independent; no checkpoint
# is read. It makes ~/.cache/zspark/venv-reference once (uv, Python 3.12, torch + torchvision from PyTorch's cu130
# index: aarch64 wheels, CUDA 13 as the GB10's driver) and activates it. HF_ENV_EXTRA, words, are more pip
# requirements for this call (e.g. HF_ENV_EXTRA="qwen-asr"). Pins: transformers 4.57.6, the last 4.x (the
# Unlimited-OCR card says 4.52.1; 4.57.6 is the newest release whose `transformers.utils.import_utils` still
# exports `is_torch_fx_available`, `modeling_utils` `no_init_weights` and `DynamicCache.from_legacy_cache` —
# all removed in 5.x; 5.23.0 verified by an import check of the checkpoint's own module set), the card's other
# requirements as they are at its README; Pillow stays 12.3.0 (13.0.0's bicubic resize writes different bytes —
# verified — and `unlimited_ocr/resize_cases.py`'s fixture records them), PyMuPDF is the card's. Needs network
# the first time only (a download takes no lock).
VENV=${ZSPARK_REFERENCE_VENV:-$HOME/.cache/zspark/venv-reference}
PY="$VENV/bin/python"
[ -x "$PY" ] || uv venv --python 3.12 "$VENV" >&2 || return 1
if ! "$PY" -c 'import torch, torchvision' 2>/dev/null; then
    uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu130 torch torchvision >&2 || return 1
fi
if ! "$PY" -c 'import transformers, fitz, einops, addict, easydict, safetensors, matplotlib, psutil, PIL; assert transformers.__version__ == "4.57.6", transformers.__version__; assert PIL.__version__ == "12.3.0", PIL.__version__' 2>/dev/null; then
    uv pip install --python "$PY" "transformers==4.57.6" "Pillow==12.3.0" einops addict easydict pymupdf \
        psutil matplotlib safetensors numpy tqdm requests >&2 || return 1
fi
[ -z "${HF_ENV_EXTRA:-}" ] || uv pip install --python "$PY" ${HF_ENV_EXTRA} >&2 || return 1
source "$VENV/bin/activate"
