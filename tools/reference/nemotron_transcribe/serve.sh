#!/usr/bin/env bash
# Spec 151 WS-04: the model card's vLLM launch for NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 inside
# `zspark151-nemo-vllm:local` (build_image.sh). Loopback only — never a LAN address. The checkpoint is
# the HF hub snapshot 66c6adc (ZSPARK_NEMOTRON_NVFP4 overrides; ~/models/… is metadata-only).
#
#   docker run --rm -it --gpus all -v $HOME:$HOME zspark151-nemo-vllm:local \
#       bash tools/reference/nemotron_transcribe/serve.sh
#
# The card's profile: bf16 activations, fp8 KV cache, ReplaySSM on flashinfer's Mamba backend, MTP8,
# nemotron_v3 reasoning parser, the CTC-timestamp middleware; one audio of up to 3,600 s a prompt.
set -euo pipefail
SNAP="$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4/snapshots/66c6adc4020179567074d0399cda2a4eeb2bb347"
[ -f "$SNAP/config.json" ] || SNAP="${ZSPARK_HOST_HOME:-$HOME}/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4/snapshots/66c6adc4020179567074d0399cda2a4eeb2bb347"
CK=${ZSPARK_NEMOTRON_NVFP4:-$SNAP}   # HF snapshot = the weights; ~/models/… is metadata-only
export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0              # word timestamps need the V1 model runner (the card)
export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$HOME/.cache/zspark/scratch-151/flashinfer}
exec vllm serve "$CK" \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 \
  --trust-remote-code --dtype bfloat16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 \
  --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware \
  --host 127.0.0.1 --port 8000
