#!/usr/bin/env bash
# Spec 151 WS-04: the runtime smoke check of `zspark151-nemo-vllm:local` on the host (spark-b, GB10).
# `vllm serve --help` can only run where a GPU is visible — vLLM infers a device default while adding
# CLI args — so this runs in the container on the host, never in the image build. Loopback only; no
# server is started.
#
#   docker run --rm --gpus all -v $HOME:$HOME zspark151-nemo-vllm:local \
#       bash ~/repo/zspark-151r/tools/reference/nemotron_transcribe/env_check.sh
set -euo pipefail
LOG=${1:-$HOME/.cache/zspark/scratch-151/env-check.log}
mkdir -p "$(dirname "$LOG")"
{
  python3 -c "import vllm, torch; assert vllm.__version__ == '0.32.0', vllm.__version__; print('vllm', vllm.__version__, 'torch', torch.__version__)"
  python3 -c "import nemo.collections.speechlm2.vllm.salm as s; s.register(); print('nemo_speechlm registered')"
  VLLM_PLUGINS=nemo_speechlm vllm serve --help=all > /tmp/serve-help.txt 2>&1
  for arg in --use-replayssm --mamba-backend --reasoning-parser --kv-cache-dtype --speculative-config --limit-mm-per-prompt --middleware --trust-request-mm-kwargs; do
    grep -q -- "$arg" /tmp/serve-help.txt && echo "arg ok: $arg" || { echo "MISSING: $arg"; exit 1; }
  done
  python3 -c "import vllm, nemo.collections.speechlm2.vllm.salm.audio as a; print('placeholder est (5s):', a.NeMoSpeechLMProcessingInfo._estimate_audio_tokens(80000, None, None))"
} 2>&1 | tee "$LOG"
echo "env check passed; log: $LOG"
