# Nemotron-3.5-Transcribe reference environment (spec 151 WS-04): vLLM 0.32.0 with NeMo Speech main's
# `nemo_speechlm` vLLM plugin, on GB10 (aarch64, CUDA 13, sm_121a). Built on the local spec-142 image
# (`zspark142-vllm-audio:local`, itself eugr/spark-vllm-b12x + audio decoders): Ubuntu 24.04, python3.12,
# torch 2.13.0+cu130, nvcc 13.0 on PATH, flashinfer-python with the sm_121a JIT cache.
#
# The card's environment recipe (`uv sync --extra asr --extra vllm --locked` inside NVIDIA-NeMo/Speech)
# resolves vllm==0.32.0 and torch==2.13.0 — exactly what is installed here. This Dockerfile installs the
# PyPI wheels over the image's matching torch and checks out the pinned NeMo commit instead of `uv sync`
# (which builds a second venv for a python the base lacks; `pip` against the image's Python 3.12 installs
# the same resolution).
#
# Build (spark-b):  tools/reference/nemotron_transcribe/build_image.sh
FROM zspark142-vllm-audio:local

# Replace the image's 2026-10-15 vllm dev build with the 0.32.0 release. The wheel is staged in the
# build context by build_image.sh (a slow PyPI reach from the Sparks truncated the 321 MB download on
# the first attempt; a staged file cannot be corrupted mid-build). deps resolve against a constraint
# holding torch and transformers at the image's versions so the base's CUDA stack is untouched.
COPY vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl flashinfer_python-0.7.1-py3-none-any.whl /tmp/
RUN printf 'torch==2.13.0\ntransformers==5.19.0\n' > /tmp/base.txt \
 && pip install --no-cache-dir -c /tmp/base.txt \
      /tmp/vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl \
      "av" "scipy" "soundfile" "soxr" "mistral_common[audio]" \
 && pip install --no-cache-dir --no-deps /tmp/flashinfer_python-0.7.1-py3-none-any.whl
# flashinfer-python 0.7.1 (staged wheel — a PyPI reach from the Sparks truncated the first try):
# vllm 0.32.0's requirements pin says ==0.7.0.post1, but the base image's flashinfer-jit-cache-sm121a
# is 0.7.1 and flashinfer's own jit.env pair check hard-fails on a python/jit-cache mismatch
# (RuntimeError at `import flashinfer.fused_moe` during NVFP4 MoE backend selection on GB10). vLLM's
# flashinfer call sites are version-agnostic between 0.7.0 and 0.7.1.
COPY nemo /opt/nemo-speech
# NeMo Speech: --no-deps editable install (the card's own recipe) plus the `asr` extra's packages and the
# two the card adds by hand (av, opencv — opencv already satisfies vllm's pin).
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/nemo-speech \
 && pip install --no-cache-dir \
      "aistore" "onnx>=1.7.0" "scikit-learn" "smart-open" "tensorboard" "text-unidecode" "wrapt" \
      "braceexpand" "kaldialign" "lhotse==2.0.0a6" "librosa>=0.10.1" "packaging" "sacrebleu" \
      "whisper_normalizer" "datasets>=3.2.0" "pandas" "sentencepiece<1.0.0" \
      "hydra-core>1.3,<=1.3.2" "lightning>2.2.1,<=2.4.0" "omegaconf<=2.3" "torchmetrics>=0.11.0" \
      "webdataset>=0.2.86" "nv_one_logger_core>=2.3.1" "nv_one_logger_training_telemetry>=2.3.1" \
      "nv_one_logger_pytorch_lightning_integration>=2.3.1" "peft>=0.18.1" "wandb" "nltk" \
      "matplotlib" "jieba" "pypinyin" "pypinyin-dict" \
      "av==19.0.2" "editdistance" "jiwer"
# The card's vLLM 0.32.0 DFlash2 fix: anchor ids are clamped before indexing the successor table; without
# it the server can fail when requests finish. Idempotent, asserts the exact upstream lines.
RUN python3 - <<'EOF_PATCH'
from pathlib import Path
import vllm
path = Path(vllm.__file__).parent / "model_executor/models/qwen3_dflash2.py"
source = path.read_text()
if "safe_anchor_token_ids" not in source:
    old = "    successors = successor_table[candidate_ids]\n"
    anchor = "            anchor_token_ids[:, None, None].expand(-1, 1, top_k),"
    assert source.count(old) == 1 and source.count(anchor) == 1, "unexpected vLLM version"
    source = source.replace(old, (
        "    safe_anchor_token_ids = anchor_token_ids.clamp(\n"
        "        min=0, max=predecessor_table.shape[0] - 1\n"
        "    )\n" + old
    )).replace(anchor, "            safe_anchor_token_ids[:, None, None].expand(-1, 1, top_k),")
    path.write_text(source)
EOF_PATCH
# The build's own smoke check: import vllm + the plugin's register(). `vllm serve --help` cannot run at
# build time — vLLM infers a device default while adding CLI args and the build has no GPU; the
# arg-presence check is env_check.sh's, run on the host.
RUN python3 -c "import vllm, torch; assert vllm.__version__ == '0.32.0', vllm.__version__; print('vllm', vllm.__version__, 'torch', torch.__version__)" \
 && python3 -c "import nemo.collections.speechlm2.vllm.salm as s; s.register(); print('nemo_speechlm registered')"
ENV VLLM_PLUGINS=nemo_speechlm
ENV VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
ENV VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
ENV VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
