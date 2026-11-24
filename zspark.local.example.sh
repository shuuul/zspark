# zspark.local.example.sh — copy to zspark.local.sh (untracked) and edit for this machine.
#
# start.sh sources zspark.local.sh from the repository root; it is the one place the
# machine's checkpoint paths and the stack's shape are set. Every word of `zspark stack`'s
# command line lives in STACK_ARGS, quoted so paths with spaces survive; words after a
# lone "--" are Flash Next's own `serve` flags, on both ranks where --peer is given
# (`zspark stack` alone lists the flags; docs/serving.md has what each chooses).
#
# A checkpoint is a directory: either a plain directory of the model's files, or the
# Hugging Face snapshot under models--ORG--NAME/snapshots/HASH (the stack then serves it
# under NAME). Download the revision each recipe pins — its `revision:` line at the head
# of the page under docs/recipes/ is the commit to hold; a newer snapshot is a different
# checkpoint:
#
#   hf download nvidia/Qwen3.8-Flash-Next-FP4-Local \
#       --revision b9ce18a6887e18c089fdea8509a33e63429a3300
#
# then name the printed .../snapshots/b9ce18a... directory below (or copy the files and
# name that directory). Nothing is downloaded or built by start.sh itself.

STACK_ARGS=(
    # The one Spark default: Flash Next alone behind the router on 127.0.0.1:8888,
    # serving the number of streams `serve` would.
    ~/models/Qwen3.8-Flash-Next-FP4-Local

    # OCR and ASR opt in one flag each; either may be left out. Each is a `zspark serve`
    # of its own at --parallel 1 behind the same address, on --mps percent of the SMs
    # (25 by default). --asr-name keeps the published base id so clients name it.
    # --ocr ~/models/Unlimited-OCR-2                                     # baidu/Unlimited-OCR-2
    # --asr ~/models/Qwen3-ASR-1.7B-NVFP4-W4A16 --asr-name Qwen3-ASR-1.7B

    # For two Sparks, set PEER in .env.local and uncomment --peer below.
    # Flash Next then spans two ranks and needs an NCCL build (zig build -Dnccl=/usr).
    # Qwen3-ASR runs on the peer. Nemotron SpeechLM spans both Sparks and reads the same
    # --asr path on each. Use --peer-dir if the peer's Flash Next path differs.
    # --peer "${PEER:?Set PEER in .env.local}" --peer-dir ~/models/Qwen3.8-Flash-Next-FP4-Local

    # The router's address: 127.0.0.1:8888 binds loopback; 0.0.0.0:8888 serves the LAN.
    # --listen 0.0.0.0:8888

    # -- Flash Next's own serve flags, last: --drafter drafts with DFlash2 in the MTP
    # head's place, --parallel/--context size its pool, --name the id it serves under.
    # -- --drafter ~/models/Qwen3.8-Flash-Next-DFlash2 --parallel 4
)

# Example two-Spark stack with Flash Next and Nemotron ASR across both Sparks, DFlash2
# drafting, a LAN listener and MPS disabled. Set PEER in .env.local.
#
# STACK_ARGS=(
#     ~/models/Qwen3.8-Flash-Next-FP4-Local
#     --asr ~/models/Nemotron-3.5-Transcribe-30B-A4B-NVFP4 --asr-name nemotron-transcribe
#     --peer "${PEER:?Set PEER in .env.local}" --listen 0.0.0.0:8888 --mps 0
#     -- --drafter ~/models/Qwen3.8-Flash-Next-DFlash2 --parallel 4
# )

# Environment the wrappers read (no zspark.local.sh entry needed):
#   ZSPARK_BIN  the zspark binary, ./zig-out/bin/zspark unset; zspark-router must sit
#               beside it, as zig build leaves them
#   LOG_DIR     one log a start, ~/.cache/zspark/logs unset (stack-latest.log links it)
#   STATE_DIR   the pid, lock and recorded argv; default ~/.cache/zspark/stack-CKSUM,
#               where CKSUM identifies this checkout's absolute path
#   STOP_S      seconds to wait after SIGTERM (120); timeout fails and keeps the record
#   DRY_RUN=1   start.sh prints the stack's command and changes nothing; stop.sh says
#               what it would signal
