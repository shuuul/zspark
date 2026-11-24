#!/bin/bash
# Writes zig/tests/fixtures/text/grammar_masks.txt, and with a checkpoint directory grammar_masks_checkpoint.txt:
# xgrammar's masks, for the differential tests of zig/src/text/json_schema.zig. `grammar_masks.zig` says what the
# files hold.
#
#   tools/reference/grammar_masks.sh XGRAMMAR_CHECKOUT XGRAMMAR_BUILD [CHECKPOINT_DIR]
#
# XGRAMMAR_CHECKOUT is a clone of https://github.com/mlc-ai/xgrammar (the fixtures in the repository are from
# commit 8262b5c, the one TensorFold 0.6.5's xgrammar 0.1.32 wheel is built near), XGRAMMAR_BUILD a directory
# where its static library was built without Python:
#
#   cmake -S XGRAMMAR_CHECKOUT -B XGRAMMAR_BUILD -DXGRAMMAR_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_BUILD_TYPE=Release
#   make -C XGRAMMAR_BUILD -j20          # about a minute
#
# The shim and the generator are built in WORK (default: a new temporary directory). No GPU.
set -euo pipefail
if [ $# -lt 2 ]; then sed -n 2,15p "$0"; exit 2; fi
X=$(realpath "$1")
B=$(realpath "$2")
CKPT=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
WORK=${WORK:-$(mktemp -d)}
export PATH=/home/linuxbrew/.linuxbrew/bin:$PATH
COMMIT=$(git -C "$X" rev-parse --short HEAD)

g++ -std=c++17 -O2 -fPIC -shared -o "$WORK/libzxg.so" "$HERE/grammar_masks.cc" -I"$X/include" \
  -I"$X/3rdparty/dlpack/include" -I"$X/3rdparty/picojson" "$B/libxgrammar.a" -lpthread
zig build-exe -OReleaseSafe -L"$WORK" -lzxg -lc -rpath "$WORK" --dep schema \
  -Mroot="$HERE/grammar_masks.zig" -Mschema="$REPO/zig/src/text/json_schema.zig" \
  --cache-dir "$WORK/zig-cache" --global-cache-dir "$WORK/zig-global" -femit-bin="$WORK/grammar_masks"

OUT=$REPO/zig/tests/fixtures/text
"$WORK/grammar_masks" "$OUT/tiny_tokenizer.json" tiny "$COMMIT" > "$OUT/grammar_masks.txt"
if [ -n "$CKPT" ]; then
  "$WORK/grammar_masks" "$CKPT/tokenizer.json" checkpoint "$COMMIT" > "$OUT/grammar_masks_checkpoint.txt"
fi
ls -la "$OUT"/grammar_masks*.txt
