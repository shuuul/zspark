# Build and deploy from a clean checkout

This guide installs zspark's source and uses checkpoints you download separately. It does not require the maintainer's shell environment, cache paths, or machine names. The [weight guide](weights.md) records checkpoint revisions and licenses; [HTTP smoke](http-smoke.md) checks a running deployment.

## Requirements

- NVIDIA DGX Spark / GB10 with 128 GB unified memory, Linux aarch64, CUDA 13 and a compatible NVIDIA driver. The engine compiles for `sm_121a`.
- Zig 0.17.1, Rust and Cargo 1.85 or later, Git, Bash, GNU coreutils, util-linux (`flock`), and the CUDA toolkit's supported C++ host compiler. `curl` and `jq` are needed for the HTTP smoke.
- Two-node deployment additionally needs NCCL headers/library on both nodes, a configured direct ConnectX link, `iproute2`, RDMA userspace tools, and key-based SSH from head to peer. Pin the same source revision and toolchain on both nodes.
- Free disk for the selected checkpoints and build outputs. Checkpoint sizes and device footprints are in their recipes. A model directory must include its configuration/tokenizer/processor files, auxiliary safetensors and optional separate drafter, not only the main shards.

Install Zig from the [official download](https://ziglang.org/download/) or your package manager at exactly 0.17.1. Install CUDA and NCCL using [NVIDIA's Linux CUDA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and [NCCL installation guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html); installing NCCL is separate from installing CUDA. Do not change a serving machine's driver or package versions merely to follow a recipe.

Check the installed tools on each node:

```bash
zig version
rustc --version
cargo --version
nvcc --version
nvidia-smi
```

On Debian/Ubuntu, `dpkg-query -W libnccl2 libnccl-dev` reports the installed NCCL packages. The verified lab toolchain is recorded below; compatible package names and installation paths depend on your distribution.

## Source and build

Clone on each node, choose the same source commit, and build in the clone:

```bash
git clone https://github.com/shuuul/zspark.git
cd zspark
# Use the same release tag or full commit on both nodes.
zig build -j2 -Dnccl=/usr
./zig-out/bin/zspark version
./zig-out/bin/zspark-router version
```

`-j2` bounds simultaneous compilation on a machine that also serves models. The first build fetches CUTLASS using the hash in `build.zig.zon` and the audio decoder crates pinned in `zig/src/audio/Cargo.lock`. TensorFold's host is already vendored. `-Dnccl=/usr` names the NCCL installation prefix. `-Dcuda=/usr/local/cuda` selects a different toolkit prefix if needed. For one Spark, NCCL is optional. Runtime needs the built `zspark` and `zspark-router` beside each other, plus their linked CUDA/NCCL libraries.

For xgrammar-backed constraints such as regex and EBNF, build the optional grammar library from its pinned source checkout:

```bash
git clone --branch v0.2.9 --recurse-submodules https://github.com/mlc-ai/xgrammar
zig build grammar -Dxgrammar="$PWD/xgrammar"
```

Keep `zig-out/lib/libtfgrammar.so` with the binaries when installing elsewhere, and restart after installing it. See [structured output](serving.md) for the loader paths. The clean-stack validation below emitted the missing-library warning: its six smoke checks passed, but they do not prove full xgrammar constraint coverage. No grammar library was copied from production to make the clean build pass.

A GPU-free contributor check requires Zig and the host tools, without the CUDA toolkit or checkpoints:

```bash
zig build test-host
```

It skips optional checkpoint fixtures when their variables are unset. It still resolves the pinned package dependency. See [CONTRIBUTING.md](../CONTRIBUTING.md) for targeted device checks and the repository's machine/GPU/cache locks.

## Link, SSH and checkpoint placement

The addresses `192.0.2.1` (head) and `192.0.2.2` (peer) below are documentation examples, not addresses to copy onto your network. Use the peer's configured IPv4 address on the direct link, not its management/LAN address.

```bash
ip route get 192.0.2.2
rdma link
ssh -o BatchMode=yes 192.0.2.2 'hostname; zig version'
```

Verify the host key normally; do not disable host-key checking. If accounts differ, configure SSH's `User` for the peer IP in your local SSH config. `--peer` takes a plain IPv4 address, not `user@host` or an SSH alias. The stack discovers the link's NCCL interfaces on each node through `zspark stack link-env PEER_IPV4`.

Download the full fixed revisions in [weights.md](weights.md), including the selected drafter. On two nodes Flash Next and its drafter must be available to both ranks; Nemotron ASR and its default DFlash2 drafter also need their documented paths on both nodes. `--peer-dir` can change Flash Next's target path on the peer; `--peer-zspark` names that node's binary. Do not assume an HF cache snapshot's symlinks work after copying only its directory. Follow the weight guide's copy and verification instructions; the GLM drafter no-copy restriction still applies.

## Start and stop the stack

Copy the portable configuration example, then set its paths and flags:

```bash
cp zspark.local.example.sh zspark.local.sh
```

A two-node Flash configuration can use the following `STACK_ARGS` array, with the example peer address replaced by your direct-link address. Both nodes must have the specified target and drafter paths; `--peer-zspark` below assumes the same home and checkout layout on both nodes.

```bash
STACK_ARGS=(
    "$HOME/models/Qwen3.8-Flash-Next-FP4-Local"
    --peer 192.0.2.2
    --peer-dir "$HOME/models/Qwen3.8-Flash-Next-FP4-Local"
    --peer-zspark "$HOME/repo/zspark/zig-out/bin/zspark"
    --listen 127.0.0.1:18888
    --port 18901 --at 29651 --mps 0
    -- --drafter "$HOME/models/Qwen3.8-Flash-Next-DFlash2" --parallel 4
)
```

The router port, backend ports 18901–18903, and NCCL ports 29651–29652 must be unused. The backend and link ports are distinct from the client-facing router. The root example also shows optional OCR and Nemotron ASR. Selecting fewer models changes the required device memory; it does not change the checkpoint's precision.

Before starting, inspect `nvidia-smi` and `free -h` on both nodes. A second checkout and different ports isolate process ownership, not GPU memory. Do not launch another full checkpoint while an existing deployment occupies the necessary memory. Recorded Flash TP=2 configurations take about 44–49 GiB per rank at parallel 1, and the current Flash/Nemotron configuration takes about 79 GiB per node. Leave room for transient load buffers, host allocations and the operating system; these figures are recorded configurations, not an admission guarantee for every context or concurrency setting.

```bash
DRY_RUN=1 ./start.sh
./start.sh
# Follow the log path printed by start.sh until the router is ready.
tools/http_smoke.sh --help
./stop.sh
```

`start.sh` returns when the supervisor starts, before model readiness. Each model's health check and the router startup are in the log. `stop.sh` sends SIGTERM only to the recorded supervisor whose process start time and argv still match; it waits up to `STOP_S` (120 seconds). A timeout is a failure and keeps the record for inspection, without force-killing another process. A deployment started by another launcher must be stopped through its own lifecycle, not through a new checkout's PID file.

## Validation scope

A clean source checkout does not mean a freshly provisioned operating system or an empty global compiler/package cache. Record the source commit, toolchain, build commands and model identities when claiming a reproduced deployment. Do not use older benchmark tables as evidence that a new checkout started successfully.

Validation on 2026-11-13 used source commit `fdd9f947d4ed38d2fb00d667e1f9fc838ab3aef2`, Zig 0.17.1, CUDA 13.0 (`nvcc V13.0.88`) and NCCL 2.32.6 (`libnccl2` and `libnccl-dev` package `2.32.6-1+cuda13.2`). The clean head clone built with `zig build -j2 -Dnccl=/usr --summary all` under a minimal environment; both binaries reported that source commit. `ldd` resolved CUDA and NCCL, and `stack link-env` discovered the direct-link NCCL settings. These checks reused the installed operating system and global dependency cache.

The clean peer clone also built successfully with the same toolchain and `-j2 -Dnccl=/usr` under a minimal environment: 101/101 build steps succeeded. A clean-head launcher dry-run printed the intended two-rank command with separate client/backend/NCCL ports and the clean peer binary. Neither check loads the model.

After the owner authorized a maintenance window on 2026-11-13, the original stack was gracefully stopped and both GPUs and their locks were verified released. The clean clones then started Flash Next and Nemotron ASR across both nodes through root `start.sh`: router `127.0.0.1:18888`, backends 18902/18903, NCCL ports 29651/29652, MPS off, Flash parallel 4, and the pinned DFlash2 directories. Both backends reached health; HTTP smoke passed six checks including streamed chat, a required tool call, unknown-model rejection and ASR. OCR was not mounted. Root `stop.sh` completed successfully and both nodes had no remaining GPU processes. The original staged production revision `6a886889` was then restored with its original configuration; the same six checks passed against its LAN endpoint.
