# Measuring a kernel and a round on the GB10

Traps of measurement: what a claim takes, what a trace shows, dirty lines between kernels, and how a round's
excess is attributed. Part of the `cuda-gb10` skill.

- One before-and-after pair of loads is not a measurement: on one binary and one setting two loads' decode
  seconds were 2.35% apart (166.26, 162.92 s over 71 requests), and a first pair read as -2.2% what four loads
  showed to be 0.2 to 0.4%. A round's claim takes four loads in the order off, on, on, off and the requests
  paired by name (their replies' hashes equal): the gain is each on load against the off load beside it, held
  against what the two loads of one side differ by (0.2 to 0.3% there) (spec 073 WS-02).
- Dirty lines before a read cost by amount (the `dirty` form: a `cudaMemsetAsync` before a 10 to 16 MB
  weight-streaming launch): up to 2 MB nothing measurable (under 1 us), 4 MB 1 to 3 us, 8 MB 10 to 15, 32 MB 26
  to 60; the small launch between (2 us) pays too, 1.6 to 2.2 us at 8 MB and 6 to 14 at 32 (spec 073 WS-02).
- A kernel alone is not the kernel in a round: weigh it by the shapes the round gives it and compare the two
  (spec 070's trace of four replies). 87% of a decoding reply's trellis launches are of 2 to 8 rows (2 rows
  27%, 3 18%, 4 14%), 10% of one. There the layer's first large linear (80 or 96 blocks of 8 warps, after the
  activation kernels) is 5.6 us slower at 2 rows than alone, 9 at 4, 23 at 16 (1.1% of the reply's kernel
  time), while the next linear and the experts match their standalone times. The mechanism is the read-out's
  logits as dirty lines — a kernel's own stores linger in L2 and write back during the next kernel's read: a
  64 MB-read GEMM after a chain that stored its 3 MB state ran +16.7 us, +16.5 with a 10 to 150 us spin
  between (no decay: not clocks, not occupancy), the same 3 MB laid by `cudaMemsetAsync` cost nothing at that
  size (the memset's own table begins at 4 MB), and not storing the state removed it. So the first linear pays
  the logits' writeback: 2 us of them at one row and 22 at 16, the logits' bytes scaled. A lull before the
  linear (8 to 38 single-block launches back to back, 7 to 32 us with the other SMs idle) changes nothing, so
  no clock ramp shows; the GPU idle before it (a synchronize) is +2.5 us on one linear and -2 on another; and
  evicted inputs are what the standalone profile already has (16 MB of other weights before every launch). A
  standalone profile bounds a kernel from below (spec 097 WS-05).
- `nsys --trace=cuda` on `serve` records decoding rounds as graphs by default, hiding their individual
  kernels. Use `--cuda-graph-trace=node` for a serving round's kernel histogram; it records replayed nodes
  on both TP2 ranks (spec 166). Host `Face.forward` calls count eager work and graph construction, not
  every replay; use device-node counts for the actual placement launches.
- `ncu` on the GB10 has no `dram__` metric (unified memory); `lts__d_sectors_fill_sysmem.sum` is the sectors
  L2 filled from memory, `lts__t_sectors_srcunit_tex_op_read_lookup_miss` nearly the same. The bf16
  `m16n8k16` peak by `sm__pipe_tensor_subpipe_hmma` is about 106 TFLOPS at 2.14 GHz (442 GFLOP in 8.16 ms read
  as 51%). Durations under `ncu` moved 8.2 to 9.3 ms between metric sets on one binary: compare counts, not
  times (spec 107 WS-03).
- A 16k-token prompt by `generate` (nsys, rows=2048, spec 105 WS-03): the routed experts 3.83 s of 9.4; the
  head's g16x1940 325 ms is `generate`'s alone, scoring every row, where `serve` scores the last
  (`.none`/`.last`).

## Loads: the design, their spread, the build

- **Off, on, on, off** is the standard of every served claim since spec 073: one binary a side, both machines
  held alone, every request paired by name and its reply's hash checked equal on each side. The gain is each on
  load against the off load beside it, held against off to off and on to on. Same-bits loads already taken may
  stand in for the off side (spec 116). A table that records figures and claims no gain takes one load (spec
  116's close).
- **What two loads of one binary differ by** (the spread a claim must clear), by what was measured:

| Measure | Spread | Source |
| --- | --- | --- |
| Decode tok/s, every reply, Flash Next NVFP4 / the 3.05 bpw pack | 1.5% / 0.3%; 2.35% once over 71 requests | spec 073 WS-02, WS-04 |
| Decode tok/s, every reply, GLM FP4-Local two ranks (MTP) | off to off 0.5 to 2.9%; on to on 0 to 1.0% | spec 114 WS-02 |
| The same, DFlash2's default, two loads | 3.1% | spec 114 WS-02 |
| A 16k cold prompt's first token, GLM two ranks | off to off 2.3%; three loads of one setting 9.38 to 10.0 s; the first load of a session 0.86x to 1.00x | spec 114 WS-08, WS-09 |
| `total_tps` of four pooled streams, one load's span 2.5 s | about 7% load to load | spec 097 WS-04 |

- **The first load of a session reads low**, and a load's first run 3 to 6%: drop it (spec 076; 1.5 to 2% twice
  on Flash Next, 073; the first of four GLM loads 0.86x on both sides, 114 WS-09; the first of five pooled
  loads, 097). Never take it as the base alone.
- **Order and count.** Eight loads base, A, B, C, C, B, A, base show drift where pairs do not (spec 076).
  Repeats of a request inside one load are not loads, and one load's two values are not a range (spec 097
  WS-07); judge first tokens on load means (spec 095). GLM on two ranks: a change under 2% needs ten replays a
  load and more than one load a build (sd 0.1 to 0.2 tok/s, spec 093). Wait for the load average under 1.0 with
  `machine.lock` held: NVFP4 loads taken at 3.5 to 6.1 overlapped (spec 095).
- **Traced is not timed.** `nsys` lengthens what a rank waits on (a told wait 104.5 untraced against 107.4,
  spec 093; 43.2 against 45.0 tok/s, 090; +1.0 to 1.5 ms a round, 038). A wait a trace shows is confirmed by an
  untraced pair before it is built on (GLM's told path: 3.8 ms traced, nothing timed, spec 093). Take a trace's
  figures from its second pass (62 lazy `cudaGraphInstantiate` in the first 20 s). `two_ranks.sh` works with
  `WRAP='nsys launch --session-new=NAME --trace=cuda --cuda-graph-trace=node'`; start/stop the session on
  each machine around the same request wave (spec 166). Compare untraced waves before and after it.
- **`nsys launch` costs nothing a client sees.** Graph-level (`-t cuda,osrt`) on `serve`, captured by `nsys
  start`/`stop` while a bench runs: 142.2 / 137.7, 61.3 / 61.7, 91.7 / 91.9 tok/s traced against untraced on
  Flash Next FP4-Local; `--cuda-graph-trace=node` 60.9 against 61.8 on one stream (spec 144 WS-01). A trunk
  round is one graph in a capture, so a round's time splits by the gaps between them
  (`scratch-144/tools/rounds2.py`).
- A host `cudaDeviceSynchronize` duration includes the GPU work it waits for; it is not recoverable CPU
  overhead. Control-send host duration, peer wait and NCCL device intervals are separate measurements.
  Use interval unions and overlaps for device occupancy; a sum of NCCL kernel durations alone does not
  establish the time a control-protocol change could save (spec 166).
- **A client sees a round 5 to 9 ms off once in 15 to 40 rounds**: count outliers with a sign test (spec 089
  WS-04).
- **The drive sleeps.** The NVMe (Samsung MZALC4T0HBL1) enters a deep idle state after 100 to 120 ms (APST,
  `default_ps_max_latency_us` 100000); the next read waits 9.9 to 10 ms against 183 us awake.
  `weights/drive.zig` reads 4 KiB O_DIRECT every 50 ms while serving: long rounds 30 to 2 of 402, reply time
  -1.7% (spec 089). A cold PLE row is checked with `posix_fadvise(DONTNEED)` then `fincore` before a load;
  `majflt` cannot tell cold from warm under WILLNEED, `inblock` can (spec 087 WS-01, 089 WS-01).
- **A switch must reach the timed path**: `ZSPARK_SHARED` never reached `serve` (spec 058).
- **A whole-load tok/s cannot resolve an effect its own size.** The pool's PLE rows asked ahead of the round
  (0.76 ms of host a round for two seats in spec 095's trace) led by +4.2% on the means of five loads a side,
  and the ranges overlapped (base 183.35 to 212.69, head 197.83 to 216.84); the branch was kept (spec 097
  WS-04). For a mechanism measured in a trace, count it there (the rounds and their host gaps from `nsys`) or
  lengthen the load's span; do not add loads of a short one.
- **Safe and fast builds differ in host work, not in decoding.** One commit's GLM load took 37 s fast and 47
  safe (byte-wise host loops under bounds checks), both printing one version; the recipe's tables redone fast
  had the safe build's replies, decode and cold prompt within the spread, the load 15.0 s against 16.0
  (131fd5a3, bbe474bf). Time the build that is served (`-Doptimize=fast`, the default since 131fd5a3; `zspark
  version` prints the mode); `check.sh` keeps the tiers safe.
- **A bench must reach its own server.** A vLLM server another client started held 127.0.0.1:18888 while zspark
  queued for the locks; the readiness check passed on it and every request had a 404. `run_two.sh` waits for
  zspark's own serving line (spec 111 WS-01).
- **Pass a list as an array in zsh.** `"$VAR"` holding `layers=4 windows` went as one word and the two-rank
  check ended `NoPeer` (spec 111 WS-01 and WS-02, 071 before them).
- **A projection from standalone figures overshoots the served gain**: DSA's window form projected 0.48 s and
  served 0.27 (spec 116 WS-01); 114 WS-03 and WS-04's workers estimated 0.6 and 0.7 s and the merge served
  1.26 s (the other way). Serve before claiming.
