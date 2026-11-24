# A round's switches, measured (spec 073 WS-06b)

How a switch's cost is measured in a served round and what the measurements found; moved from `SKILL.md`'s
section of the same name. The protocol is [`references/measuring.md`](references/measuring.md)'s.

- Verify the off side by the kernels in a trace, never by the command line: `env $E cmd` through ssh into zsh
  sets ONE variable (zsh does not split an unquoted variable) — twelve loads read +1.4% where the truth was
  +4.1%. Put the environment in a bash script on the far side; rank 1 of a two-rank run inherits nothing.
- A served round is a CUDA graph: `nsys --trace=cuda` alone shows only a prompt's kernels,
  `--cuda-graph-trace=node` every node. Node tracing costs 0.2 us a launch inside the graph and 2.5% of a
  reply, and a difference reads about 5% high where launches are removed; graph granularity costs nothing
  measurable (check a node-level total against it). Pair rounds by index across loads (bit-exact switches give
  the same rows a round); drop a request's last.
- A profile alone does not predict the round, in either direction: the DeltaNet chain's float4 state -4%
  alone, -27% in the round (37.7 to 25.4 us); the deep tile on the 320-wide read-out's up -5% alone, +1.5 to
  +4% in the round; small trellis linears gain 16 to 18% from the read ahead where alone showed nothing.
- A launch removed is worth its kernel time plus about 2 us of host when eager and 0.2 us inside a graph. The
  joined experts' whole gain is 144 launches less a round; their gate and up launch itself got slower.
- Four eager launches of a short chain over one row are their floors, not their bytes (spec 102 WS-02):
  `common/pick.cu` over a 79,591-wide bf16 row (four kernels of 78 blocks, 0.16 MB) takes 31 us a row from
  the projection's end, 123 at four rows one after another; its kernels' own time is 24 us a row. A launch
  over the rows (a block row a row) would be about one row's (reckoned, not built).
- CUTLASS tiles: 16x8x64 gains on the wide bf16 products (in_proj -3.1%, the head -5.7%) and loses on the
  320-wide up. A tile is chosen by shape, in a round.
- A served round's cost a row is not one figure (spec 075 WS-02, the 3.05 bpw pack): the trunk's kernels take
  2.1 to 2.6 ms a row from 8 to 31 rows and 1.16 from 64 to 128 by the client's rounds. Three quarters of a
  row under 31 is the experts' `grouped_kernel` (a program a row and slot, 70 to 80 ns a block). Why it falls
  past 32 rows was not traced: rows sharing experts (199 distinct of 250 at 25 rows, 391 of 730 at 73) and the
  wide kernels from `gdn.wide_rows` both start there — not resolved (spec 075). Hold two rounds against each
  other at equal rows before looking for a fixed cost: a "45 ms intercept" was 13 rows.
- `cudaFree` waits for the device as a synchronize does (the first of 13 took 2.1 ms behind 2.8 ms of queued
  work, the other twelve 5 us each); a `cudaMalloc` of a small buffer while serving is 40 us on the host,
  13 of them 0.5 ms of device idle a round. A device copy of 3 MB is 30 us (36 of them, 113 MB, 1.2 ms).
- A launched round of 2,175 kernels under `nsys --cuda-graph-trace=node` is 1 ms longer by the client than
  untraced (114 for 113 ms); the device idles 3.7 ms while it is launched where a replayed graph's idles 0.4.
- A store past a `cudaMalloc`'s end is not caught where it happens (spec 086): 5 to 7 rows (165 to 231 KB)
  past a 2.1 MB buffer among 80 GB of them ended five loads of seven in `an illegal memory access` (at once in
  the one that logged the round), under the name of whatever call next read the error (`b16 gemm: CUTLASS
  launch failed`, `cudaGetLastError()`): the message names no kernel. Whether such a store can also pass
  unseen was not observed (spec 086). `compute-sanitizer --tool memcheck --padding 1048576` before
  `zspark serve` names the kernel, the host frames and the allocation at the first such store (the pack loads
  in 30 s under it, a group of four replies of 48 tokens takes a minute; 94 GB used with `--context 16384`, so
  leave the tool that room). Without `--padding` it reported nothing in a load that may not have met the round
  (not separated). After a report with `--destroy-on-device-error kernel` the server spun without progress; a
  `TERM` to the server under the tool ends its requests in `misaligned address` and the summary in an internal
  error, so read the reports, not the exit. End the server, not the tool, or no summary is written and the
  server lives on.
- A decoding round's fixed costs (spec 089 WS-00, Flash Next; traced, not timed):
  - `cudaGraphLaunch` returns after a time that goes with its nodes, the device idle meanwhile: 0.14 to 0.16
    ms for 1,480 nodes and 6 to 8 us for 52 to 56 under `--cuda-graph-trace=graph` (0.10 to 0.14 us a node).
    Under `=node` the same calls are 0.65 to 0.79 ms and 26 to 38 us (0.5 us a node) and a 40 ms round is 1.8
    ms longer: read a launch call, or anything between kernels, from a graph-level trace (spec 089 WS-02).
    Between two nodes of a graph the device idles 0.11 us (node trace).
  - A blocking `cudaMemcpy` of 4 bytes waits for the graph queued before it as a synchronize does; a
    496,640-byte copy to pageable host memory is 25 us with its synchronize; an eager launch's call 3 to 5 us.
  - A pass's small copies around a graph cost less than they look (spec 095 WS-02, standalone `passcost.cu`):
    a 4-byte pageable upload, the launch, an 8-byte upload that waits for the graph, a synchronize and a 1 KiB
    pageable download take 13 us over a 94 us kernel; from pinned memory 18 (slower); without the waiting
    upload 9; both copies held in the graph as nodes to and from pinned memory, one `cudaStreamSynchronize`,
    12. Removing a host copy around a pass is worth 1 to 4 us, not the 26 to 38 a node trace showed.
  - `cudaMalloc` while decoding comes in bursts: 73 small buffers in one round are 6 to 7 ms of device idle
    (11 once), 197 to 341 at a request's start 17 to 48 ms. Made while the device computes a window instead,
    146 took 21 ms of host (145 us a call for 90), all inside a 140 ms window (spec 089 WS-02).
  - Rows copied from a file mapping of tens of GB (`MADV_WILLNEED` then `memcpy`, 1,632 bytes a row) cost
    0.75 to 0.85 ms a round whatever the rows (2 to 16) when the page cache does not hold them, some rounds
    10 ms, and 20 to 28 us a row when it does; the cache outlives the process, so a bench that repeats its
    prompts measures the warm case only.
  - A state-streaming kernel (the DeltaNet chain's, its replay) runs at 198 GB/s by its bytes reckoned from
    shapes, and a 52 MB bf16 GEMM once a round (the PLE's key, 10,240 by 2,560) at 219: at their bytes' rate.
  - A small GEMM is over its bytes' rate by a few us whatever its size: 6.6 MB 33.4 us for 28.2 at 232 GB/s,
    2.6 MB 16.2 for 12.0, 0.25 MB 2 us over (reckoned from the pack's headers).
- List the register count for every changed kernel instance with `-Xptxas -v` and inspect the instances the round launches.
- A state read and written once is worth more than the launch saved (spec 089): the DeltaNet replay after a
  round kept in part (31 us, 3 MB of state in and out) left to the next chain, which loads that state anyway
  and replays the kept rows in registers, costs that chain 6 us a launch on average. A weight-free copy saved
  (`place_kernel`, 1.07 us, 3.4 a DeltaNet chain) is its launch's floor and no more. Together 1.0 to 1.7% of a
  reply's time on the 3.05-bit pack; `topk` (a block a row) and the one-block grouping after it were not
  joined: a grid-wide hand-off costs a launch.
- An attention block of mma warps over one staged tile is bound by its warps' waits, not by the tensor cores
  (spec 093 WS-04, `glm5/mla.cu` `mla_chunk_kernel`, 512 keys of a 512-wide latent). `ncu` at 2 rows: 18.4
  cycles an instruction, 6.5 of them on shared memory's scoreboard, the tensor pipe at 17% of its peak. A
  row's 32 heads as two blocks of four warps, each staging the same 32 KB tile (50,688 B, so one an SM): 89 to
  105 us at 1 to 6 rows and 179 at 7, when the blocks pass 48. One block of eight warps over one tile
  (68,608 B, the same bits): 93 to 107, 111 at 7, a 512-row window 7.0 ms to 3.9. What shortened a launch was
  the loads: a thread copying an eighth of one key (one lookup of its row) and asking for the next tile's same
  lines (`prefetch.global.L1`, two a thread) while this tile is multiplied, and stores of two floats: 65 to 75
  us, 3.5 ms a window; from L2 60 to 68, so the asking closed the gap to L2 (25 to 40 us) to 5 to 18. Refused:
  chunks of 256 keys (60 to 90 with their merge, a window 6.0 ms against 4.8), two tiles in flight (no room
  under one block an SM), the score as four interleaved sums.
- Staging by `cp.async` is not shorter where a block's own chain is the time (spec 093 WS-04, `glm5/index.cu`
  `score_kernel`: 129 blocks of four warps, each 64 keys of 256 B against one row's 32 x 128 query, fragments
  read lane by lane from global memory, 8 KB of shared sums, 11 blocks an SM): 21 us cold and 8.8 from L2 at
  one row, 4.5 more a row. The same products with the tile staged (16 KB, and the query 8 KB or left in global
  memory; 32 KB, 3 blocks an SM), a block's rows in turn or a block a row: 24 cold, 16 from L2, 13.5 more a
  row, with or without the head sum. `ncu` on the staged form: 30.5 cycles an instruction, 19.7 on shared
  memory's scoreboard, issue slots 6% busy. A prefetch of a lane's keys before its first step: within the
  spread. Why the staged chain is longer: not resolved (spec 093 WS-04).
- Blocks that start and return are not free at thousands (spec 093 WS-04): a grid of a block a pool of the
  context, 16,384 for the one or two that worked, was 10.2 us against 2.1 for the grid of those; 768 empty of
  1,024 a row at 128 threads were 1.5 to 8 us of 22 to 48 at 1 to 6 rows.
- One thread walking a vector is a serial chain of loads (spec 093 WS-04, `index.cu` `write_kernel`): 128
  values read, summed twice and stored twice by one thread were 9.5 to 12.8 us; a thread a value into shared
  memory, one thread adding them there in order, a thread a store: 2.6, the same bits.
- A thread that adds each value as its load arrives keeps two loads in flight (spec 095 WS-04,
  `bf16/reduce.cuh` `b16_slices`, standalone): 32 split-K slices from L2 took 3.4 us; loaded eight at a time
  and then added in the same order (the same bits), 2.0. A round on the 3.05-bit pack 0.4 to 0.5%.
- Asking L2 for the next launch's weight from the kernel before it does not pay (spec 095 WS-04, standalone,
  the CUTLASS bf16 GEMMs of 2.6 to 31.5 MB after a 3-block kernel issuing `prefetch.global.L2` of 1, 2 or 4
  stages of 128 B every row and K slice): the GEMM is shorter by what the small kernel is longer, the pair
  level.
- Dirty lines are not written back while the device idles: they wait in L2 for the first kernel that streams
  past it, which pays them (spec 096 WS-02, standalone `pair_bench.cu`, 64 MB read before each rep). The
  DeltaNet chain's 3 MB state stored, then `out_proj` (31.5 MB bf16, CUTLASS slim, 4 slices): 162 us against
  145 with the store left out, and 162 after a 3 MB `cudaMemsetAsync` alone; a 4 us kernel or 10, 25, 50 or
  150 us of a spinning block between them changes nothing. About 181 GB/s for the writes. The store's operator
  does not move it either: `st.global.wt` (documented for system memory only), `.cs` and an `L2::cache_hint`
  evict_first policy left 14.5 to 17 us of it; evict_last hid it only because that bench wrote the same 3 MB
  every rep. A launch order cannot hide such a write: only fewer bytes written can.
