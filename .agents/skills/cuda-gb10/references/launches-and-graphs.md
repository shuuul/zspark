# Launches, clusters and graphs on the GB10

What launches, clusters, waves and captured graphs cost, from the hand-off inside a launch to the small
kernels of a model. Part of the `cuda-gb10` skill.

- A grid can hand a value from some blocks to others inside one launch (work taken by `atomicAdd` on a counter
  in device memory, a done counter polled by `atomicAdd(p, 0)`, the value read back by `cp.async.cg`): an
  empty launch of 48 blocks of 1,024 threads doing only that is 2.8 us (spec 070 WS-04).
- Clusters cost here. A cluster of 8 whose blocks each hold 90 KB ran in 31.8 us against 21.5 without the
  cluster (the eight do not start together — why, not resolved). With small blocks the cluster itself is
  free, but the meeting (`cluster.sync` twice, sums read through DSMEM) is 6 to 7 us of a 25 us launch: K
  slices as warps of one block meeting in its own shared memory after `__syncthreads` give the same bits in
  18.3.
- A program that starts, reads one flag and returns is about 5 ns of a launch's duration: 88 empty of 160 at
  16 rows were 25 us of 700 at most, and a grid that walks only the tiles in use did not win them back (the
  walking kernel was 5% slower at the same grid; spec 071 WS-11).
- A grid of one block a program has no warp tail to even out; what it has is waves. At 3 blocks an SM (144
  resident) each wave costs: `joined_down_kernel` (20 blocks an entry, 168 registers) takes 2.3 us an entry
  and 5 to 7.5 us for the entry at which 20 x entries passes a multiple of 144 (entries 15, 22, 29, 36, 44;
  TensorFold's `grouped_kernel` the same). A last wave of n blocks takes about 4.4 + 0.115 n us (20 blocks
  6.6 us, 144 blocks 21): a block's own chain is the floor, so by that fit (no such kernel was built) the
  same programs in whole rounds over fewer resident blocks take the same time, and only fewer waves (more
  blocks an SM) would save it. At 4 blocks an SM with 8 programs a block of columns
  (`joined_narrow_gate_up_kernel`, 192 resident) the crossings do not show: 5.5 us an entry straight through
  (spec 073 WS-09, `ZSPARK_TRELLIS_PROFILE=units`). Refused: the same kernel under `__launch_bounds__(128,
  4)` (128 registers, 4 blocks an SM) — the steps move to the multiples of 192 and stay as high, and between
  them it is 2.5 to 4.5 us slower, 1.5% slower over a reply's calls, 5 to 13% from L2. Not tried: a block
  that walks programs with its loads in flight.
- Small kernels (a model's glue: norms, write-backs, gates), spec 073 WS-03, Flash Next, each a kernel's own
  duration in a trace:
  - Weigh a kernel by the shapes real use gives it before tuning it. With a drafter on, a reply's rounds are
    of 2 to 6 rows and 1 row is 1 round in 238; a kernel tuned at 1 row is tuned for a shape that does not
    occur.
  - The floor of a launch is 1.0 to 1.2 us (10 to 40 blocks of 256 threads doing a line of work); inside a
    captured graph the gap between two kernels is 0.13 us, so a launch merged away saves its floor, not a
    gap.
  - A scalar 4-byte store a thread sends a 32-byte sector several times: a 3.07 MB fp32 state written a float
    at a time was 12.2 MB of sectors written (ncu `lts__t_sectors_op_write`), 3.5 MB read. Storing rows of
    four floats (`float4`, 16-byte aligned) took 1.5 to 2.5 us off a 33 to 78 us launch, the same bits.
    In the pool's DeltaNet kernels (spec 096 WS-01, Flash Next NVFP4, same replies): the chain 97.7 to 79.2
    us a launch that runs (-18.5% over the load), the replay of a slot's kept rows 64.6 to 60.8 (-6.9%),
    registers unchanged (64, 62): a replay's time is its rows' loop (two barriers a row), not its state's
    bytes.
  - One block a row with a barrier a step loses to many small blocks at few rows: a row's 4 streams in one
    block of 640 threads (9 `__syncthreads`) took 5.7 us at a row where two launches of 10 and 40 blocks of
    64 threads took 4.3 together; a block a stream (3 barriers, 4 blocks a row) took 2.3. Split the grid
    along whatever axis is independent until the SMs are in use, even if a block then re-reads a small
    input.
  - That re-read is only free in L2: four blocks a row each summing 112 KB of slots were faster to 8 rows and
    2.7 times slower at 512 rows (57 MB read four times). Dispatch the form by the bytes, not once.
  - The first launch after a memory-bound writer pays 0.7 to 2.6 us whatever it is (a 2.3 us write-back was
    3.7, one of 3.1 was 5.7, a 2.3 us joined site 3.0, after a launch that wrote 5 MB); the launch after that
    one is clean. A merged launch pays it once, as its first half did: time a candidate after such a
    neighbour, not only alone.
  - A kernel that sums K slices of a product costs by the slices: 1.2 us on one slice, 3.5 on 32 (the
    read-out's act). A profile case without them reads as a 3x slowdown "in the round".
  - An attention block of 4 warps that reads 64 KB a tile is bound by what one SM pulls, about 20 GB/s: 2
    blocks over 1.2 MB took 30 us, 8 blocks over 4.7 MB 38 (ncu: 71% of stalled samples on the long
    scoreboard, the busiest SM at 19% of its issue rate). More blocks is the lever; splitting the keys
    changes the sums' order.
  - That lever, measured (spec 073 WS-08, `attention_tiles.cu`, kernel and merge a call): a 64-key tile a
    block against a chunk of eight a block is 32.5 us to 12.5 at 1 row of 512 keys, 33 to 24 at 8 rows, 47 to
    32 at 1 row of 2,051, 49 to 45 at 2; it loses from about 96 tile-rows (192 blocks of 43.8 KB, 2 an SM, so
    two waves): 36 to 37 at 16 rows of 512, 51 to 59 at 4 of 2,051. FP8 blocks (28 KB, 3 an SM) hold to 144
    tile-rows. Past the waves the SMs have the memory's rate already and the partials are extra bytes (24 KB
    a tile a row written, then read by the merge).
  - A per-thread chain of loads behind a branch is serial memory latency: a merge of 33 partials a value was
    14 us at ONE row (0.2 us a partial) with two `expf` and two loads a partial a thread; the factors found
    once a (row, head) in shared memory and a chunk's eight values loaded before they are mixed, 4.5 us.
  - Selects are not free where a warp's lanes differ: `kind == 0 ? a : kind == 1 ? b : fma(..)` on 32 values
    a thread a tile cost 1 us of a 31 us launch (8 tiles), and a branch instead cost 4. The factors (1, 0)
    and (0, 1) through the one `fma(mine, a, part * b)` cost nothing measurable; one `expf` in place of two
    paid for the extra multiply.
  - A captured graph freezes the grid: a launch sized by a context bucket (8,192 and more) is sized for the
    longest list, so a form chosen on the host by the held length is chosen for the bucket. The choice by a
    row's real length has to be made on the device, inside the frozen launch. Done for the attention by
    tiles (`tiles_a_block`): the grid is the finer form's, every block reads the rows' positions and the
    blocks the coarser form does not need return. 7 of 8 of 528 blocks returning cost 0.4 us of a 55 us
    launch (8 rows of 2,051 keys); the merge must switch form too, or it loses 1 to 3 us (a block merge on 5
    partials a row against a thread a value).
  - A pass of about 20 small eager launches (the pool's MTP head pass, embedding to the head layer's last
    launch) captured as a graph keyed by rows and context bucket: 1,256 to 1,171 us median from embedding to
    projection, about 80 us a pass, against 14 to 26 us between loads of one side (four streams).
    About 0.5% of decoding at four streams: a trace resolves it, a
    whole-load tok/s does not.
- A sum a later kernel would make is cheapest where it already exists (spec 073 WS-07): a site that read 11
  slots of fp32 a row and summed them took 4.4 us at one row, 9.7 at 8 and 495 at 512; reading the one row an
  earlier 1.6 to 2 us launch had summed in the same order (the same bits) 2.4, 5.6 and 240. The bytes a small
  kernel reads are its time even at one row.
- A hand-off between the blocks of one launch is not cheaper than a small launch after it: the blocks of a
  row's tile each `__threadfence` and `atomicAdd` a counter, the last one sums 11 slots read back with
  `__ldcg`. That cost a 40 to 130 us trellis kernel 2.5 to 3 us at 1 to 4 rows, 6 at 8 and 10 at 16, against
  a separate launch of 1.6 to 3.1 us (spec 073 WS-07, `experts_joined_narrow.cu`). The 2.8 us empty hand-off
  above is the floor.
- Two Sparks: a host message costs some twenty gathers (spec 095 WS-00, Flash Next on two ranks, one traced
  load a rank). A frame on a TCP socket over the ConnectX link (`TCP_NODELAY`, a blocking read) reaches the
  other Spark 0.1 to 0.8 ms after it is sent (median 0.4 over 2,800 passes, the two traces set by the
  gathers' ends), and `ping` there is 0.1 to 1.5 ms a round trip; an NCCL all-gather over the same link
  (RoCE, 4 channels, inside a graph) moves 156 KiB in 24 us, 496 KiB in 125, a decoding window's partials (a
  72 to 123 KB buffer zeroed before each) in a median 11 us. What the other rank must know before a
  collective belongs on the device or in the collective, not the socket.
