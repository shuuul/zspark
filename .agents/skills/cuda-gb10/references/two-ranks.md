# Two Sparks as two ranks: the link and what crosses it

What the ConnectX link between two DGX Sparks gives, what zspark sends over it in a window and a round, and
which overlaps pay. Part of the `cuda-gb10` skill. A rank a Spark, NCCL over RoCE; rank 0 leads and tells
rank 1 each operation. The host-message and in-graph gather latencies are in
[`launches-and-graphs.md`](launches-and-graphs.md); loading over the link in [`loading.md`](loading.md).

## The link's rate

| Probe | Figure | Source |
| --- | --- | --- |
| Send and receive both ways at once (`Gather.scatter`'s form), NCCL 2.32.3, both rails, 4 channels, best of 20 | 20.3 GB/s a direction at 5.2 MB, 22.4 at 16.8, 22.8 at 33.5, 22.9 at 67 | 116 WS-03 |
| Send alone, 33.5 MB | 23.2 GB/s | 116 WS-03 |
| All-gather / reduce-scatter, the same sizes | 18.7 to 20.6 / 17.9 to 20.0 GB/s | 116 WS-03 |
| One rail (`NCCL_IB_HCA=rocep1s0f1`) | 12.7 GB/s | 116 WS-03 |
| `ranks-probe` medians, all-gather a rank | 23.1 us at 16 KB, 30.8 at 64 KB, 34.5 at 128 KB, 84.3 at 1 MB, 1.00 ms at 16 MB, 1.85 at 32 MB | 045 WS-01 |
| Both rails and 4 channels against one rail | 1.4x from 64 rows up (2,048 rows 1.85 against 2.6 ms), the same below; against NCCL's own channel count the same time at 8 rows for 439 MiB against 797 a Spark | 045 |
| `spark_copy.sh` (dd and nc, direct I/O, 1 GiB ranges) | 1,585 / 3,095 / 4,952 MB/s at 1 / 2 / 4 streams, 5,670 at 8, 7,803 at 16; rsync over ssh 0.49 GB/s | 022 |

- **The ceiling is about 23 GB/s.** The two rails are PCIe Gen5 x4 each, about 13 GB/s alone and 23 to 24
  together, under the 200 Gb/s port's 25. Nothing in NCCL's environment moved it: 2, 8 or 16 channels, NCCL's
  default, `NCCL_NCHANNELS_PER_NET_PEER=4` (23.6 at 33.5 MB), `NCCL_IB_QPS_PER_CONNECTION=4` with split data
  (23.2), a 16 MB buffer of 512 KB chunks (23.0). No setting is changed (116 WS-03).
- **A trace's per-collective rate overstates the link.** Spec 111 read 30 to 35 GB/s from a trace by each
  collective's shorter duration of the two ranks'; the probe without a model puts the link at 23. Take the
  probe's. Spec 114 WS-08's trace read 25 GB/s for 33.5 MB scatters the same way.
- **Small gathers, three figures**: 045's probe 34.5 us at 128 KB (eager, 41.3 captured); 095's in-graph 156
  KiB in 24 us; 098 WS-01's 156 KiB scored row 34 us inside the pass graph (4,463 gathers, mean 34.2). Take 34
  us for a 156 KiB row in a served graph; 095's 24 was one trace. Under 0.1% of a 55 ms round either way.
- **A gather's time is mostly the peer's lateness**: GLM rank 0 87.3 us against rank 1's 39.7 (093). "RING 86
  us against TensorFold's RING_LL 31" (091) was two NCCL versions' kernel names (2.32.3 has only
  `AllGather_RING`; TensorFold ships 2.30.7); `NCCL_PROTO=LL` changed nothing (85.7 to 83.4 us).

## What crosses it

- **The world count is part of the bits.** Partials are added rank 0 first, so a two-rank reply is not one
  Spark's: the reference for two devices is the loopback, not an unsplit forward (045). Flash Next on two
  ranks leaves vLLM's primes reply at token 19 where vLLM's own first choices are 0.31 apart; its drafted,
  sampled, pool and held-state replies are its own serial ones in bits (105 WS-04).
- **Gathers are in place** (`sendbuff == recvbuff + rank * sendcount`), fp32 [world, rows, dims]; no partial
  copied (045).
- **A decoding round**: GLM's 92 all-gathers a round of 2 rows took 3.5 ms (rank 1 2.4), a call p5 6 us,
  median 17, p95 74 on both ranks: what one waits for is the other's jitter and the round's start (114 WS-01).
  A verify window's rows settled on the device (each rank picks the start of its half of each row, one gather
  of some 4 KiB a row) replaced a 155 KiB gather, a synchronize and a channel message a row (114 WS-02).
- **Host hand-offs on the GB10**: `cudaMemcpyAsync` from pageable memory is a synchronous copy here
  (`pageableMemoryAccessUsesHostPageTables` 1). Per-round words written by a launch that carries them in its
  arguments (`common/words.cu`), the follower polling its channel 200 ms before it blocks, and the picks
  digested one window behind from a pinned copy took rank 1's idle 4.2% to 3.5% of a trace and the replies
  1.025x (114 WS-02).
- **A prompt window**: from 256 rows each rank writes back and reads out its own rows only
  ([`chunking.md`](chunking.md)). Flash Next's 16k prompt, 8 windows of 2,048 rows (111 WS-02): the link 460
  ms, one rank waiting for the other's compute 221 to 285 ms, 50 to 78 ms of host before each window (the PLE
  rows, gathered ahead since: 16k 2539-2616 to 2775-2783 tok/s). The waits are the routed experts' rows
  landing unevenly on the halves; no overlap hides them.

## A decoding round on two ranks

- **Tell the follower nothing it can derive.** Rank 1 waited for rank 0's socket message 2.9 ms a prose round
  (6.0%), 6.1 a code round (9.4%), a median 399 us late; ConnectX sockets are slow (64 B round trip 932 us
  blocking, 460 spinning; 093). Rank 1 now derives the drafts and the greedy accept itself, with one NCCL word
  a round (u32 [2,48]): lag 3 us, prose 49.7 to 52.1 (+4.8%), code 91.5 to 100.4 (+9.7%), bits equal (095
  WS-01, f4318ce). On GLM the same idea showed 3.8 ms traced and nothing untraced (29.18 against 28.89):
  confirm a traced wait with an untraced pair ([`measuring.md`](measuring.md)).
- **Split the head as TensorFold does** (scheme B) rather than whole on rank 0: 2.0 to 11.8% faster a request
  (090 WS-04, 24731ec); a host model had preferred whole because "a gather is not free".
- Fits: one rank 50.4 ms + 2.54 a row, a two-rank share 31.2-32.5 + 0.82-1.11 a row; the 97 gathers a round
  do not grow with rows (090 WS-00, reckoned).
- GLM's two-rank hall stays eager: a graph over NCCL must be told to the follower (099 WS-03, cost not
  measured); Flash Next's two-rank pool captures its graphs at load (098 WS-01).
- The pool, prose total tok/s, one rank against two (094 WS-04): 33.5/48.2 (1.44x) at 1 stream, 52.1/73.9
  (1.42x) at 2, 83.3/104.7 (1.26x) at 4; lists 1.17x.
- Dead memsets before in-place all-gathers: nothing timed (0.66 to 0.70 ms traced; 095 WS-02).

## Overlaps that pay, and ones that do not

- **Hiding a layer's collectives whole needs a window cut into parts**, each part reading the experts again:
  each layer's collectives sit between kernels that need their result whole (a share before the block reads
  every row, a scatter before the next read-out). 4,096-row windows took 0.37 s off a 16k prompt against
  2,048, and the link a part could hide was at most 0.46 s: not built (spec 111).
- **The scatter beside the rank's own rows pays** (116 WS-03): from `split_rows` a rank makes the other's rows
  of its partial first, sends them on a non-blocking stream of the device's highest priority, makes its own
  rows meanwhile and joins the exchange's event before the glue. Each part is the same kernel on a span of
  rows, so the bits are unchanged. Standalone, 33.5 MB: beside a compute-bound kernel (1.39 ms) the exchange
  takes 1.65 ms for 2.86 serial; beside the memory-bound routed combine (1.44 ms) 2.68 for 2.97, the bus
  shared. A 9,137-token prompt's first token 6,110 to 5,857 ms (4%). TensorFold overlaps the same exchanges
  in 3 pieces on a side stream; vLLM and SGLang do not overlap them.
- **The partials bf16 pay, a quantized window's only** (116 WS-04; vLLM and SGLang reduce bf16, TensorFold
  fp32): the side stream's scatters 526 to 301 ms, the main stream's wait 228 to 128, after the combine 65 to
  4; first token 5,195 to 5,032 ms. About 0.2 s at 16k is still exposed behind the out-projection. Steps,
  verify windows, the MTP layer and DFlash2 keep fp32.
- Both with the rest of 116: a 16k cold prompt 9.1-9.3 to 7.6 s ([`chunking.md`](chunking.md)).
- **A told op a row is dear**: the pool's sampled acceptance (an `Op.row`, a 496 KiB all-gather, a
  synchronize and a download a row on both ranks) takes 0.4 to 0.8 ms a row on rank 0, where the link moves
  the bytes in some 25 us: rank 0's gather waits for rank 1's host. 11 rows a 4-stream round are 6.7 ms of 78
  (144 WS-01). The eager head pass (pool, two ranks) costs 1.42 ms against the pair's graphed 1.38: a graph
  saves little there.
- **A told frame pays the idle wake** (146 WS-05, `ranks-probe`): a 1 KiB `Said` all-gather takes 27 us back
  to back and 226 us (p90 418) after a 1 ms host gap: NCCL's proxy thread sleeps once it holds no op, and its
  core leaves LPI-2/3 in 231/433 us. A `SCHED_IDLE` spinner on that core made every frame wait for the 1 ms
  tick — worse in every case; the other levers (`/dev/cpu_dma_latency`, `pm_qos_resume_latency_us`) need
  root. Fewer frames a round is what is left.
- **Head passes graphed on two ranks lose** (146 WS-04 a): both ranks capture the pass, the partials' gathers
  inside, 1 stream -1.3% (six pairs down): the graphed pass is longer than the eager one, more than the
  device spacing it saves.
- **`cudaMalloc` bursts cost idle GPU** (146 WS-04 b): a kept state's 73 + 13 buffers each their own
  `cudaMalloc` cost 8.75 calls and 416 us of idle GPU a 4-stream round, 378 calls (22 ms) at a reply's end
  with no spares; one block a state (`cuda.cut`) and spares refilled while a round runs: 0.06 a round, +0.7%
  at 4 streams.

## What a second Spark buys

| When | Finding | Source |
| --- | --- | --- |
| 2026-10-13 | Flash Next NVFP4, depth 0 on two ranks: 28 tok/s against one rank's 36 to 88; cold prompts level (16k 1,598 against 1,561 tok/s); 41,131 MiB a rank against 80,171. Then: memory, not a stream's speed | 050 WS-07 |
| 2026-10-12 | One share alone against the whole model: prose 1.54 to 1.56x, code 1.47 to 1.51x, a 2,048-row prompt window 1.34x; with gathers estimated 1.33 and 1.22 | 050 WS-03 |
| 2026-10-14 | Flash Next NVFP4 drafting, two ranks against one: prose 50.2-50.7 against 35.7, code 91.6-93.2 against 73.6; prompts at 500 tokens 1,362-1,407 against 1,156-1,182 tok/s, at 4,000 level | 090 WS-03 |
| 2026-10-12 to 10-16 | Two whole copies behind a router: 146.8-157.2 tok/s for 8 streams against 78.3-81.3 for 4 on one; the hop about 80 ms on a second turn | 088 |
| 2026-10-18 | Flash Next FP4-Local with drafts: every reply 70.4 tok/s against one Spark's 62.0 (1.14x), 44,276 MiB a rank | 105 WS-04 |
| 2026-10-18 | Two ranks' cold prompt past one Spark once its windows were one Spark's width: 16k 2775-2783 against 2068 tok/s | 111 |
| 2026-10-28 | A path that serves several streams lags the lone path's optimizations: the pool's prompt round was the pair before 106, 111 and 116 (whole fp32 all-gathers, PLE rows gathered with the device idle, attention over extents), 1.29x the pair's first token. Ported cause by cause: 16k 7.01 to 5.48 s. Port each lone-path win to the round too | 149 WS-01, WS-02 |
| 2026-10-28 | Host side of a prompt window: per-row `madvise(WILLNEED)` takes the mapping lock (27 ms of 32,768 rows, no faster on 8 threads); `posix_fadvise` on the file's descriptor and a gather in 8 parts: 46 to 9.5 ms warm | 149 WS-05 |
| 2026-10-28 | The pool at 4,096-row rounds: 5-10% off 4k-32k, but +11.4 GiB a rank leaves the host 30-36 GiB and 64k turns unsteady (rounds 1.17 to 1.8 s, mapped pages read again); width costs host memory before it costs time | 149 WS-06 |
| 2026-10-28 | A serve on the LAN port takes other clients' requests through the router: time on loopback | 149 WS-03 |

## Sources

Specs 045, 050, 088, 090, 091, 093 to 095, 098, 099, 105, 111, 114, 116, 144, 146, 149 (WS-03 and WS-04's
entries in commits 9d0d30d8 and 33edafaa); `docs/recipes/qwen3.8-flash-next-fp4-local.md` ("On two ranks"),
`docs/recipes/glm-5.3-flash-fp4-local.md`.
