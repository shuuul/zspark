# Loading a checkpoint on the GB10: the standard pipeline

The owner, 2026-10-20: every model's load follows this pipeline. A new model, or a load that changes, is built
to it and measured against it. GLM-5.3-Flash FP4-Local on two ranks went from a 37 s share (47 s in a safe
build) to 9.4 s, and ready from 40 to 11.6 s, with the same loaded bytes (spec 116's close and its load work:
e61646d2 to 5d4b5332, c938af51; `--weights-from-peer` 132cfd04 to 4881c445).

## Measure first

`ZSPARK_LOAD_PHASES=1` prints a load's phases: the reader's read and buffers, and the loader's wait,
`cudaMalloc` count and time, device wait, copy and host work. Then the session's steps after `main` began.
Find the critical path before changing anything. Measure in the build that is served: `-Doptimize=fast`, the
default since 131fd5a3. A safe build cost GLM's load 10 s in byte-wise host loops (bounds checks), and
`zspark version` shows the mode (`0.0.0+fast`).

## The pipeline

1. **Read with O_DIRECT, threaded, into two stage buffers kept for the whole load** (`safetensors.Stage`). The
   loaders read tensors through a view of the stage. Don't read over the file's own mapping and then map it
   back: putting 113 GiB back cost 3.3 s of `mmap`/`munmap` under the load's memory pressure (compaction
   stalls, THP).
2. **One reader thread runs a part ahead.** While the loading thread packs part N, the reader reads part N+1
   into the other stage. Two parts at most are held. An event keeps the reader from overwriting a stage that
   the GPU still reads.
3. **Unpack on the GPU, from the staged host bytes.** On GB10 the GPU reads pageable host memory through ATS,
   so scale decode, packing and column cuts run as kernels that read the stage and write device buffers, with
   no host copy of the weight in between. (A host-side multithreaded unpack over experts once took 18 s to 7
   s; on the GPU it left the critical path.) Any device-side transform needs a byte-compare test against the
   host path on real layers.
4. **Read only this rank's bytes.** Partial tensors are read as contiguous ranges. Where a rank's half is
   strided (down's columns), read the contiguous half and swap the rest with the peer over NCCL (about 288 MiB
   a layer each way at about 22 GiB/s). GLM's reads went from 113.6 to 90 GiB a rank.
5. **Overlap everything that does not depend on the weights.** Start reads during the ranks' NCCL join. Draft
   heads' host widening runs on threads. Caches and graphs are made right after the last part, or during the
   load where `create` allows.
6. **One checkpoint for two ranks** (`serve --weights-from-peer`): rank 0 reads both ranks' ranges once and
   sends rank 1's part over the link while it reads the next one. Rank 1 builds its checkpoint from rank 0's
   headers, with PROT_NONE maps, so an unstaged read faults rather than opening a file. Bound by rank 0's
   disk: 165 GiB in 15.7 s, a share in 18.4 s.

## Measured rates

| What | Figure |
| --- | --- |
| NVMe, O_DIRECT, 8 to 16 threads, whole shards | 9.5 to 11 GiB/s; `dd iflag=direct` 10.5 GB/s; a buffered `dd` 1.2 GB/s |
| Host to device copy (`cudaMemcpy`, pageable stage) | about 48 GiB/s |
| `cudaMalloc` | proportional to bytes, 26 to 28 GiB/s. A pool saves nothing, and it serializes with `cudaMemcpy` from another thread |
| The link, NCCL over ConnectX, two rails | about 22 to 23 GB/s |
| `munmap` on an idle machine | 0.05 s; under a load's memory pressure, per-span remaps of 113 GiB 3.3 s |

## Not available or refused

- **No reading straight into the device allocation**: `cudaMalloc` memory is neither CPU- nor IO-reachable.
  Pinned host memory used as device memory costs about 6% of decoding's read bandwidth.
- **No GPUDirect Storage**: `nvidia_fs` is absent, and cuFile would only run in compatibility mode (a host
  bounce).
- **Giving parts back on a separate thread loses**: 16.8 s against 15.4, contention on the kernel's mmap
  lock.
- **Recycling the old part's pages with `mremap` loses**: unmap fell but mapping rose, and the copy slowed
  1.8 to 6.2 s.

## Flash Next FP4-Local on the pipeline (branch `fn-load`)

The same pipeline, shared code (`weights/loading.zig`: `Slot`, `Ahead(Loader)`, the report; GLM's `pack.Each`
and qwen4's `model_load.Reading` are its two loaders). What differs from GLM's, measured with
`ZSPARK_LOAD_PHASES=1` and `serve --context 262144`, the replies' shas unchanged:

- Before: one Spark read its 68 GiB over the mapping in 7.65 s, then `cudaMalloc` 2.87 s, the host's A16 pack
  2.10 s, the copy of 71 GiB 1.46 s: load 15.9 s, ready 17.3. Two ranks read the whole 68 GiB each, and
  `take` copying down's columns on the host cost about 3 s of a rank's 3.5 s of host work: a share 14.4 to
  14.7 s.
- The reader a part ahead alone: load 8.6 s, ready 10.05, the loader then level with the reads.
- **Place the A16 experts by the device** (`zs_nvfp4_experts_pack_a16`, `pack`'s 144-word blocks): the copy
  falls to 8 GiB and the pack to 0.08 s; load 8.2 s, ready 9.6, read-bound (waited for reads 2.9 s).
- csf stays a host decode (0.24 s a load, 0.16 a rank): Flash Next's csf is `weights/csf.zig`'s fixed and
  exceptions format, not `csf_32x2_v1`, so `zs_nvfp4_csf_decode` does not read it. The decoded scales live in
  `Staging.keep`'s two arenas, a layer's kept until the layer after next.
- A rank: gate and up by its rows, down whole and its columns read a row's stride apart by the placing kernel
  (no host copy, no swap: Flash Next cuts down by inputs, its outputs whole). 49.4 GiB a rank; the first two
  parts read during the join (1.0 s). A share 5.2 to 5.6 s, ready 7.3 (from 16.5).
- 16 reader threads in place of 8: 9.45 GiB/s against 9.17 on one Spark, within a load's spread; not kept.
- Down's halves swapped over NCCL (branch `load3`): 39.8 GiB a rank, but no gain until the reader's threads
  followed the spans' size (below): a share 5.2 to 4.3 s, ready 8.5 to 7.8. The MTP head is a staged part
  after the head, placed by the device; one Spark reads its first parts during setup (ready 9.4 to 9.1,
  loaded 8.7 to 8.5).

## The floor

A rank's bytes at the disk's 10.5 GiB/s: GLM 90 GiB, about 8.5 s. After that the loading thread's
`cudaMalloc` (3.1 to 3.3 s for 1,677 calls) is the next bound, once reads fall under about 6.5 s. Fewer calls
would not remove it: the cost is the bytes (below). Only overlapping the allocation with reads (it already
runs under them) or allocating less hides it.

## Allocation and page pressure (branch `glm-load`)

- **`cudaMalloc` costs bytes, not calls**: 86 GiB in 1,656 calls 3.36 s, in 46 calls 3.22 s (about 26 GiB/s);
  1,600 calls of 4 KiB 0.002 s. VMM (`cuMemCreate` + `cuMemMap` + `cuMemSetAccess`) is no faster: 40 GiB 1.37
  s against 1.385. One allocation a layer would save about 0.15 s: dropped (probes `mb.cu`, `vmm.cu`).
- **An allocating thread serializes with another thread's copies and kernels**: 1.79 s together against 1.385
  + 0.38 apart (`ov.cu`). An allocation moved to a helper thread overlaps nothing on the device.
- **Lay and fill a stage before the shares, not after**: fresh host pages are dear under a load's pressure —
  `cudaHostRegister` of a 2.18 GiB stage (THP faults) took 5.5 s on rank 0 with some 95 GiB taken, 0.3 s at
  the session's start; reading a 2.18 GiB file through its mapping 1.7 to 2.3 s against 0.24 by direct reads.
- **Page cache counts**: a download's 57 GiB of cache on the same machine made a two-rank load 41.6 s against
  23.6 with it evicted (`posix_fadvise(DONTNEED)` on the files, after `sync`; no root needed). Look at
  `free`'s buff/cache before a timed load.
- **Host work beside the reader stalls it**: a head widened on the loading thread between parts (0.75 to 2.95
  s) delayed the next part's read, the share 21.7 s against 19.5. Reading rank 0's own ranges beside rank 1's
  on a second thread gained nothing (16.6 s against 16.9): one disk.
- **Small spans want more readers** (probe `rd.c`: O_DIRECT spans as `Stage.fetch`, one shard of Flash Next,
  alternated): spans of 456 KiB (a rank's halves with down's rows halved) 6.3 to 7.1 GiB/s on 8 threads, 9.3
  to 9.7 on 16, 9.4 on 32; spans of 880 KiB 9.1 to 9.7 on 8, 10.0 on 16 and 32. `Stage` reads on 16 below a
  mean span of 640 KiB, on 8 above (GLM's 2.3 MiB). GLM's two-rank load on 16 was confounded by 27 GiB of
  page cache (measured 18.0 s); on 8 with the cache evicted, 9.1.
- **Touch a host buffer before `cudaMemcpy` writes it** (probes `d2h*.cu`): copies from the device into fresh
  pageable host pages now and then crawl — 473 MiB (a GLM draft head's bf16 rows) 3.7 to 4.3 s, about 0.13
  GB/s, 1 in 16 copies outside a load and every one inside GLM's; into pages touched first (`memset`, 0.094 s
  for 473 MiB) or `MAP_POPULATE`d, 54 GB/s, 0 in 16 slow. GLM's draft heads 2.8 to 0.43 s a rank.
