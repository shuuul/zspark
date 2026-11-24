# The GB10's specification sheet

Read on a DGX Spark (`cudaGetDeviceProperties`, `nvidia-smi`, `lscpu`, `/proc`, `/sys`). Part of the
`cuda-gb10` skill. A number here is the runtime's own; what was measured by a kernel is in `SKILL.md`,
"Measured bounds".

## Device

| Property | Value | What it means for a kernel |
| --- | --- | --- |
| Name, compute capability | NVIDIA GB10, 12.1 | target `sm_121a`; family `sm_120` |
| Driver, runtime, nvcc | 580.178.04; CUDA 13.0 (13000); nvcc 13.0.88 | |
| `multiProcessorCount` | 48 | a grid of 48 blocks, or a multiple, when a block fills an SM |
| `warpSize` | 32 | |
| `maxThreadsPerBlock` | 1,024 | 32 warps |
| `maxThreadsPerMultiProcessor` | 1,536 | 48 warps an SM: one block of 1,024 threads leaves 512 unused |
| `maxBlocksPerMultiProcessor` | 24 | |
| `regsPerBlock`, `regsPerMultiprocessor` | 65,536 both | 42 registers a thread at 1,536 threads, 64 at 1,024, 512 at 128 |
| `sharedMemPerMultiprocessor` | 102,400 B | the SM's, for all its blocks |
| `sharedMemPerBlock` | 49,152 B | without opting in |
| `sharedMemPerBlockOptin` | 101,376 B | with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` |
| `reservedSharedMemPerBlock` | 1,024 B | counted against the SM for each block |
| `l2CacheSize` | 25,165,824 B (24 MB) | a weight over 24 MB is cold every round; small buffers between launches stay |
| `persistingL2CacheMaxSize` | 18,874,368 B (18 MB) | an access policy window (persisting, 18 MB over a prompt's rows) made the NVFP4 experts 1.5x slower, and calls outside it too (spec 107 WS-03) |
| `accessPolicyMaxWindowSize` | 134,217,728 B | |
| `memoryBusWidth` | 256 | |
| `maxThreadsDim` | 1024, 1024, 64 | |
| `maxGridSize` | 2,147,483,647, 65,535, 65,535 | |
| `totalConstMem` | 65,536 B | |
| `totalGlobalMem` | 130,661,208,064 B | the machine's memory: see "Unified memory" |
| `integrated` | 1 | no separate VRAM; `nvidia-smi` gives memory.total and power.limit as N/A |
| `unifiedAddressing`, `canMapHostMemory`, `hostRegisterSupported` | 1 | |
| `managedMemory`, `concurrentManagedAccess` | 1 | |
| `pageableMemoryAccess`, `...UsesHostPageTables` | 1, 1 | the GPU can read pageable host memory through the host's page tables |
| `directManagedMemAccessFromHost` | 0 | |
| `concurrentKernels` | 1 | |
| `asyncEngineCount` | 1 | |
| `streamPrioritiesSupported` | 1 | |
| `globalL1CacheSupported`, `localL1CacheSupported` | 1, 1 | |
| `cooperativeLaunch` | 1 | grid-wide sync (`cooperative_groups::grid_group`) is available |
| `computePreemptionSupported` | 1 | |
| `hostNativeAtomicSupported` | 1 | |
| `memoryPoolsSupported`, `sparseCudaArraySupported` | 1, 1 | |
| `unifiedFunctionPointers`, `timelineSemaphoreInteropSupported` | 1, 1 | |
| `ECCEnabled`, `isMultiGpuBoard`, `gpuDirectRDMASupported` | 0 | |
| `clocks.max.sm` (`nvidia-smi`) | 3,003 MHz | idle draw about 6 W, P8 |

## What follows from the SM's numbers

- Blocks an SM by shared memory: floor(102,400 / (bytes + 1,024)), capped by 24 and by 1,536 threads. 51 KB or
  more: one. 33 KB: two. 24 KB: three.
- A block of 1,024 threads is alone on its SM by threads as well (1,536), whatever its shared memory.
- 48 SMs of 100 KB are 4.9 MB of shared memory in all: the most weight that can be on its way through
  `cp.async` or TMA at one time.
- The memory's ceiling, 215 to 235 GB/s, is reached from 80 blocks of 128 threads (two an SM) reading
  directly, or from 48 blocks with deep stages; 20 blocks of 128 get 85 GB/s.

## Instructions and features

`SKILL.md` has the table by dtype. In short: `mma.sync` for bf16/f16 (`m16n8k16`, as `sm_80`), FP8 (`sm_89`),
and the `sm_120a` kinds `f8f6f4`, `mxf8f6f4`, `mxf4`, `mxf4nvf4` with block scales (`m16n8k32`, `m16n8k64`),
sparse forms too; `cp.async`, `ldmatrix`; TMA (`cp.async.bulk.tensor`) without multicast and without sub-byte
types towards global memory; clusters of 1 in practice. Not here: `wgmma` (`sm_90a`), `tcgen05`/UMMA/TMEM
(`sm_100a` and the datacenter family), 228 KB of shared memory.

## Unified memory and the host

| | |
| --- | --- |
| CPU | 20 cores, aarch64: 10 Cortex-X925 (3.9 GHz), 10 Cortex-A725 (2.8 GHz); L2 25 MiB in all, L3 24 MiB in two |
| Memory | 128 GB LPDDR shared by CPU and GPU (`MemTotal` 127.6 GB; CUDA sees 130.7 GB total) |
| Pages | 4 KiB base; huge pages of 64 KiB, 2 MiB, 32 MiB, 1 GiB exist, none reserved; transparent huge pages on `madvise` |
| NUMA | one node |
| Kernel | Linux 7.0.0-1019-nvidia |

- The page cache and the CPU's allocations take from the same memory as the GPU's: check `free` as well as
  `nvidia-smi` before a large load.
- What a kernel writes goes back to that memory after it ends: the next kernel's reads share the bus with it
  (5 to 10 us on a 6.5 MB read after an 8 MB write, spec 070).

## Pages the CPU wrote last (spec 071 WS-10 and WS-12; probe `place.cu`)

Ordinary device memory, 40 MB buffers: 175 to 183 us, 230 to 238 GB/s. Pinned host memory, 128 MB: 203 to
213 GB/s.

- The pages the CPU wrote last are read at 285 to 290 GB/s (40 MB, 143 to 147 us): the buffer `cudaMalloc`
  returned last (its pages zeroed by the kernel, 24 GB/s), or pinned host memory after a CPU write. Tens of
  MB: the last 32 MB of 1 GiB, a whole 40 MB buffer; of 128 MB pinned, 47 to 54 us of 638 saved (227 to
  236 GB/s), spread over the buffer and not one region.
- Any CPU write does it: `memset` to 0 or 0xAB, a `DC ZVA` loop, plain 64-bit stores, `stnp`; a rewrite keeps
  it. A CPU read alone does not; a read after the write adds to it (236 GB/s after the writer reads, 251
  after all 20 read half, against 227 to 230). The kernel's zeroing was only the first writer found, not the
  mechanism.
- The writing core counts: one Cortex-A725 saves 7 to 9 us of 640, ten on disjoint slices 35 to 50, one
  Cortex-X925 47 to 54, ten 66, all twenty 61 to 76 — at 5 us a MiB read, 2, 7 to 10, 9 to 11, 13 and 12 to
  15 MiB at the least. The small cluster's L3 (8 MiB) fits; the big one's 16 MiB and the 49 MiB of all L2 and
  L3 are not reached; one big core gives nearly what ten do.
- It is a cache state, ended by `DC CIVAC` over the range from a core of the other cluster (no store issued;
  207 to 217 of 228 GB/s), by all ten, all twenty, and `DC CVAC` from all twenty; from the writer's own
  cluster `DC CIVAC` and `DC CVAC` leave it, with the lines counted. Flushing other addresses for 23 ms does
  not end it.
- Not a place: the same addresses are ordinary 100 ms later; a buffer at any offset of 2 MiB, 32 MiB or
  512 MiB, in either of the runtime's address regions, has it or not by when its pages were written.
- It ends when 32 MB more are zeroed (16 MB: half; 8 MB: nothing lost), whoever reads them or not: a later
  `cudaMalloc`, a host `malloc` with its first touch.
- It ends with time: once the device has read the buffer, 100 ms without the device reading any memory (80 ms
  begins); empty launches do not keep it, reading another buffer does. Never read, it survives 300 ms and not
  3 s. Read without pause it wears from 288 to 257 GB/s in 4 s.
- A device write over the buffer (`cudaMemset`) leaves it fast, the new bytes read back; write order among
  buffers changes nothing; zeros read no faster than ones once settled.
- The gain needs the ordinary path saturated: 20 blocks of 128 threads 86 against 85 GB/s, 40 blocks 165
  against 159, 80 blocks 265 against 218, 1,280 blocks of 256 284 against 241. Windows of 2 or 4 MB cannot
  show it (a launch's own time), 16 MB can.
- Not the GPU's L2: 16 MB read once is read again in 13 us (1,280 GB/s) and still after the CPU zeroes 64 MB;
  40 MB read between evicts it.
- Allocators, settled (64 MB windows): `cudaMalloc` 215 to 219 GB/s, `cudaMallocAsync` 214 to 216,
  `cuMemCreate` and `cuMemMap` 215 to 216 (granularity 2 MiB, minimum and recommended), `cudaHostAlloc` 202
  to 205, `cudaMallocManaged` 161 to 175, pageable `mmap` read through the host's page tables 160 to 173 with
  or without `MADV_HUGEPAGE`. None is a lever.

So 285 GB/s is not what DRAM delivers to the device: it is DRAM's 230 to 238 GB/s with a cache answering part
of the reads beside it (40 MB in 146 us against 177 needs 7 MB or more answered at no cost to memory's path),
and a timing that shows it is flattered. Which cache is not established: dirty lines in the big cluster's L3
answering the device's snoops fit the small cluster's amount and the read-back's gain, but a flush from that
cluster would then end it; a cache past both clusters that takes lines written whole fits the flushes and the
device's overwrite, not the read-back's gain, and nothing shows its size. Idle states are not it (sysfs
`CoreIdle-OFF+PlatIdle-DSUOFF`; 17 entries on cpus 10-19 in 40 ms left it fast). The root probes that would
settle it (`dmesg`, the caches' and the interconnect's counters, page frame numbers; the SMMUs' counters
`smmuv3_pmcg_*` count translations only) were not run — not resolved (spec 071 WS-10 and WS-12).

- Tens of MB of the pages the CPU wrote last (the last 32 MB or so of a `cudaMalloc`, whose pages the kernel
  zeroes; pinned host memory after any CPU write, zeros or not) read 17 to 25% faster, 285 to 290 GB/s against
  230 to 238, until the device has read no memory for 100 ms or 32 MB more are zeroed. It is a cache answering
  part of the reads, not DRAM and not a placement: a `DC CIVAC` over the range from a core of the other CPU cluster ends
  it; no address, alignment or allocator has it, and 46 GiB of weights cannot. A standalone timing allocates
  64 MB after its weights, or its last copies flatter the kernel; waiting alone does not clear them. Which
  cache is not known (zspark spec 071 WS-10 and WS-12).
- A later probe (spec 097 WS-05, a 512 MiB buffer read in 8 MiB slices) found the last 8 to 32 MiB at 233 to
  249 GiB/s against the body's 200 to 205, and the tail still fast (234) with only the first 256 MiB written,
  the freshly written region slower (120 to 178): "the allocation's tail, not the write's residency". Not a
  contradiction of 071: an unwritten tail is the pages the kernel zeroed last at `cudaMalloc`, 071's own case;
  097's writer was the device, which 071 never found to count. Left as found.

## Not read or measured yet

SM clock under load, the FP8 and FP4 `mma` peaks, the L1's size, the GPU's TLB reach and which page sizes its
mappings use, whether an L2 persistence window helps a small hot weight.
