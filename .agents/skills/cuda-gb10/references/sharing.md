# Several processes on one GB10

What the chip and driver 580.178.04 give processes that share the GPU, read and measured for zspark's stack
(Flash Next, Unlimited-OCR and Qwen3-ASR as three `serve`s). Part of the `cuda-gb10` skill.

## By default: time slicing, about half each

- Compute mode `Default`, MIG `N/A`. Two processes with work are time-sliced: each gets about half the GPU
  beside another whatever its size. Flash Next's decoding (53 tok/s alone at `--parallel 2`) 22 beside OCR's
  pages, 21 beside ASR's segments, 14 beside both; OCR's pages and ASR's segments 2x their time beside it
  (spec 139). An idle neighbour costs nothing.
- `nvidia-smi compute-policy --set-timeslice 0..3` (DEFAULT, SHORT, MEDIUM, LONG) is listed on GB10 but needs
  root ("Insufficient Permissions" as the user); not tried. A slice's length moves the switch's cost, not the
  half each process gets.
- Stream priorities (`cudaDeviceGetStreamPriorityRange`: 0 least, -5 greatest) order a process's own
  streams: the Programming Guide's "Stream Priorities" (the `cuda` skill's
  `cuda-programming-guide/advanced-host-programming.md`) speaks of the scheduler choosing among pending streams
  and says nothing of other processes, whose contexts are time-sliced. Green contexts (`cuDeviceGetDevResource`,
  `cuGreenCtxCreate`, the `cuda` skill's `cuda-driver.md`) split one process's SMs among its own contexts.
  Neither reaches another process; MPS's client priority (below) is the cross-process one.

## CUDA MPS: works as the user, with unified memory

- `nvidia-cuda-mps-control -d` runs as the user (no root) with `CUDA_MPS_PIPE_DIRECTORY` and
  `CUDA_MPS_LOG_DIRECTORY` of its own: only processes given that pipe directory are its clients; every other
  CUDA process keeps its own context, time-sliced against the MPS server. Flash Next (81 GiB of buffers,
  graphs, page-locked staging), OCR and ASR all ran as clients, the same reply bits as without MPS.
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` a client is honoured: at 25 the probe sees 12 SMs (also as its green
  context resource and `MULTIPROCESSOR_COUNT`) and a 32-wave spin takes 4.0x (1,126 against 282 ms).
  `cuDeviceGetExecAffinitySupport(SM_COUNT)` is 1 under MPS. `CUDA_MPS_CLIENT_PRIORITY=1` (below normal) is
  honoured too: it moved Flash Next from 39 to 43 tok/s beside OCR at 25%.
- A fatal fault in one client ends the server and its other clients (MPS's documented behaviour; not
  provoked). zspark's stack stops every child when one ends anyway.
- `nvidia-smi` lists the clients with their own memory and `nvidia-cuda-mps-server` at 61 MiB. The control
  daemon starts its server as `nvidia-cuda-mps-server -force-tegra`: the driver treats GB10's integrated GPU as
  Tegra's for MPS. Start the daemon before any client: one that initializes CUDA first runs without MPS and
  says nothing.

| Flash Next tok/s beside OCR / ASR / both; OCR s a page, ASR RTF beside Flash Next (alone 4.8, 0.058) | FN | Small beside FN |
| --- | --- | --- |
| Time slicing | 21.8 / 21.0 / 13.5 | 9.6 (2.0x), 0.117 (2.0x) |
| MPS, small at 50% | 33.1 / 29.9 / 19.7 | 11.2 (2.3x), 0.123 (2.1x) |
| MPS, small at 25% | 39.4 / 36.5 / 27.9 | 15.2 (3.2x), 0.162 (2.8x) |
| MPS, small at 25% and priority 1 | 42.8 / 40.3 / 33.8 | 19.5 (4.1x), 0.218 (3.8x) |

On two Sparks (Flash Next's two ranks, OCR beside rank 0 and ASR beside rank 1, a daemon on each, the small
models at 25% and priority 1): Flash Next 58.0 tok/s beside OCR, 54.8 beside ASR, 52.5 beside both, of 72.7
alone, where time slicing left 29.7, 27.8 and 19.2; OCR 2.6x and ASR 2.4x their time alone beside it. A small
model on either Spark slows both ranks' lockstep; that two Sparks lose less is read (not traced) as a rank's
GPU idle while it waits on the link, which a small model takes.

The cap trades one side for the other along a line: Flash Next's fraction of alone plus OCR's is about 1.06
under every MPS setting against 0.91 time-sliced, so MPS's own gain is about 15% of the GPU (no switching, the
kernels overlap) and the share decides who has it. The small models alone at 25% are 5 to 9% slower. Flash
Next's decoding is bound by bytes: beside a small model it does not get back to 53 at any share, as both read
the one memory.
