# Block-FP8 linears on the GB10: time by shape (spec 070 WS-08)

`qmmf_kernel<FP8G>` (`zig/kernels/nvfp4/qmmf.cu`) and `qmmt_kernel` (`qmmf_tile.cu`, one scale a tile and group,
K slices summed in the block from 40 tiles) at Flash Next's shapes on the hybrid's default layout. Durations a
launch, us; 1, 4, 9 and 16 rows within 1%; read-only over the bytes a qmmf launch reads (codes plus 256 B of
scales a tile and group). The measuring protocol is [`measuring.md`](measuring.md)'s.

| Linear | N x K | Tiles x K slices | MB a launch | Launches a round | In the round | qmmf | GB/s | Read-only | qmmf_tile |
|---|---|---|---|---|---|---|---|---|---|
| DeltaNet in_proj qkv + z | 16,384 x 2,560 | 256 x 1 | 44.56 | 36 | 200.6 | 192.4 | 232 | 206.5 | 181.1 |
| attention q + k + v | 13,312 x 2,560 | 208 x 1 | 36.21 | 12 | 164.8 | 155.7 | 233 | 160.0 | 147.7 |
| DeltaNet out_proj, attention o_proj | 2,560 x 6,144 | 40 x 8 | 16.71 | 48 | 85.8 | 86.4 | 193 | 72.2 | 67.7 (fused) |
| draft head fc_embedding, fc_hidden | 2,560 x 2,560 | 40 x 4 | 6.96 | about 16 | not traced | 37.3 | 187 | 32.6 | 29.3 (fused) |
| read-out down + inject | 324 x 10,240 | 6 x 8 | 4.18 | 96 | 28.1 | 25.7 | 163 | 19.0 | 24.5 (cluster) |
| read-out up | 10,240 x 320 | 160 x 1 | 3.48 | 97 | 20.8 | 17.2 | 202 | 17.2 | 16.1 |
| shared expert gate + up | 1,280 x 2,560 | 20 x 4 | 3.48 | 48 | 23.3 | 19.5 | 179 | 17.3 | 18.5 (cluster) |
| shared expert down | 2,560 x 640 | 40 x 1 | 1.74 | 48 | 11.9 | 9.3 | 187 | 9.2 | 8.65 |
| lm_head | 248,320 x 2,560 | 3,880 x 1 | 675.4 | 1 | 2,785 | 2,871 | 235 | 2,927 | 2,708 |
| draft head (NVFP4, 79,591 ids) | 79,591 x 2,560 | 1,244 x 1 | 114.6 | 7.9 | 478.2 | 495 | 232 | 485 | not its kernel |

- One K slice: qmmf is at the memory's rate (it passes the read-only kernel: staged `cp.async` against plain
  reads). Its time is its bytes, the scales included: 256 B of fp32 beside 4,096 B of codes are 5.9% of both.
  Without staging them 190.5 us become 178.8; a stage without them, four blocks an SM, adds nothing (179.3).
- K slices by a cluster are what the shapes under the rate lack. 2,560 x 6,144 by cluster size: 71.8 us at one
  slice, 73.5 at 2, 77.9 at 4, 86.2 at 8. The same slices written to a part buffer and summed by a second
  kernel: 72.7 + 1.8. The read-out's down (6 tiles): 25.6 by the cluster of 8, 19.2 to parts (its read-only
  time), 56 at one slice (6 blocks).
- One block summing its tile's slices in slice order (the cluster's bits) wins from 40 tiles (86.4 to 72.7 us
  with qmmf's scales) and loses under them (20 tiles: 20.3 against 19.6; 6 tiles: 56 against 25.7): too few
  blocks for 48 SMs. From 40 tiles it gives the same bits at the memory's rate; under 40 the blocks are too few
  and the cluster stays.
- In a round the small launches are 2.4 to 3.8 us slower than standalone (about 1.4 ms a round in all); not
  reproduced standalone.

## The last allocation reads faster

The last 32 MB or so that `cudaMalloc` handed out read about 25% faster than every other byte (the read-only
kernel 279 against 222 GB/s on 32 MB windows, qmmf on a 44.6 MB weight 151 to 157 us against 190). What it
follows — the pages the CPU wrote last, and that it ends by itself — is in `specs.md`, "Pages the CPU wrote
last". Time new work at 64 MB allocated after the weights (bimodal medians are the sign of its lack).
