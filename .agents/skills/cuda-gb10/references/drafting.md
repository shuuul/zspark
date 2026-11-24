# Drafting on the GB10

What a drafted row and a round cost on this chip, which stopping rules and drafters paid, and what does not
pay. Part of the `cuda-gb10` skill. A round is one verify window (the kept row plus the drafts) and the head
passes that made its drafts; MTP is a checkpoint's own draft layer, DFlash2 GLM's block drafter. Verification
is exact only by the row rule ([`chunking.md`](chunking.md)).

## Why drafting pays here

A round reads every weight once whatever its rows, so a drafted row costs little beside the round's base:

| Model | Base a round | A row more | Source |
| --- | --- | --- | --- |
| Flash Next NVFP4, one rank, verify window | 51.0 ms at 1 row | +3.3 ms, 96.5 at 15 | 061 WS-01 |
| Flash Next NVFP4, a head pass | 1.67 ms | +1.63 a draft (the pack 1.23 + 1.23) | 061 WS-01 |
| Flash Next NVFP4, two ranks | 31.2-32.5 ms a share | +0.82-1.11 (one rank 50.4 + 2.54) | 090 WS-00 |
| Flash Next FP4-Local | a round of 50 ms | a drafted row 3 to 4 ms | 105 |
| GLM FP4-Local, MTP, two ranks | 53.5 ms for 2 rows, about 41 of it weights at the bus | | 114 WS-01 |
| GLM FP4-Local, DFlash2 | 89 ms for about 6 rows, 4.8 tokens | each row past the second about 9 ms | 114 |

- A token kept is worth 18 ms alone and 8 ms at 4 seats in the pool (077). The packs beat TensorFold by a
  cheaper round, not more drafts: prose 58.5 against 56.8 tok/s (3.05-bit), 52.7 against 49.7 (4.05-bit), at
  1.17 / 1.19 drafts kept a round (102 WS-03).
- GLM's row against TensorFold's, resolved for 099's method (measuring at equal rows, not fitting over
  differing ones): 091 read a row 21.2 ms against 24.8 but a round 111.7 against 90.6, zspark verifying 5.27
  rows for 3.65; 093 fit a row 1.0 ms dearer (9.27 against 8.26); 099 WS-01 at equal rows found the base
  58.33 against 63.1 and DFlash2's 1+7 rows 114.1 against 113.6 modelled.

## Stopping rules

| Rule | Finding | Source |
| --- | --- | --- |
| Depth by TensorFold's `DepthRule`, power 2, costs as constants a storage | reasoning +7.8% NVFP4, +4.9% pack; pack prose -1.0%. Power 1 (-2.5/-5.3%), acceptance by depth (-7.2/-9.2%) and measured costs (code -1.7, prose -2.7%) lose; timing costs at load would add 3 to 5 s. Oracle ceiling about +22 to 23% | 061 |
| `trust`, kept over predicted within 0.25 to 1, weight 0.05 a round | pack prose -2.0 to -0.4%; a kind's replies differ by 8 points, so judge a kind by its mean | 061 |
| Which draft lands, a fitted predictor | none reached the gate (AUC 0.889/0.901 against 0.945, probes overfit at 68 replies); "form 2" (square, calibrated product, 7 numbers a pack) kept, code +1.5 to +3.9%; "form 1" loses (prose -0.6%) | 085 |
| vLLM's online refit | replayed: -0.3 to +0.3 on prose; not built | 102 WS-01 |
| GLM MTP, 1,919 recorded chains replayed | nothing beat 0.35 / no cap by more than 0.5%; TensorFold's cap 3 -8.2%; the cap flag withdrawn | 092 |
| GLM MTP cap 5 on two ranks | +8.5% (41.85/42.46 against 38.59/38.17); the owner set `two_rank_mtp_cap` 5, `default_cap` stays 15 | 099 WS-04 |
| GLM MTP `--confidence` on two ranks | 0.65 chosen, 1.029x 0.35's replies | 114 WS-02 |
| DFlash2 floor | `fnc7:0.2` 34.63 against `fc5:0.3` 33.35 tok/s (+3.8%; the model read one point high); 0.2 against 0: chains 44.3-44.7 against 41.2-41.7. Cap alone -4.6% and a drafter chosen a round -5.0% lose. Bounds: knowing the round +10.3%, the kept drafts +22.7% | 067 WS-12, 099 |

- The 092 / 099 contradiction resolves for 099: 092 modelled a cap by replaying recorded chains, 099 timed
  served replies on the whole model, and a cap's gain is the round's time it saves, which a replay of counts
  does not hold. The two-rank cap is 5.
- The loopback cannot judge drafters: over 16 layers MTP kept 4 of 122 drafts and DFlash2 1 of 122, the taps
  clamped to the last loaded layer. Judge on two ranks over the whole model (099 WS-04).
- DFlash2's room (drafts a block) 7 / 5 / 3: 44.58 / 40.60 / 36.77 tok/s (099). A block is one pass whatever
  it keeps.

## Drafts by copy, by nucleus, by the pool's round

- Copied drafts (060), taken only where the copy begins with the head's first draft (ungated, plain replies
  lose up to 7%, reckoned): edits 1.45 to 1.67x NVFP4, 1.69 to 1.95x on the pack; a 200-line file returned
  with an edit 198 against 106 tok/s. The verify window stays 16 rows: at 63, edits would reach 1.84 to 2.27x
  but plain replies 0.52 to 0.84x (reckoned).
- A nucleus draft from the device's top 64 plus the row's sum settles 84 to 98% of drafts: prose +2.2 to 2.5%,
  reasoning +4.3 to 5.8% (065).
- The pool drafts by the shared round (077 `mtp.Paying.Round`): pack fit 26.9 ms + 3.35 a seat + 2.53 a row, a
  row counted whole to 32 rows, 0.6 to 64, 0.46 past. NVFP4's fit (52.1 + 3.5 + 2.2) gave nothing. Caps of 1
  or 2 drafts +5 to 6% at 4 streams (076). Withdrawn: trust by depth, cost-side ranking.
- Head passes shared: one launch for every seat, 5.69 to 1.45 ms a round, +3.2 to 4.6% at four streams; the
  head's projection over 79,591 ids reads 114.6 MB (about 0.5 ms) whatever the rows (078).
- Several streams on FP4-Local (144 WS-01, `nsys` of serve at `--parallel 4`): a round of 4 seats is some 16
  rows (3.9 verified, 2.8 kept a seat), never 32; 29.5% of verified rows are rejected (31% on one Spark). A
  row costs the trunk some 2 ms on two ranks, 2.5 on one (33 to 57 ms from 4 to 16 rows; 47 to 83 from 4.5 to
  19), the routed experts most of it (17.4 to 48.6 ms a round on one Spark): rejected rows are 12 to 14% of a
  4-stream round. The head's chase runs to the deepest seat: 6.6 passes a round at 4 streams on two ranks.
- FP8 trunk caches with the bf16 head keep fewer drafts: on the same seeds 3 to 17% fewer kept a reply, 6 to
  39% more rounds, than bf16 caches (two ranks, 144 WS-01): one stream 63.8 against 72.4 tok/s, four 134
  against 149.

## What a drafter may store

The model that emits the reply runs as stored; a drafter may be quantized at load (the owner, 2026-10-10):

- Drafts through the stored head were slower: 8% prose and 12% code on the 3.05 pack, 20% and 34% on NVFP4,
  with no draft more kept (071 WS-06). The draft head is rows decoded on the device and packed NVFP4
  (`DraftHead.initDecoded`). The MTP head as block FP8: a draft 1.97 to 1.55 ms (070 WS-09).
- The MTP head's cache stays bf16 whatever the trunk's: an FP8 trunk cache cost 1.23 drafts a round (5.55 to
  4.32; code 76.1 to 65.6 tok/s), 0.69 of them the head's; bf16 there costs 276 MiB alone, 1,032 in the pool
  (051/052). TensorFold's head cache follows the trunk.
- An FP8 trunk cache still costs Flash Next FP4-Local's drafts: 5.0 kept a round against 5.6 on code (100).
- DFlash2 (067): NVFP4 at load but k/v and the kernel projections (bf16, exact over 2,047 positions), 415 MiB
  a rank; a four-bit head copy cut the block's head 5.7 to 1.8 ms (block 11.06 to 6.96), its cost in kept
  drafts not measured then; the recipe since: 5.7 to 1.5 ms of an 11.9 ms block, at a cost in kept drafts.
  DFlash2's weights are never committed, redistributed or written as a copy; since 2026-10-20 they may cross
  the link into rank 1's memory (root `AGENTS.md`). GLM's mixed NVFP4 linears against stored bf16: chains
  43.92/44.88 against 42.57/42.75 tok/s, drafts kept identical (099). Rescoring four-bit picks on bf16 rows:
  no gain (098 WS-04).
- DFlash2 attention as an online softmax: `attend` 0.129 to 0.024 ms; device top-k 0.22 ms against the host's
  0.7 to 2.2 (067).

## GLM FP4-Local's drafters (two ranks)

- MTP 43.1 against DFlash2 48.2 tok/s; DFlash2 1.12 to 1.16x on every reply, 1.40x on short code (107 WS-04,
  114). Its pass over a cold prompt costs the first token about 8% (0.92x); widened rows 0.59 GiB a rank.
- Refused on GLM (114): MTP step graphs, an NVFP4 `eh`. MTP and DFlash2 keep the quantized forms in every
  phase: they only draft ([`precision-forms.md`](precision-forms.md)).
- The first token is handed out before drafting starts (113 WS-06).

## Flash Next's DFlash2 at two ranks (135)

- Its drafts were not the gap: prose-long ran 227 rounds and kept 284 drafts with DFlash2 and with MTP; the
  gap was the drafter's cost a round. Of it: the bf16 proposing head (half the vocabulary, 636 MB a rank)
  2.63 ms at the bus, the other bf16 GEMMs 2.55, the gathers 1.1 to 2.3.
- The four-bit drafter (`mixed`: NVFP4 linears, the head `qmmf_kernel`, decoded in registers into
  `m16n8k16`): the head 2.59 to 0.75 ms a round; at two ranks 1.072 times stored's decode over 36 cases (31
  ahead), 0.950 of MTP's, 0.74% fewer drafts kept, 0.86 GiB less a rank (six loads).
- Captured drafter passes at two ranks took nothing off a round (31.56 against 31.69 ms, the device idle 1.0
  ms either way): the eager launches were already ahead of the device. A trace's idle and gather time is not a
  graph's gain; read the device's busy share first. Refused.
- A cold prompt told once (rank 1 running the windows and the drafter's taps from frames) with the taps fed a
  window's rows a pass: 52 ms off a 4k first token (3%), 348 off 16k (5%), against some 150 to 200 ms the
  trace's gaps promised at 4k; DFlash2 still 14% behind MTP's 1.49 s there.
- One Spark (both halves on one device, `mixed`, 143 WS-03, `nsys --cuda-graph-trace=node`): a round 50.1 ms
  wall, 49.2 of device work; the head 2.05 ms of it (4.2%): two half-head `qmmf` launches of 124,160 rows,
  730 us each (about 233 GB/s, at the bus), `top_kernel` 2 x 170 us and `rescore_kernel` 2 x 124 us, each on 7
  blocks (one a row) of 48 SMs. Upstream's draft vocabulary ([0, 98,304) and the last 288, the 27B's) would
  lie almost wholly in rank 0's contiguous half, so at two ranks it cuts some 0.2 ms of a 31.6 ms round;
  refused at two ranks.
- A selection over 124,160 logits split along the row (143 WS-09): 31 slices of 4,096 columns a row, 256
  threads, a warp butterfly and one warp over the warps a pick, then a merge block a row over the slices' 16
  each: `top` 168 us a launch to 43 + 15 (7 to 217 blocks); the order (value, then the lower token) is total,
  so the picks are the same bits. `rescore` a block a pick, a warp staging both rows in shared memory and one
  lane running the ordered `fmaf` chain: 116 to 26 us (a thread a pick's dependent 2-byte loads were latency).
  0.58 to 0.17 ms a round, the same replies and drafts.
- `nsys` over a two-rank load: `WRAP=` on rank 0 alone, `--cuda-graph-trace=node` to see a graph's kernels;
  traced is not timed ([`measuring.md`](measuring.md)).

## Sources

Specs 051, 052, 060, 061, 065, 067, 070, 071, 076 to 078, 085, 090 to 093, 098 to 100, 102, 105, 107, 114, 135, 143;
`docs/recipes/glm-5.3-flash-fp4-local.md`, `docs/recipes/qwen3.8-flash-next-fp4-local.md`.
