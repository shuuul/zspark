# Deployed weight identity manifest

The exact file identity of the weights the owner's two-Spark LAN stack was serving when it was
recorded on 2026-11-13. Live process inspection on spark-a showed `zspark stack` serving Flash Next
from `~/models/Qwen3.8-Flash-Next-FP4-Local` with drafter `~/models/Qwen3.8-Flash-Next-DFlash2`, and
Nemotron ASR from the HF snapshot
`models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4/snapshots/66c6adc…` with its default
DFlash2 drafter `~/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`. The GLM drafter is
excluded by policy.

Every deployed file was hashed with `sha256sum` (symlinks resolved, single stream) on both Sparks
on 2026-11-13 under a shared `~/.cache/zspark/machine.lock`. All four file lists and all 71 SHA256
sums are identical on spark-a and spark-b. Of the 70 files present both upstream and locally, every
one equals the pinned revision's bytes:

- 33 **LFS** files: the SHA256 equals the repository's LFS `oid` published by
  `https://huggingface.co/api/models/<repo>/tree/<revision>?recursive=true`, so the deployed bytes
  are exactly the pinned revision's upload.
- 37 **non-LFS** files: the deployed bytes were refetched from
  `https://huggingface.co/<repo>/resolve/<revision>/<path>` and compared equal.

The remaining two divergences are file-set only, both in the Nemotron DFlash2 directory: one local
file has no upstream counterpart (`configuration.json`) and one upstream file is absent locally
(`licenses/vLLM-LICENSE.txt`); details in that checkpoint's section.

## `nvidia/Qwen3.8-Flash-Next-FP4-Local` @ `b9ce18a6887e18c089fdea8509a33e63429a3300` — target, served

License: NVIDIA Open Model Agreement. Deployed path `~/models/Qwen3.8-Flash-Next-FP4-Local`:
on spark-a a symlink to the HF snapshot of this revision; on spark-b a plain directory copy.
33 files, 102,262,310,289 bytes.

```text
8237eb95e69c239c1e6d842039e720117d7907478b67433bb16215b2dbc4d002  .gitattributes
b0ad2a8a314ca96618e21e40d50c0af6e181e6060f9cdd1996c7074d19284c1a  README.md
c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041  chat_template.jinja
b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28  config.json
e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e  generation_config.json
a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d  merges.txt
5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1  model.safetensors.index.json
27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516  preprocessor_config.json
ebf3648c5f4cd1f24e79789e7803498dc4e74cb52515cc515b2d22cf8161953b  tensors/model-00001-of-00021.safetensors
79a643fc5c024089cecd32ca27dc43e671cbec70ff3628b728909b7656ac43ec  tensors/model-00002-of-00021.safetensors
c45f942203db4c7f5a1071700b93b908495cc790271acddfff39091c2b953923  tensors/model-00003-of-00021.safetensors
071ab9cf963ca0f1a6a194b4cec485f5de18ef9804da915b2e6abd0eb0c707af  tensors/model-00004-of-00021.safetensors
f80d17870eb1d36917257494770984953476583c57737f61fdbd341c957b9715  tensors/model-00005-of-00021.safetensors
3e9183d4ebb267fa92429184d5132d302cb1a98fd049149e5d01dccd4ca65795  tensors/model-00006-of-00021.safetensors
aa1ad959c4d62e0bc4600fd0b57cc2939b463e9fb53dc01020e188552b52ce5a  tensors/model-00007-of-00021.safetensors
0b18844027be74b84b240f23bf3a744bc29b35de0bd384f41c40cd85e6339326  tensors/model-00008-of-00021.safetensors
2d0b6c66790c17f8b1fdcf442f541778c8f0a381aca26c11b9d7cfcf7c96f2f0  tensors/model-00009-of-00021.safetensors
3ed9bd73745aebf6dc136f437b67d6b9fc3d38d25ad84300b399164bc87a72b7  tensors/model-00010-of-00021.safetensors
4936ff28c6cc8b66653ddb9cae28bc284569a1a2f7116819a017dc1a0e517ed8  tensors/model-00011-of-00021.safetensors
dd7df5a6ff73e9960412243f744eab9ec35f4be40da5bb158dc4ba8796b33742  tensors/model-00012-of-00021.safetensors
28a455c8ea4ebaa847fb4cbed2bc1de9eff0b501dd3f848a3a3cfc38778e8500  tensors/model-00013-of-00021.safetensors
6941676e685abf9817d9c48a17fdeda20b2b72d16217555252aa4c602b7f98b8  tensors/model-00014-of-00021.safetensors
2a1129b5acfc24a45bf402aa63dcef08b531aedfdaf5233c65fd3d12dc303fdd  tensors/model-00015-of-00021.safetensors
635e97fbad78fa988647d407e2b58e55d551ef94d798c9b6fb5550de25e86429  tensors/model-00016-of-00021.safetensors
bf60d564d89a805b67f8be5f2847a1b1567a70dbc23d19754395c127568474ad  tensors/model-00017-of-00021.safetensors
c5bde22f9364e82eb25973456795592fdcb4b158bc2095898979a2566bf4c562  tensors/model-00018-of-00021.safetensors
da3dc3b4f428f5b9dd53362691043d0242f48d03adca9b7b553bd9f3a1e96350  tensors/model-00019-of-00021.safetensors
4978bb6b8480cfc02d3c8bc84ba86202889f5f10d090bb17c21fee663cea7683  tensors/model-00020-of-00021.safetensors
48203838f2c76921bddbf1cc51d20a23858c128821a21add55fa1778e5e25283  tensors/model-00021-of-00021.safetensors
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3  tokenizer.json
b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27  tokenizer_config.json
7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13  video_preprocessor_config.json
ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003  vocab.json
```

## `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744e85c17364dcc54a6a420af7135d4c7f6` — served drafter

License: Apache-2.0. Deployed path `~/models/Qwen3.8-Flash-Next-DFlash2`: on spark-a a symlink to
the HF snapshot of this revision; on spark-b a plain directory copy. 6 files, 1,206,437,758 bytes.

```text
8e63f02a5caa4dbc92838d44dddcfa2c1b76f355503fcbeaa3d6512f7fe132ef  .gitattributes
296dfce04d501f313f4223a4ab2378f198ecb860e3b24a4689e66aff651f0477  README.md
6d8dcc9a9472bddb644c881fe83050b5343d09bf5ffb0912e5598e55fe671e99  assets/dflash2-figure.png
2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9  config.json
d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f  model.safetensors
91708e18c67f1c0e8f6c19cfbe7cc9f59ff257c05d61bb30b407ae4c36354169  patches/qwen38-dflash2-capture.patch
```

## `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` @ `66c6adc4020179567074d0399cda2a4eeb2bb347` — ASR target, served

License: OpenMDW-1.1. Deployed path: the HF snapshot directory itself, on both Sparks.
24 files, 21,577,427,198 bytes.

```text
3e1957759651c38eaf0a5aa5b9c5e2544218fb63a23b7ab1f39d4df26b7f6778  .gitattributes
a7454a46e4dfa3803f1d6049cfda5292ea1e7643b3a62bae075befe0bb0232a6  CONTRIBUTING.md
18c3a96fd1ee779263aa9210e3d962d5899511890254f3f3efd70ece81d89866  LICENSE
dc4049b66d750e2a40351e4c2b64cbf56f1f7d9fb354bf03c7d618bfec1d9336  README.md
d61d7ca8e896ccc5de6573ce75cc91335fe7460718b8cd84a8f1312fd5b1b72c  THIRD_PARTY_NOTICES.md
bcf8a0ae24191ed08feede766aedb2cca1431dd389badc8c691f6d85c4cad712  architecture.png
c439e20c5565d590915ff8777c519b3ebcd3da685e677893bee56f73036eb67c  chat_template.jinja
c6509c7a68ae4bd94fca06a39216029c233f8dc43428d6228fe425cfe4926313  config.json
2f3543adfd622fcd474bb42a04b4b331be63899fbe133d26b0c90e15fbaab74b  finetune_gains.png
1ff3bfb8a60751cc1f68d756dc05b20910c3d124f38ef8ac5aed59e374aa7f09  generation_config.json
a7a120da10970753a539fc8426238bab866725ce75cf4e2b54d3a5dd75948efc  hero.png
dff9458dd0e62f9c10551b152452a79bfe3e36b3d803a43a28dad27706894d4c  hf_quant_config.json
a44a6081c73ad75f0255bb2bb5cab74ef1829565a895a24e53a4f11290ab7655  licenses/Qwen3-ASR-LICENSE.txt
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  licenses/vLLM-LICENSE.txt
8f11ffe191b8ade26ff9b43196a2cfcc356f6ee73cfb3c6a6336968f178611d0  llm_backbone/config.json
633b7ca55c9bd105e03499ad238f65193b949b4f0003226a5b2639b970162fe3  model-00001-of-00002.safetensors
75666c368d00e037216cd86e0086df242f126a2f1def78a71c93fe06c66ea676  model-00002-of-00002.safetensors
fd902c40ab19dcb1a9f0a77a70043a2e5a3cadf682d0aa28ad10d57ce978f42e  model-ctc-bf16.safetensors
b6569930eb0de2b020487e2002c99f841f3462c6741e7edcbb8527bfe332ac18  model-mtp-fp8.safetensors
841f8f120aca973398891436b32e5de7444ce784a61217ccbf46e26738a7bd92  model.safetensors.index.json
e08b3c7f5f0056515a9772c9fd3f87af1120d1d9e900d406d046c487fa0a9e4a  perception.safetensors
6da08005d04388bace9ada9219d54dcf60296bbbbb91d8601205ec7b5eb1479f  serving_performance.png
d071004cb4450d3aa31f87aab979cd09ffc99c4e709e6cf72c324346cd948d0d  tokenizer.json
e11d0bb1b107a182a6db567822b605db852edcb01e87368bd8e28c62c2991d15  tokenizer_config.json
```

## `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2` @ `4172ee5b7bfe55f59e784922878c1bd9bdfeb6ba` — served drafter

License: OpenMDW-1.1. Deployed path `~/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`:
a plain directory on both Sparks, not an HF cache snapshot. 8 files, 1,141,313,666 bytes.
The weight file `model.safetensors` hashes to the pinned revision's LFS `oid`, and `config.json`
matches the pinned bytes. Two differences from the snapshot, neither weight-relevant: an added
`configuration.json` (`{"task":"speaker-verification"}`, sha256 `59c32d8e…`) and the absent
`licenses/vLLM-LICENSE.txt`.

```text
11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361  .gitattributes
a7454a46e4dfa3803f1d6049cfda5292ea1e7643b3a62bae075befe0bb0232a6  CONTRIBUTING.md
18c3a96fd1ee779263aa9210e3d962d5899511890254f3f3efd70ece81d89866  LICENSE
ffe530b6d351fd32eea0da563869d8ad39bf6d7753af0632463ba5915bec0ec8  README.md
b576f2583c8b73bc2718808d2ad66c0ae486cb19e39b865aeaaf1adfb62bf35c  THIRD_PARTY_NOTICES.md
03afd1ca1788d19cd1ac9e8884defdaa5817f7ddf25586a571d78a06c6308346  config.json
59c32d8ecd861042417cf746d42a1792649fd68034c2d32cd76a4e017d67eccb  configuration.json
3552a1723ffbdabfd45626b4b5cccce111049677be980ff6bd6d4e8a0cf9ee6a  model.safetensors
```

## Reproducing the check

Download each checkpoint at the pinned revision (never `main`), hash every file, and compare
with the corresponding block above:

```bash
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local \
  --revision b9ce18a6887e18c089fdea8509a33e63429a3300
hf download incoai/Qwen3.8-Flash-Next-DFlash2 \
  --revision a22a1744e85c17364dcc54a6a420af7135d4c7f6
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 \
  --revision 66c6adc4020179567074d0399cda2a4eeb2bb347
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2 \
  --revision 4172ee5b7bfe55f59e784922878c1bd9bdfeb6ba

# in each snapshot or copied directory:
(cd DIR && find . ! -type d | sed 's|^\./||' | sort | xargs -d '\n' sha256sum) | sort -k2
```

For LFS files the published expected SHA256 also comes straight from the API, which lets an
outsider verify a download without trusting this document:

```bash
curl -s "https://huggingface.co/api/models/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300?recursive=true" \
  | jq -r '.[] | select(.type=="file" and .lfs != null) | "\(.lfs.oid)  \(.path)"'
```
