#!/bin/bash
# Copies a checkpoint directory (or one file) to the other Spark over the ConnectX link, unencrypted and in
# several streams: rsync over ssh is bound by one thread of ssh and one of rsync (230 to 490 MB/s on a 200 Gb/s
# link). The link is private, so the bytes go through plain TCP (nc); ssh carries only the commands.
#
# Measured on 2026-10-05 (spec 022) with one 10 GiB file of random bytes, spark-a to spark-b, both machines
# locked, sha256 equal after each: rsync over ssh 486 MB/s (481 with aes128-gcm); this script 1.6 GB/s with one
# stream, 3.1 with 2, 5.0 with 4, 5.7 with 8 and 7.8 with 16. Eight is the default: past it every range opens
# more ssh sessions than sshd multiplexes over one connection (MaxSessions 10), which works and warns. Ranges
# of 256 MiB were slower than 1 GiB at 16 streams (6.6 GB/s), and a range is the unit a stream takes, so a file
# smaller than STREAMS ranges uses fewer streams (10 GiB in 4 GiB ranges: 3.9 GB/s).
#
#   tools/spark_copy.sh SRC DEST_ADDRESS DEST_DIR [STREAMS]
#   tools/spark_copy.sh ~/models/Qwen3.8-Flash-Next-FP4-Local 198.51.100.2 /home/example/models 8
#
# SRC lands as DEST_DIR/$(basename SRC). Run it on the machine that holds SRC, under the machine lock
# (`flock -s ~/.cache/zspark/machine.lock`, and the other machine's too: a copy disturbs timings on both).
# Directories, symbolic links (kept as links, not followed) and files under 64 MiB go by `rsync -a` over ssh.
# Larger files are cut into ranges of 1 GiB (SPARK_RANGE_MIB), each sent by `dd | nc` to a `nc | dd` at its offset, STREAMS
# (default 8) at a time, read and written with direct I/O so neither machine's page cache fills: on a Spark the
# page cache shares memory with the device.
#
# SPARK_SSH is the ssh command for the other machine (default `ssh`); from spark-a to 198.51.100.2 it is
# `ssh -o HostKeyAlias=192.0.2.227`. SPARK_SOURCE is this machine's address on the link (default: the local
# address of the route to DEST_ADDRESS). SPARK_PORT is the first of STREAMS ports the sender listens on (9400).
# SPARK_VERIFY=1 compares sha256 of every large file afterwards (about as long as the copy).
set -euo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,8p' "$0" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/local_env.sh"
load_local_env "$HERE/.." || exit $?
src=$(realpath -s "$1")
host=$2
dest_dir=$3
streams=${4:-8}
ssh_cmd=${SPARK_SSH:-ssh}
port=${SPARK_PORT:-9400}
source_address=${SPARK_SOURCE:-$(ip -o route get "$host" | sed -n 's/.* src \([0-9.]*\).*/\1/p')}
block=$((16 << 20))           # dd's block, and what direct I/O is aligned to
range_blocks=$((${SPARK_RANGE_MIB:-1024} / 16))   # blocks a range
small=$((64 << 20))           # files below this go with rsync

parent=$(dirname "$src")
name=$(basename "$src")

# Directories, links and small files.
rsync -a --max-size=$((small - 1)) -e "$ssh_cmd" "$src" "$host:$dest_dir/"

# The large files, relative to the parent, and their ranges: "path<TAB>first block<TAB>blocks".
jobs=$(mktemp)
files=$(mktemp)
trap 'rm -f "$jobs" "$files"; kill $(jobs -p) 2>/dev/null || true' EXIT
(cd "$parent" && find "$name" -type f -size +$((small - 1))c -printf '%s\t%T@\t%m\t%p\n') > "$files"
total=0
while IFS=$'\t' read -r size mtime mode path; do
  total=$((total + size))
  blocks=$(((size + block - 1) / block))
  for ((first = 0; first < blocks; first += range_blocks)); do
    printf '%s\t%d\t%d\n' "$path" "$first" "$((blocks - first < range_blocks ? blocks - first : range_blocks))" >> "$jobs"
  done
  # sized first, so its ranges can land in any order; a file there already (read-only, as the hub's are) is replaced
  $ssh_cmd "$host" "mkdir -p $(printf %q "$dest_dir/$(dirname "$path")") && rm -f $(printf %q "$dest_dir/$path") && truncate -s $size $(printf %q "$dest_dir/$path")" < /dev/null
done < "$files"

count=$(wc -l < "$jobs")
echo "$(wc -l < "$files") large files, $((total >> 20)) MiB in $count ranges, $streams streams from $source_address" >&2
start=$(date +%s.%N)

# Worker k takes ranges k, k + streams, ...; it listens on its own port and the receiver connects to it.
worker() {
  local k=$1 n=0 path first blocks
  while IFS=$'\t' read -r path first blocks; do
    if [ $((n++ % streams)) -ne "$k" ]; then continue; fi
    dd if="$parent/$path" bs=$block skip="$first" count="$blocks" iflag=direct status=none | nc -l -N "$source_address" $((port + k)) &
    # the receiver tries until the listener is up; a failed try has written nothing
    $ssh_cmd "$host" "set -o pipefail; for try in \$(seq 50); do nc -d $source_address $((port + k)) | dd of=$(printf %q "$dest_dir/$path") bs=$block seek=$first conv=notrunc oflag=direct iflag=fullblock status=none && exit 0; sleep 0.1; done; exit 1" < /dev/null
    wait $!
  done < "$jobs"
}
pids=()
for ((k = 0; k < streams; k++)); do
  worker "$k" &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ $status -ne 0 ]; then
  echo "a stream failed: the copy is incomplete" >&2
  exit 1
fi
end=$(date +%s.%N)
echo "$((total >> 20)) MiB in $(echo "$end - $start" | bc) s: $(echo "$total / 1000000 / ($end - $start)" | bc) MB/s" >&2

# Times and modes as the source's, so a later `rsync -a` finds nothing to do.
while IFS=$'\t' read -r size mtime mode path; do
  printf 'touch -d @%s %q && chmod %s %q\n' "$mtime" "$dest_dir/$path" "$mode" "$dest_dir/$path"
done < "$files" | $ssh_cmd "$host" bash

if [ "${SPARK_VERIFY:-0}" = 1 ]; then
  theirs=$(cut -f4 "$files" | $ssh_cmd "$host" "cd $(printf %q "$dest_dir") && xargs -d '\n' -P $streams -n 1 sha256sum | sort -k2")
  mine=$(cd "$parent" && cut -f4 "$files" | xargs -d '\n' -P "$streams" -n 1 sha256sum | sort -k2)
  if [ "$mine" != "$theirs" ]; then
    echo "sha256 differs between the two machines" >&2
    exit 1
  fi
  echo "sha256 equal for $(wc -l < "$files") files" >&2
fi
