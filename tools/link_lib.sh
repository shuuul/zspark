# What the scripts that run two ranks or two copies share, sourced by `two_ranks.sh`, `two_copies.sh`,
# `check.sh` and `reference/glm5_client_bench.sh` (spec 121): NCCL's settings for the link, and the guard that
# refuses a busy GPU. Both are shipped over ssh into the other Spark's shell (`declare -f link_env`, the guard's
# text in `bash -c`), so each stays one function of plain bash with no reference to the caller's variables.

# NCCL's settings for the link to $1, from this machine's own `ip` and sysfs; $2 the rails, $3 the channels.
link_env() {
  local peer=$1 rails=${2:-2} CHANNELS=${3:-} route dev addr hca gid= hex n hcas
  route=$(ip -o route get "$peer") || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  addr=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls "/sys/class/net/$dev/device/infiniband" 2>/dev/null | head -1)
  [ -n "$hca" ] || { echo "two_ranks.sh: $dev, the route to $peer, has no RoCE device" >&2; return 1; }
  # the RoCE v2 entry whose GID is this address mapped into IPv6
  hex=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' ${addr//./ })
  for n in "/sys/class/infiniband/$hca/ports/1/gid_attrs/types/"*; do
    [ "$(cat "$n" 2>/dev/null)" = "RoCE v2" ] || continue
    [ "$(cat "/sys/class/infiniband/$hca/ports/1/gids/${n##*/}" 2>/dev/null)" = "$hex" ] && { gid=${n##*/}; break; }
  done
  [ -n "$gid" ] || { echo "two_ranks.sh: $hca has no RoCE v2 GID of $addr" >&2; return 1; }
  hcas=$hca
  [ "$rails" = 1 ] || for n in /sys/class/infiniband/*; do
    [ "${n##*/}" != "$hca" ] && [ "$(cat "$n/ports/1/gid_attrs/types/$gid" 2>/dev/null)" = "RoCE v2" ] || continue
    case $(cat "$n/ports/1/gids/$gid" 2>/dev/null) in 0000:0000:0000:0000:0000:ffff:*) hcas=$hcas,${n##*/} ;; esac
  done
  echo "NCCL_SOCKET_IFNAME=$dev NCCL_IB_HCA=$hcas NCCL_IB_GID_INDEX=$gid${CHANNELS:+ NCCL_MIN_NCHANNELS=$CHANNELS NCCL_MAX_NCHANNELS=$CHANNELS}"
}

# The text a `bash -c` runs before its command, under the locks: it ends the run (75) when a process holds the
# GPU. PREFIX is the script's name in the message; `plain` is check.sh's form (no host, on stdout, indented
# for its `exec "$@"` line), otherwise the two machines' form (the host named, on stderr).
gpu_guard() { # PREFIX [plain]
  local get='held=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null)'
  if [ "${2:-}" = plain ]; then
    printf '%s\n      %s' "$get" 'if [ -n "$held" ]; then echo "'"$1"': the GPU is in use, nothing is run: $held"; exit 75; fi'
  else
    printf '%s\n  %s' "$get" 'if [ -n "$held" ]; then echo "'"$1"': the GPU of $(hostname) is in use, nothing is run: $held" >&2; exit 75; fi'
  fi
}
