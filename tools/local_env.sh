# Read machine-local defaults as literal KEY=value lines. The caller's variables win.
load_local_env() {
  local file=$1/.env.local line name value
  [[ -f $file ]] || return 0
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    case $line in ''|\#*) continue ;; esac
    name=${line%%=*}
    value=${line#*=}
    case $name in
      PEER|PEER_REPO|MODEL_LAN_HOST|MODEL_LAN_PORT|SPARKDASH_HOST|SPARKDASH_URL|SPARKDASH_SPARK|SPARK_SSH|SPARK_SOURCE)
        if [[ $line == *=* ]]; then
          if ! [[ -v $name ]]; then export "$name=$value"; fi
          continue
        fi ;;
    esac
    printf 'Invalid setting in %s. See .env.local.example.\n' "$file" >&2
    return 2
  done < "$file"
}
