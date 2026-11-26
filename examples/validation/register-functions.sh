#!/usr/bin/env bash
set -euo pipefail

# Registra le sei funzioni sintetiche (sim_*) sull'istanza Serverledge locale.
# Passa --update per creare o sovrascrivere in modo idempotente le definizioni in etcd.
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --update ) ]]; then
  echo "Uso: validate.sh register [--update]" >&2
  exit 2
fi
update=("$@")

register() {
  local name="$1" tag="$2" memory="$3" cpu="$4" concurrency="$5" handler="${6:-synthetic}"
  local result
  result="$(docker compose run --rm -T cli --host 127.0.0.1 --port 13300 create \
    --function "$name" --tag_pattern "$tag" --memory "$memory" --cpu "$cpu" \
    --max_concurrency "$concurrency" --runtime python314 --src /lab/workload.py \
    --handler "workload.$handler" "${update[@]}")"
  printf '%s\n' "$result"
  python3 -c 'import json, sys; sys.exit(json.load(sys.stdin).get("Created") != sys.argv[1])' "$name" <<< "$result"
}

register sim_hello 'amd64*' 128 1.0 1
register sim_resize 'amd64*' 256 0.5 2
register sim_telemetry gateway-arm 128 1.0 1
register sim_archive archive-arm 128 1.0 1
register sim_burst forward-arm 128 1.0 1
register sim_index regional-arm 128 1.0 1
