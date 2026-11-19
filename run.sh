#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ $# -eq 0 ]]; then
  set -- examples/scenarios/startup.toml --arrivals 5
elif [[ $1 == --arrivals ]]; then
  set -- examples/scenarios/startup.toml "$@"
fi

if [[ ${CONDA_DEFAULT_ENV:-} == serverledge-simulator ]]; then
  exec python -m sim "$@"
fi

exec conda run --no-capture-output -n serverledge-simulator python -m sim "$@"
