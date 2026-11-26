#!/usr/bin/env bash
set -euo pipefail

# Avvia l'addestramento dello scaler PPO (Gymnasium + Ray RLlib) attivando automaticamente l'ambiente Conda.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SCENARIO="examples/scenarios/autoscaling.toml"
DEFAULT_CHECKPOINT="checkpoints/ppo"

# Determina il nome predefinito dell'ambiente Conda da environment.yml, se presente.
detect_default_env() {
  local env_file="$ROOT_DIR/environment.yml"
  if [[ -f "$env_file" ]]; then
    local name
    name="$(sed -nE 's/^[[:space:]]*name:[[:space:]]*([^[:space:]#]+).*/\1/p' "$env_file" | head -n 1)"
    if [[ -n "$name" ]]; then
      printf '%s\n' "$name"
      return
    fi
  fi
  printf '%s\n' "serverledge-simulator"
}

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  local candidate
  for candidate in \
    "${CONDA_EXE:-}" \
    "/opt/conda/miniforge3/bin/conda" \
    "/opt/conda/bin/conda" \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$HOME/mambaforge/bin/conda"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

activate_conda_env() {
  local env_name="$1"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
    return 0
  fi

  local conda_bin
  if ! conda_bin="$(find_conda)"; then
    echo "Errore: conda non trovato nel PATH né nei percorsi standard." >&2
    echo "Imposta CONDA_EXE o attiva manualmente l'ambiente '$env_name'." >&2
    exit 1
  fi

  # Carica gli hook di Conda nella shell corrente ed entra nell'ambiente indicato.
  eval "$("$conda_bin" shell.bash hook)"
  conda activate "$env_name"
}

usage() {
  cat <<USAGE
Uso:
  ./train.sh [--env NOME_ENV] [SCENARIO] [OPZIONI TRAINER...]

Opzioni del launcher:
  -e, --env NOME    Ambiente Conda da attivare (default: ${CONDA_ENV_NAME}, oppure variabile CONDA_ENV)
  -h, --help        Mostra questo messaggio e le opzioni del trainer

Opzioni principali di python -m trainer:
  --region NOME         Regione target (default: primary_region)
  --iterations N        Numero di iterazioni RLlib (default: 10)
  --max-steps N         Passi Delta massimi per episodio (default: 50)
  --max-arrivals N      Limite arrivi esterni per episodio
  --max-time SEC        Orizzonte temporale episodio in secondi
  --workers N           Numero di EnvRunner paralleli Ray (default: 0)
  --gpus G              GPU per learner (es. 1.0 per CUDA, default: 0.0)
  --checkpoint DIR      Directory in cui salvare il checkpoint RLlib
  --seed SEED           Seed opzionale (default: seed dello scenario)

Esempi:
  ./train.sh
  ./train.sh --iterations 20 --workers 4 --checkpoint checkpoints/ppo
  ./train.sh examples/scenarios/autoscaling.toml --iterations 50 --gpus 1.0
  ./train.sh --env serverledge-simulator examples/scenarios/autoscaling.toml --max-steps 100

Senza argomenti avvia: ${DEFAULT_SCENARIO} --checkpoint ${DEFAULT_CHECKPOINT}
USAGE
}

CONDA_ENV_NAME="${CONDA_ENV:-$(detect_default_env)}"
SHOW_HELP=0
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env)
      if [[ $# -lt 2 ]]; then
        echo "Errore: $1 richiede il nome dell'ambiente Conda." >&2
        exit 2
      fi
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --env=*)
      CONDA_ENV_NAME="${1#*=}"
      shift
      ;;
    -h|--help)
      SHOW_HELP=1
      shift
      ;;
    --)
      shift
      TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$SHOW_HELP" -eq 1 ]]; then
  usage
  exit 0
fi

# Risolve eventuali percorsi scenario relativi prima di spostarsi nella root.
if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
  TRAIN_ARGS=("$DEFAULT_SCENARIO" --checkpoint "$DEFAULT_CHECKPOINT")
elif [[ "${TRAIN_ARGS[0]}" == -* ]]; then
  TRAIN_ARGS=("$DEFAULT_SCENARIO" "${TRAIN_ARGS[@]}")
elif [[ -f "${TRAIN_ARGS[0]}" ]]; then
  TRAIN_ARGS[0]="$(realpath -- "${TRAIN_ARGS[0]}")"
fi

cd -- "$ROOT_DIR"
activate_conda_env "$CONDA_ENV_NAME"
exec python -m trainer "${TRAIN_ARGS[@]}"
