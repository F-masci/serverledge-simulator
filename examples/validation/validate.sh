#!/usr/bin/env bash
set -euo pipefail

# Orchestratore di validazione: avvia il lab Serverledge reale sull'host,
# esegue il simulatore e il sistema reale sulla stessa traccia campionata
# e produce il confronto affiancato delle metriche.

VALIDATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$VALIDATION_DIR/../.." && pwd)"
DEFAULT_SCENARIO="$ROOT_DIR/examples/scenarios/validation.toml"

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
  local env_name="${CONDA_ENV_NAME}"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
    return 0
  fi

  local conda_bin
  if ! conda_bin="$(find_conda)"; then
    echo "Errore: conda non trovato nel PATH né nei percorsi standard." >&2
    echo "Imposta CONDA_EXE o attiva manualmente l'ambiente '$env_name'." >&2
    exit 1
  fi

  eval "$("$conda_bin" shell.bash hook)"
  conda activate "$env_name"
}

usage() {
  cat <<USAGE
Uso:
  validate.sh [run] [OPZIONI]
  validate.sh up [--local-etcd] [--no-build] [opzioni docker compose...]
  validate.sh down [--local-etcd] [opzioni docker compose...]
  validate.sh register [--update]
  validate.sh simulator [scenario] [--arrivals N] [...]
  validate.sh replay [--scenario FILE] [--arrivals N] [--output FILE] [--dry-run] [...]
  validate.sh compare <risultati.jsonl> [--scenario FILE]
  validate.sh ps | logs [servizi...] | compose <argomenti...>

Opzioni per 'run' (validazione end-to-end):
  --scenario FILE       Scenario da validare (default: examples/scenarios/validation.toml)
  --arrivals N          Numero di arrivi da generare e riprodurre (default: 100)
  --concurrency N       Massimo richieste HTTP contemporanee nel replay (default: 32)
  --timeout S           Timeout HTTP in secondi per singola richiesta (default: 30)
  --url URL             URL del load balancer Serverledge (default: http://127.0.0.1:13231)
  --output-dir DIR      Cartella in cui salvare simulator.txt, real.jsonl, comparison.txt e report.html
  --settle-s S          Attesa in secondi dopo l'avvio per il discovery in etcd (default: 3)
  --local-etcd          Avvia un container etcd locale su 127.0.0.1:2379 invece dell'etcd remoto
  --no-build            Salta il rebuild dell'immagine Docker di Serverledge
  --skip-up             Presume che il lab Docker Compose sia già avviato
  --skip-register       Salta la registrazione delle funzioni sintetiche sim_*
  --down                Ferma e rimuove i container del lab al termine della validazione
  --no-open             Non aprire automaticamente report.html nel browser al termine
  -e, --env NOME        Ambiente Conda da usare (default: ${CONDA_ENV_NAME})
  -h, --help            Mostra questo messaggio

Esempi:
  ./examples/validation/validate.sh run --arrivals 100
  ./examples/validation/validate.sh run --local-etcd --arrivals 50 --down
  ./examples/validation/validate.sh replay --arrivals 100 --dry-run
  ./examples/validation/validate.sh compare examples/validation/results/.../real.jsonl
USAGE
}

CONDA_ENV_NAME="${CONDA_ENV:-$(detect_default_env)}"
CLEANUP_DO_DOWN=0
CLEANUP_SKIP_UP=0
CLEANUP_LOCAL_ETCD=0

cleanup() {
  if [[ "${CLEANUP_DO_DOWN}" -eq 1 && "${CLEANUP_SKIP_UP}" -eq 0 ]]; then
    echo ""
    echo "==> Arresto del lab Serverledge..."
    run_compose "${CLEANUP_LOCAL_ETCD}" down
  fi
}

open_html_report() {
  local html_file="$1"
  if command -v xdg-open >/dev/null 2>&1; then
    nohup xdg-open "$html_file" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then
    nohup open "$html_file" >/dev/null 2>&1 &
  else
    python -m webbrowser "file://$html_file" >/dev/null 2>&1 || true
  fi
}

run_compose() {
  local local_etcd="$1"
  shift
  local args=()
  if [[ "$local_etcd" -eq 1 ]]; then
    export LAB_ETCD_ADDRESS="127.0.0.1:2379"
    args+=(--profile local-etcd)
  fi
  (cd -- "$VALIDATION_DIR" && docker compose "${args[@]}" "$@")
}

cmd_run() {
  local scenario="$DEFAULT_SCENARIO"
  local arrivals="100"
  local concurrency="32"
  local timeout_s="30"
  local url="http://127.0.0.1:13231"
  local output_dir=""
  local settle_s="3"
  local local_etcd=0
  local do_build=1
  local skip_up=0
  local skip_register=0
  local do_down=0
  local auto_open=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --scenario) scenario="$(realpath -- "$2")"; shift 2 ;;
      --arrivals) arrivals="$2"; shift 2 ;;
      --concurrency) concurrency="$2"; shift 2 ;;
      --timeout) timeout_s="$2"; shift 2 ;;
      --url) url="$2"; shift 2 ;;
      --output-dir) output_dir="$2"; shift 2 ;;
      --settle-s) settle_s="$2"; shift 2 ;;
      --local-etcd) local_etcd=1; shift ;;
      --no-build) do_build=0; shift ;;
      --skip-up) skip_up=1; shift ;;
      --skip-register) skip_register=1; shift ;;
      --down) do_down=1; shift ;;
      --no-open) auto_open=0; shift ;;
      -e|--env) CONDA_ENV_NAME="$2"; shift 2 ;;
      --env=*) CONDA_ENV_NAME="${1#*=}"; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Opzione sconosciuta per 'run': $1" >&2; exit 2 ;;
    esac
  done

  if [[ ! -f "$scenario" ]]; then
    echo "Errore: scenario non trovato: $scenario" >&2
    exit 2
  fi

  if [[ -z "$output_dir" ]]; then
    output_dir="$VALIDATION_DIR/results/$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "$output_dir"
  output_dir="$(realpath -- "$output_dir")"

  local sim_report="$output_dir/simulator.txt"
  local real_jsonl="$output_dir/real.jsonl"
  local cmp_report="$output_dir/comparison.txt"
  local html_report="$output_dir/report.html"

  CLEANUP_DO_DOWN="$do_down"
  CLEANUP_SKIP_UP="$skip_up"
  CLEANUP_LOCAL_ETCD="$local_etcd"
  trap cleanup EXIT

  activate_conda_env
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

  echo "=== Validazione Serverledge Simulator vs Sistema Reale ==="
  echo "Scenario:   $scenario"
  echo "Arrivi:     $arrivals"
  echo "Output dir: $output_dir"
  echo ""

  if [[ "$skip_up" -eq 0 ]]; then
    echo "==> [1/5] Avvio del lab Serverledge reale (Docker Compose)..."
    local up_args=(up --wait)
    if [[ "$do_build" -eq 1 ]]; then
      up_args=(up --build --wait)
    fi
    if [[ "$local_etcd" -eq 1 ]]; then
      # Avvia prima etcd locale così i worker lo trovano subito pronto.
      run_compose 1 up -d --wait etcd
    fi
    run_compose "$local_etcd" "${up_args[@]}"
    if [[ "$settle_s" != "0" ]]; then
      echo "Attesa ${settle_s}s per il discovery dei nodi in etcd..."
      sleep "$settle_s"
    fi
  else
    echo "==> [1/5] Avvio lab saltato (--skip-up)."
  fi

  if [[ "$skip_register" -eq 0 ]]; then
    echo ""
    echo "==> [2/5] Registrazione funzioni sintetiche (sim_*)..."
    if [[ "$local_etcd" -eq 1 ]]; then
      export LAB_ETCD_ADDRESS="127.0.0.1:2379"
    fi
    (cd -- "$VALIDATION_DIR" && bash register-functions.sh --update)
  else
    echo ""
    echo "==> [2/5] Registrazione funzioni saltata (--skip-register)."
  fi

  echo ""
  echo "==> [3/5] Esecuzione del simulatore..."
  (cd -- "$ROOT_DIR" && python -m simulator "$scenario" --arrivals "$arrivals") | tee "$sim_report"

  echo ""
  echo "==> [4/5] Replay della stessa traccia sul sistema Serverledge reale..."
  python "$VALIDATION_DIR/replay.py" \
    --scenario "$scenario" \
    --arrivals "$arrivals" \
    --concurrency "$concurrency" \
    --timeout "$timeout_s" \
    --url "$url" \
    --output "$real_jsonl"

  echo ""
  echo "==> [5/5] Confronto simulatore vs sistema reale..."
  python "$VALIDATION_DIR/compare.py" "$real_jsonl" --scenario "$scenario" --html "$html_report" | tee "$cmp_report"

  echo ""
  echo "Risultati salvati in: $output_dir"
  echo "Report HTML:          $html_report"
  if [[ "$auto_open" -eq 1 ]]; then
    open_html_report "$html_report"
  fi
  if [[ "$do_down" -eq 0 && "$skip_up" -eq 0 ]]; then
    echo "Il lab Serverledge è ancora attivo. Per fermarlo: ./examples/validation/validate.sh down"
  fi
}

cmd_up() {
  local local_etcd=0
  local do_build=1
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --local-etcd) local_etcd=1; shift ;;
      --no-build) do_build=0; shift ;;
      *) extra+=("$1"); shift ;;
    esac
  done
  if [[ "$local_etcd" -eq 1 ]]; then
    run_compose 1 up -d --wait etcd
  fi
  if [[ ${#extra[@]} -eq 0 ]]; then
    if [[ "$do_build" -eq 1 ]]; then
      run_compose "$local_etcd" up --build --wait
    else
      run_compose "$local_etcd" up --wait
    fi
  else
    run_compose "$local_etcd" up "${extra[@]}"
  fi
}

cmd_down() {
  local local_etcd=0
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --local-etcd) local_etcd=1; shift ;;
      *) extra+=("$1"); shift ;;
    esac
  done
  # Usa sempre --profile local-etcd in down così ferma anche etcd se era stato avviato.
  (cd -- "$VALIDATION_DIR" && docker compose --profile local-etcd down "${extra[@]}")
}

SUBCOMMAND="${1:-run}"
if [[ "$SUBCOMMAND" == -* ]]; then
  case "$SUBCOMMAND" in
    -h|--help) usage; exit 0 ;;
    *) cmd_run "$@"; exit 0 ;;
  esac
fi
if [[ $# -gt 0 ]]; then
  shift
fi

case "$SUBCOMMAND" in
  run)
    cmd_run "$@"
    ;;
  up)
    cmd_up "$@"
    ;;
  down)
    cmd_down "$@"
    ;;
  register)
    (cd -- "$VALIDATION_DIR" && exec bash register-functions.sh "$@")
    ;;
  simulator)
    if [[ $# -eq 0 ]]; then
      set -- "$DEFAULT_SCENARIO" --arrivals 100
    elif [[ "$1" == -* ]]; then
      set -- "$DEFAULT_SCENARIO" "$@"
    elif [[ -f "$1" ]]; then
      scenario="$(realpath -- "$1")"
      shift
      set -- "$scenario" "$@"
    fi
    activate_conda_env
    cd -- "$ROOT_DIR"
    exec python -m simulator "$@"
    ;;
  replay|load)
    activate_conda_env
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    exec python "$VALIDATION_DIR/replay.py" "$@"
    ;;
  compare)
    activate_conda_env
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    exec python "$VALIDATION_DIR/compare.py" "$@"
    ;;
  ps|logs)
    (cd -- "$VALIDATION_DIR" && exec docker compose --profile local-etcd "$SUBCOMMAND" "$@")
    ;;
  compose)
    (cd -- "$VALIDATION_DIR" && exec docker compose "$@")
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Sottocomando sconosciuto: $SUBCOMMAND. Usa --help per vedere le opzioni." >&2
    exit 2
    ;;
esac
