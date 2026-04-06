#!/bin/bash
#
# Sweep the update_rate parameter over a fixed set of values and run a full
# experiment batch for each one using the existing run_experiments.sh script.
#
# Usage:
#   bash run_update_rate_sweep.sh [runs_per_rate] [log_root]
#
# Defaults:
#   runs_per_rate = 20
#   log_root      = /home/sindiso/geometry-logs/update-rate-sweep
#

set -euo pipefail

RUNS_PER_RATE="${1:-20}"
LOG_ROOT="${2:-/home/sindiso/geometry-logs/update-rate-sweep}"
RATES=(1 2 3 4 5 6 7 8 9 10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="/home/sindiso/ros2_ws"
PARAMS_SOURCE="${SCRIPT_DIR}/../config/parameters.json"
PARAMS_INSTALLED="${WS_DIR}/install/controller_real_world/share/controller_real_world/parameters.json"
RUN_SCRIPT="${SCRIPT_DIR}/run_experiments.sh"

log() { echo "[update-rate-sweep] $*"; }

backup_source="$(mktemp)"
cp "${PARAMS_SOURCE}" "${backup_source}"

backup_installed=""
if [ -f "${PARAMS_INSTALLED}" ]; then
    backup_installed="$(mktemp)"
    cp "${PARAMS_INSTALLED}" "${backup_installed}"
fi

restore_configs() {
    if [ -f "${backup_source}" ]; then
        cp "${backup_source}" "${PARAMS_SOURCE}"
        rm -f "${backup_source}"
    fi

    if [ -n "${backup_installed}" ] && [ -f "${backup_installed}" ] && [ -f "${PARAMS_INSTALLED}" ]; then
        cp "${backup_installed}" "${PARAMS_INSTALLED}"
        rm -f "${backup_installed}"
    fi
}

trap restore_configs EXIT INT TERM

update_params() {
    local rate="$1"
    local log_dir="$2"

    python3 - <<EOF
import json
from pathlib import Path

rate = int("${rate}")
log_dir = "${log_dir}"
paths = [Path("${PARAMS_SOURCE}")]
installed = Path("${PARAMS_INSTALLED}")
if installed.exists():
    paths.append(installed)

for path in paths:
    with path.open("r") as f:
        params = json.load(f)

    params["update_rate"] = rate
    params["log_directory"] = log_dir

    with path.open("w") as f:
        json.dump(params, f, indent=4)
        f.write("\n")
EOF
}

mkdir -p "${LOG_ROOT}"

log "Starting update_rate sweep."
log "Runs per rate: ${RUNS_PER_RATE}"
log "Log root: ${LOG_ROOT}"

FAILED_RATES=()

for rate in "${RATES[@]}"; do
    rate_dir="${LOG_ROOT}/update_rate_$(printf '%02d' "${rate}")"
    mkdir -p "${rate_dir}"

    echo ""
    echo "============================================================"
    log "Running batch for update_rate=${rate}"
    log "Logs will be written under ${rate_dir}"
    echo "============================================================"

    update_params "${rate}" "${rate_dir}"

    if bash "${RUN_SCRIPT}" "${RUNS_PER_RATE}"; then
        log "Completed batch for update_rate=${rate}"
    else
        log "Batch failed for update_rate=${rate}"
        FAILED_RATES+=("${rate}")
    fi
done

echo ""
echo "============================================================"
log "Sweep complete."
if [ "${#FAILED_RATES[@]}" -eq 0 ]; then
    log "All update_rate batches completed."
else
    log "Batches with failures: ${FAILED_RATES[*]}"
fi
echo "============================================================"
