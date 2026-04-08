#!/bin/bash
#
# Sweep the update_rate parameter over a fixed set of values and run a full
# experiment batch for each one using the existing run_experiments.sh script.
#
# Usage:
#   bash run_update_rate_sweep.sh [runs_per_rate] [log_root]
#
# Defaults:
#   runs_per_rate = 1
#   log_root      = /mnt
#

set -euo pipefail

# bwUniCluster/Apptainer-safe temp directory handling.
# Slurm may export TMPDIR to a host scratch path that does not exist inside
# the container, which causes plain `mktemp` to fail. Fall back to /tmp.
if [ -n "${TMPDIR:-}" ] && [ -d "${TMPDIR}" ]; then
    :
else
    export TMPDIR=/tmp
fi
mkdir -p "${TMPDIR}"

RUNS_PER_RATE="${1:-20}"
LOG_ROOT="${2:-/mnt}"
RATES=(1 2 3 4 5 6 7 8 9 10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Source config — read-only inside the container; never written to.
PARAMS_SOURCE="${SCRIPT_DIR}/../config/parameters.json"
RUN_SCRIPT="${SCRIPT_DIR}/run_experiments_cluster.sh"

log() { echo "[update-rate-sweep] $*"; }

# ── runtime dependency check ──────────────────────────────────
# tf_transformations may be missing from older container builds.
# Install it to a writable directory in TMPDIR and add it to PYTHONPATH
# so controller nodes can import it without touching the read-only /opt/.
if ! python3 -c "import tf_transformations; import numpy" 2>/dev/null; then
    log "Missing Python deps — installing tf-transformations + numpy to ${TMPDIR}/pylocal ..."
    TF_PKG_DIR="${TMPDIR}/pylocal"
    mkdir -p "${TF_PKG_DIR}"
    pip install tf-transformations numpy --target "${TF_PKG_DIR}" --quiet
    export PYTHONPATH="${TF_PKG_DIR}:${PYTHONPATH:-}"
    log "Dependencies installed. PYTHONPATH updated."
fi
# ──────────────────────────────────────────────────────────────

# Create ONE writable working copy in TMPDIR for the entire sweep.
# The container filesystem at /opt/ is read-only (Apptainer on bwUniCluster),
# so we must never write back to PARAMS_SOURCE or the installed share copy.
WORKING_PARAMS="$(mktemp --tmpdir="${TMPDIR}" params.XXXXXX.json)"
cp "${PARAMS_SOURCE}" "${WORKING_PARAMS}"
export PARAMS_FILE="${WORKING_PARAMS}"
log "Working params file: ${WORKING_PARAMS}"

cleanup_working_params() {
    rm -f "${WORKING_PARAMS}"
}
trap cleanup_working_params EXIT INT TERM

update_params() {
    local rate="$1"
    local log_dir="$2"

    python3 - <<EOF2
import json
from pathlib import Path

rate = int("${rate}")
log_dir = "${log_dir}"
path = Path("${WORKING_PARAMS}")

with path.open("r") as f:
    params = json.load(f)

params["update_rate"] = rate
params["log_directory"] = log_dir

with path.open("w") as f:
    json.dump(params, f, indent=4)
    f.write("\n")
EOF2
}

mkdir -p "${LOG_ROOT}"

log "Starting update_rate sweep."
log "Runs per rate: ${RUNS_PER_RATE}"
log "Log root: ${LOG_ROOT}"
log "Using TMPDIR: ${TMPDIR}"

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
