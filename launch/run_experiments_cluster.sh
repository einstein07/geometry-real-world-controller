#!/bin/bash
#
# Automated experiment runner.
# Launches ARGoS bridge + controllers for NUM_RUNS consecutive runs,
# cleaning up any dangling processes between runs.
#
# Usage: bash run_experiments.sh [num_runs]   (default: 20)
#

set -uo pipefail

# ─────────────────────── configuration ───────────────────────
NUM_RUNS="${1:-20}"
NUM_ROBOTS=20                  # must match 'n' in launch-controllers.sh and ARGoS world
RUN_TIMEOUT=3600              # seconds before a run is force-killed (60 min)
ARGOS_INIT_WAIT=4             # seconds to wait for ARGoS to initialise before launching controllers
BETWEEN_RUNS_WAIT=5           # seconds between consecutive runs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="/opt/ros2_ws"
# Source config — read-only inside the container; used only to seed a writable copy.
PARAMS_SOURCE="${SCRIPT_DIR}/../config/parameters.json"

BRIDGE_SCRIPT="${SCRIPT_DIR}/launch-argos-bridge-cluster.sh"
CONTROLLERS_SCRIPT="${SCRIPT_DIR}/launch-controllers-cluster.sh"
# ──────────────────────────────────────────────────────────────

# ──── writable params file ────────────────────────────────────
# The container filesystem at /opt/ is read-only (Apptainer on bwUniCluster).
# If a caller (e.g. update_rate_sweep_cluster.sh) already created a writable
# copy and exported PARAMS_FILE, reuse it.  Otherwise create one now.
_TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${_TMPDIR}"
if [ -z "${PARAMS_FILE:-}" ]; then
    PARAMS_FILE="$(mktemp --tmpdir="${_TMPDIR}" params.XXXXXX.json)"
    cp "${PARAMS_SOURCE}" "${PARAMS_FILE}"
    export PARAMS_FILE
    _OWN_PARAMS_FILE=1
else
    _OWN_PARAMS_FILE=0
fi
# ──────────────────────────────────────────────────────────────

# ─────────────────── source ROS2 workspace ───────────────────
# Must set COLCON_TRACE before sourcing setup.bash, and disable set -u
# temporarily — setup.bash references unset colcon variables that would
# otherwise abort the script and leave PYTHONPATH incomplete.
export COLCON_TRACE="${COLCON_TRACE:-}"
set +u
# shellcheck source=/dev/null
source "${WS_DIR}/install/setup.bash"
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
# ──────────────────────────────────────────────────────────────

log() { echo "[runner] $*"; }

# ──────────────────────── cleanup ────────────────────────────
cleanup() {
    log "Cleaning up dangling processes..."
    pkill -SIGINT -f argos3              2>/dev/null || true
    pkill -SIGINT -f controller_node     2>/dev/null || true
    pkill -SIGINT -f termination_monitor 2>/dev/null || true
    pkill -SIGINT -f "ros2 launch"       2>/dev/null || true
    sleep 2
    # Second pass with SIGKILL for anything still alive
    pkill -SIGKILL -f argos3              2>/dev/null || true
    pkill -SIGKILL -f controller_node     2>/dev/null || true
    pkill -SIGKILL -f termination_monitor 2>/dev/null || true
    sleep 1
    log "Cleanup done."
}

_exit_handler() {
    log "Interrupted — running final cleanup."
    cleanup
    if [ "${_OWN_PARAMS_FILE}" -eq 1 ]; then
        rm -f "${PARAMS_FILE}"
    fi
    exit 1
}
trap '_exit_handler' SIGINT SIGTERM
# ──────────────────────────────────────────────────────────────

# ─────────── update experiment_name in parameters.json ───────
# Writes only to the writable PARAMS_FILE — never to read-only paths.
set_experiment_name() {
    local name="$1"
    python3 - <<EOF
import json

with open("${PARAMS_FILE}", "r") as f:
    params = json.load(f)
params["experiment_name"] = name
with open("${PARAMS_FILE}", "w") as f:
    json.dump(params, f, indent=4)
EOF
}
# ──────────────────────────────────────────────────────────────

log "Starting automated experiment batch: ${NUM_RUNS} runs, ${NUM_ROBOTS} robots."
log "Timeout per run: ${RUN_TIMEOUT}s"
log "Using params file: ${PARAMS_FILE}"
log ""

FAILED_RUNS=()

for run in $(seq 1 "${NUM_RUNS}"); do
    RUN_LABEL="$(printf 'run_%03d' "${run}")"

    echo ""
    echo "============================================="
    log "Run ${run} / ${NUM_RUNS}  (${RUN_LABEL})"
    echo "============================================="

    # 1. Clean up anything left from the previous run
    cleanup

    # 2. Tag this run in parameters.json
    set_experiment_name "${RUN_LABEL}"
    log "Experiment name set to '${RUN_LABEL}'"

    # 3. Start ARGoS bridge in the background
    log "Launching ARGoS bridge..."
    bash "${BRIDGE_SCRIPT}" &
    ARGOS_PID=$!
    log "ARGoS started (PID ${ARGOS_PID})"

    # 4. Wait for ARGoS to initialise
    log "Waiting ${ARGOS_INIT_WAIT}s for ARGoS to initialise..."
    sleep "${ARGOS_INIT_WAIT}"

    # Check ARGoS actually started — if it died already, skip this run
    if ! kill -0 "${ARGOS_PID}" 2>/dev/null; then
        log "ERROR: ARGoS exited prematurely. Skipping run ${run}."
        FAILED_RUNS+=("${run}")
        continue
    fi

    # 5. Launch controllers (blocks until termination_monitor fires Shutdown())
    #    Wrapped in timeout so a hung experiment doesn't stall the batch.
    log "Launching controllers (timeout: ${RUN_TIMEOUT}s)..."
    if timeout "${RUN_TIMEOUT}" bash "${CONTROLLERS_SCRIPT}"; then
        log "Run ${run} finished cleanly."
    else
        EXIT_CODE=$?
        if [ "${EXIT_CODE}" -eq 124 ]; then
            log "WARNING: Run ${run} timed out after ${RUN_TIMEOUT}s."
        else
            log "WARNING: Run ${run} exited with code ${EXIT_CODE}."
        fi
        FAILED_RUNS+=("${run}")
    fi

    # 6. Final cleanup for this run
    cleanup

    # 7. Pause before next run (skip after last run)
    if [ "${run}" -lt "${NUM_RUNS}" ]; then
        log "Waiting ${BETWEEN_RUNS_WAIT}s before next run..."
        sleep "${BETWEEN_RUNS_WAIT}"
    fi
done

# ─────────────── summary ──────────────────────────────────────
echo ""
echo "============================================="
log "Batch complete. ${NUM_RUNS} runs attempted."
if [ "${#FAILED_RUNS[@]}" -eq 0 ]; then
    log "All runs finished cleanly."
else
    log "Runs with issues: ${FAILED_RUNS[*]}"
fi
echo "============================================="

if [ "${_OWN_PARAMS_FILE}" -eq 1 ]; then
    rm -f "${PARAMS_FILE}"
fi
