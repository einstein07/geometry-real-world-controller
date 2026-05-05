#!/bin/bash
# =============================================================================
# Parameter-sweep experiment runner — bwUniCluster3 SLURM job array
#
# Runs a full ARGoS + ROS2 experiment batch for every combination of the
# parameters defined in the SWEEP PARAMETERS section below, using a SLURM
# job array for parallelism.  Each array task handles one
# (parameter-combo, replicate-batch) pair.
#
# Task ID decomposition:
#   combo_idx = TASK_ID / BATCHES_PER_COMBO   (indexes Cartesian product)
#   batch_idx = TASK_ID % BATCHES_PER_COMBO   (replicate batch within combo)
# The Cartesian product rows are in row-major order (last key varies fastest).
#
# Output structure:
#   LOG_ROOT/
#     <param1>_<val1>/<param2>_<val2>/
#       run_001/  run_002/  ...  run_020/
#     slurm_logs/
#       sweep_<jobid>_<taskid>.{out,err}
#
# Dual-mode (no separate job file needed):
#   No SLURM_ARRAY_TASK_ID, no caller-set PARAMS_FILE, no positional args
#       → submission mode: print grid summary, then sbatch "$0"
#   SLURM_ARRAY_TASK_ID set
#       → execution mode: decompose task, patch params, run experiments
#   PARAMS_FILE already set by caller  OR  positional arg given
#       → direct mode: legacy single-batch behaviour (used by
#         update_rate_sweep_cluster.sh and similar callers)
#
# Usage (login node, sweep mode):
#   LOG_ROOT=/scratch/my_data bash run_experiments_cluster.sh
#
# Usage (legacy / called by outer sweep script):
#   bash run_experiments_cluster.sh [num_runs]
# =============================================================================

#SBATCH --job-name=ros2_param_sweep
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
# --output and --error are set dynamically below (they reference LOGS_DIR,
# which #SBATCH directives cannot evaluate as shell variables).

set -uo pipefail

# ─────────────────────────── paths ───────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="/opt/ros2_ws"
PARAMS_SOURCE="${SCRIPT_DIR}/../config/parameters.json"
BRIDGE_SCRIPT="${SCRIPT_DIR}/launch-argos-bridge-cluster.sh"
CONTROLLERS_SCRIPT="${SCRIPT_DIR}/launch-controllers-cluster.sh"
# ─────────────────────────────────────────────────────────────────

# ─────────────── SWEEP PARAMETERS — edit here ────────────────────
# SWEEP_JSON: JSON object mapping parameter names (must be top-level keys in
# parameters.json) to arrays of values.  The full Cartesian product is swept;
# the last key in the object varies fastest across SLURM task IDs.
SWEEP_JSON='{
    "update_rate": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "eta": [0.1]
}'

RUNS_PER_COMBO=20    # total replicates per parameter combination
RUNS_PER_TASK=5      # consecutive replicates handled by each SLURM task
                     # RUNS_PER_COMBO must be divisible by RUNS_PER_TASK

LOG_ROOT="${LOG_ROOT:-/mnt/param_sweep}"
LOGS_DIR="${LOG_ROOT}/slurm_logs"
# ─────────────────────────────────────────────────────────────────

# ─────────────── fixed experiment settings ───────────────────────
NUM_ROBOTS=20
RUN_TIMEOUT=3600
ARGOS_INIT_WAIT=4
BETWEEN_RUNS_WAIT=5
# ─────────────────────────────────────────────────────────────────

# ─────────────── Python binary ───────────────────────────────────
_PYTHON_BIN=""
for _cand in python3 python; do
    if command -v "$_cand" >/dev/null 2>&1; then _PYTHON_BIN="$_cand"; break; fi
done
[ -z "$_PYTHON_BIN" ] && { echo "Python not found." >&2; exit 1; }
# ─────────────────────────────────────────────────────────────────

# ─────────────── mode detection ──────────────────────────────────
# direct mode: caller exported PARAMS_FILE, or a positional arg (num_runs) given
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    _MODE=execute
elif [ -n "${PARAMS_FILE:-}" ] || [ -n "${1:-}" ]; then
    _MODE=direct
else
    _MODE=submit
fi
# ─────────────────────────────────────────────────────────────────

# ─────────────── compute grid size (needed by submit + execute) ──
read -r _N_COMBOS _BATCHES_PER_COMBO _TOTAL_TASKS <<< "$("$_PYTHON_BIN" - <<PYEOF
import json
from itertools import product as iproduct

sweep = json.loads("""$SWEEP_JSON""")
n_combos = 1
for v in sweep.values():
    n_combos *= len(v)
runs_per_combo = int("$RUNS_PER_COMBO")
runs_per_task  = int("$RUNS_PER_TASK")
if runs_per_combo % runs_per_task != 0:
    raise SystemExit(f"RUNS_PER_COMBO ({runs_per_combo}) must be divisible by RUNS_PER_TASK ({runs_per_task})")
batches = runs_per_combo // runs_per_task
print(n_combos, batches, n_combos * batches)
PYEOF
)"
# ─────────────────────────────────────────────────────────────────

# ═══════════════════════ SUBMISSION MODE ═════════════════════════
if [ "$_MODE" = submit ]; then
    "$_PYTHON_BIN" - <<PYEOF
import json
from itertools import product as iproduct

sweep = json.loads("""$SWEEP_JSON""")
print("Parameter sweep — ARGoS + ROS2 experiments")
print(f"  sweep params     : {' × '.join(sweep.keys())}")
for k, v in sweep.items():
    print(f"  {k} ({len(v)} values) : {v}")
print(f"  runs_per_combo   : $RUNS_PER_COMBO")
print(f"  runs_per_task    : $RUNS_PER_TASK  (consecutive replicates per task)")
print(f"  batches_per_combo: $_BATCHES_PER_COMBO")
print(f"  total combos     : $_N_COMBOS")
print(f"  total array tasks: $_TOTAL_TASKS")
print(f"  log root         : $LOG_ROOT")
print()
print("Parameter combinations:")
keys = list(sweep.keys())
arrays = [sweep[k] for k in keys]
for combo in iproduct(*arrays):
    parts = "  /  ".join(f"{k}={v}" for k, v in zip(keys, combo))
    print(f"  {parts}")
PYEOF

    mkdir -p "${LOGS_DIR}"

    sbatch \
        --array="0-$(( _TOTAL_TASKS - 1 ))%50" \
        --output="${LOGS_DIR}/sweep_%A_%a.out" \
        --error="${LOGS_DIR}/sweep_%A_%a.err" \
        --export=ALL,LOG_ROOT="${LOG_ROOT}" \
        "$0"
    exit 0
fi
# ═════════════════════════════════════════════════════════════════

# ─────────── shared setup (direct + execute modes) ───────────────
# bwUniCluster/Apptainer-safe temp directory
_TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${_TMPDIR}"

# Source ROS2 workspace.  Disable set -u temporarily — setup.bash references
# unset colcon variables that would otherwise abort the script.
export COLCON_TRACE="${COLCON_TRACE:-}"
set +u
# shellcheck source=/dev/null
source "${WS_DIR}/install/setup.bash"
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0

log() { echo "[runner] $*"; }

cleanup() {
    log "Cleaning up dangling processes..."
    pkill -SIGINT -f argos3              2>/dev/null || true
    pkill -SIGINT -f controller_node     2>/dev/null || true
    pkill -SIGINT -f termination_monitor 2>/dev/null || true
    pkill -SIGINT -f "ros2 launch"       2>/dev/null || true
    sleep 2
    pkill -SIGKILL -f argos3              2>/dev/null || true
    pkill -SIGKILL -f controller_node     2>/dev/null || true
    pkill -SIGKILL -f termination_monitor 2>/dev/null || true
    sleep 1
    log "Cleanup done."
}
# ─────────────────────────────────────────────────────────────────

# ═══════════════════════ DIRECT MODE (legacy) ════════════════════
if [ "$_MODE" = direct ]; then
    NUM_RUNS="${1:-20}"

    # Create writable params copy if caller did not provide one.
    if [ -z "${PARAMS_FILE:-}" ]; then
        PARAMS_FILE="$(mktemp --tmpdir="${_TMPDIR}" params.XXXXXX.json)"
        cp "${PARAMS_SOURCE}" "${PARAMS_FILE}"
        export PARAMS_FILE
        _OWN_PARAMS_FILE=1
    else
        _OWN_PARAMS_FILE=0
    fi

    set_experiment_name() {
        local name="$1"
        "$_PYTHON_BIN" - <<EOF
import json
with open("${PARAMS_FILE}", "r") as f:
    params = json.load(f)
params["experiment_name"] = "${name}"
with open("${PARAMS_FILE}", "w") as f:
    json.dump(params, f, indent=4)
EOF
    }

    _exit_handler() {
        log "Interrupted — running final cleanup."
        cleanup
        [ "${_OWN_PARAMS_FILE}" -eq 1 ] && rm -f "${PARAMS_FILE}"
        exit 1
    }
    trap '_exit_handler' SIGINT SIGTERM

    log "Starting automated experiment batch: ${NUM_RUNS} runs, ${NUM_ROBOTS} robots."
    log "Timeout per run: ${RUN_TIMEOUT}s"
    log "Using params file: ${PARAMS_FILE}"

    FAILED_RUNS=()

    for run in $(seq 1 "${NUM_RUNS}"); do
        RUN_LABEL="$(printf 'run_%03d' "${run}")"
        echo ""
        echo "============================================="
        log "Run ${run} / ${NUM_RUNS}  (${RUN_LABEL})"
        echo "============================================="
        cleanup
        set_experiment_name "${RUN_LABEL}"
        log "Experiment name set to '${RUN_LABEL}'"

        log "Launching ARGoS bridge..."
        bash "${BRIDGE_SCRIPT}" &
        ARGOS_PID=$!
        log "ARGoS started (PID ${ARGOS_PID})"
        log "Waiting ${ARGOS_INIT_WAIT}s for ARGoS to initialise..."
        sleep "${ARGOS_INIT_WAIT}"

        if ! kill -0 "${ARGOS_PID}" 2>/dev/null; then
            log "ERROR: ARGoS exited prematurely. Skipping run ${run}."
            FAILED_RUNS+=("${run}")
            continue
        fi

        log "Launching controllers (timeout: ${RUN_TIMEOUT}s)..."
        if timeout "${RUN_TIMEOUT}" bash "${CONTROLLERS_SCRIPT}"; then
            log "Run ${run} finished cleanly."
        else
            EXIT_CODE=$?
            [ "${EXIT_CODE}" -eq 124 ] \
                && log "WARNING: Run ${run} timed out after ${RUN_TIMEOUT}s." \
                || log "WARNING: Run ${run} exited with code ${EXIT_CODE}."
            FAILED_RUNS+=("${run}")
        fi

        cleanup
        [ "${run}" -lt "${NUM_RUNS}" ] && sleep "${BETWEEN_RUNS_WAIT}"
    done

    echo ""
    echo "============================================="
    log "Batch complete. ${NUM_RUNS} runs attempted."
    if [ "${#FAILED_RUNS[@]}" -eq 0 ]; then
        log "All runs finished cleanly."
    else
        log "Runs with issues: ${FAILED_RUNS[*]}"
    fi
    echo "============================================="

    [ "${_OWN_PARAMS_FILE}" -eq 1 ] && rm -f "${PARAMS_FILE}"
    exit 0
fi
# ═════════════════════════════════════════════════════════════════

# ═══════════════════════ EXECUTION MODE (SLURM) ══════════════════
# Decompose SLURM_ARRAY_TASK_ID → (combo_idx, batch_idx) → param values
TASK_ID="${SLURM_ARRAY_TASK_ID}"

# Isolate each task's ROS2 graph from all other tasks running on the same node.
# Valid range is 0–232; modulo keeps us within it regardless of array size.
export ROS_DOMAIN_ID=$(( TASK_ID % 232 ))
COMBO_IDX=$(( TASK_ID / _BATCHES_PER_COMBO ))
BATCH_IDX=$(( TASK_ID % _BATCHES_PER_COMBO ))
FIRST_RUN=$(( BATCH_IDX * RUNS_PER_TASK + 1 ))
LAST_RUN=$(( FIRST_RUN + RUNS_PER_TASK - 1 ))

# Decode combo_idx → param dict, output path, and patch parameters.json.
# Python writes the patched file to stdout as the path; params written to file.
PARAMS_FILE="$(mktemp --tmpdir="${_TMPDIR}" params.XXXXXX.json)"
export PARAMS_FILE

TASK_OUTPUT_DIR="$("$_PYTHON_BIN" - <<PYEOF
import json, os
from itertools import product as iproduct

sweep = json.loads("""$SWEEP_JSON""")
keys   = list(sweep.keys())
arrays = [sweep[k] for k in keys]
combos = list(iproduct(*arrays))
combo  = dict(zip(keys, combos[$COMBO_IDX]))

# Structured output path: LOG_ROOT / key_val / key_val / ...
path_parts = [f"{k}_{v}" for k, v in combo.items()]
task_dir   = os.path.join("$LOG_ROOT", *path_parts)
os.makedirs(task_dir, exist_ok=True)

# Patch parameters.json: apply sweep values and set log_directory.
with open("$PARAMS_SOURCE") as f:
    params = json.load(f)
params.update(combo)
params["log_directory"] = task_dir

with open("$PARAMS_FILE", "w") as f:
    json.dump(params, f, indent=4)
    f.write("\n")

print(task_dir)
PYEOF
)"

log "[task ${TASK_ID}] combo=${COMBO_IDX}  batch=${BATCH_IDX}  runs=${FIRST_RUN}–${LAST_RUN}"
log "[task ${TASK_ID}] output dir: ${TASK_OUTPUT_DIR}"

_exit_handler() {
    log "Interrupted — running final cleanup."
    cleanup
    rm -f "${PARAMS_FILE}"
    exit 1
}
trap '_exit_handler' SIGINT SIGTERM

FAILED_RUNS=()

for run in $(seq "${FIRST_RUN}" "${LAST_RUN}"); do
    RUN_LABEL="$(printf 'run_%03d' "${run}")"

    echo ""
    echo "============================================="
    log "[task ${TASK_ID}] Run ${run}  (${RUN_LABEL})"
    echo "============================================="

    cleanup

    # Update experiment_name for this replicate.
    "$_PYTHON_BIN" - <<EOF
import json
with open("${PARAMS_FILE}") as f:
    params = json.load(f)
params["experiment_name"] = "${RUN_LABEL}"
with open("${PARAMS_FILE}", "w") as f:
    json.dump(params, f, indent=4)
    f.write("\n")
EOF
    log "Experiment name set to '${RUN_LABEL}'"

    log "Launching ARGoS bridge..."
    bash "${BRIDGE_SCRIPT}" &
    ARGOS_PID=$!
    log "ARGoS started (PID ${ARGOS_PID})"
    log "Waiting ${ARGOS_INIT_WAIT}s for ARGoS to initialise..."
    sleep "${ARGOS_INIT_WAIT}"

    if ! kill -0 "${ARGOS_PID}" 2>/dev/null; then
        log "ERROR: ARGoS exited prematurely. Skipping run ${run}."
        FAILED_RUNS+=("${run}")
        continue
    fi

    log "Launching controllers (timeout: ${RUN_TIMEOUT}s)..."
    if timeout "${RUN_TIMEOUT}" bash "${CONTROLLERS_SCRIPT}"; then
        log "Run ${run} finished cleanly."
    else
        EXIT_CODE=$?
        [ "${EXIT_CODE}" -eq 124 ] \
            && log "WARNING: Run ${run} timed out after ${RUN_TIMEOUT}s." \
            || log "WARNING: Run ${run} exited with code ${EXIT_CODE}."
        FAILED_RUNS+=("${run}")
    fi

    cleanup
    [ "${run}" -lt "${LAST_RUN}" ] && sleep "${BETWEEN_RUNS_WAIT}"
done

echo ""
echo "============================================="
log "[task ${TASK_ID}] Batch complete. Runs ${FIRST_RUN}–${LAST_RUN} attempted."
if [ "${#FAILED_RUNS[@]}" -eq 0 ]; then
    log "All runs finished cleanly."
else
    log "Runs with issues: ${FAILED_RUNS[*]}"
fi
echo "============================================="

rm -f "${PARAMS_FILE}"
