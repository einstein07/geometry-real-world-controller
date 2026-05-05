#!/bin/bash
# =============================================================================
# Sweep submission script — bwUniCluster3 login node
#
# Submits run_experiments_cluster.sh as a SLURM job array.  Each task enters
# execution mode inside the container, handling one (parameter-combo,
# replicate-batch) pair.
#
# The container provides the ROS2/ARGoS runtime.  The launch scripts and
# config are bind-mounted from the shared filesystem at runtime, so sweep
# parameters and scripts can be edited without rebuilding the container.
#
# Usage (from login node):
#   bash submit_sweep.sh
#
# Override defaults with env vars:
#   CONTAINER=/path/to/container.sif LOG_ROOT=/scratch/my_data bash submit_sweep.sh
# =============================================================================

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
# PKG_DIR resolves to the geometry-real-world-controller directory on the
# shared filesystem, regardless of where this script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "${SCRIPT_DIR}")"   # .../geometry-real-world-controller

CONTAINER="${CONTAINER:-/home/kn/kn_kn/kn_pop547841/geometry-real-world-controller.sif}"
LOG_ROOT="${LOG_ROOT:-/pfs/work9/workspace/scratch/kn_pop547841-mySpace/ros2-data/param_sweep}"
# ─────────────────────────────────────────────────────────────────────────────

# ── sweep sizing — keep in sync with run_experiments_cluster.sh ──────────────
# N_PARAM_COMBOS = product of all array lengths in SWEEP_JSON.
N_PARAM_COMBOS=10    # 10 update_rate × 1 eta
RUNS_PER_COMBO=20
RUNS_PER_TASK=5
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR="${LOG_ROOT}/slurm_logs"
BATCHES_PER_COMBO=$(( RUNS_PER_COMBO / RUNS_PER_TASK ))
TOTAL_TASKS=$(( N_PARAM_COMBOS * BATCHES_PER_COMBO ))

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: container not found: ${CONTAINER}" >&2
    exit 1
fi

mkdir -p "${LOGS_DIR}"

echo "Submitting parameter sweep"
echo "  container  : ${CONTAINER}"
echo "  pkg bind   : ${PKG_DIR} → /opt/ros2_ws/src/geometry-real-world-controller"
echo "  log root   : ${LOG_ROOT}"
echo "  combos     : ${N_PARAM_COMBOS}"
echo "  tasks      : ${TOTAL_TASKS}  (${N_PARAM_COMBOS} combos × ${BATCHES_PER_COMBO} batches)"
echo ""

# --bind overlays the local package directory onto the container's copy.
# The container's ROS2/ARGoS runtime is unchanged; only the scripts and config
# are replaced with the live versions from the shared filesystem.
sbatch \
    --job-name=ros2_param_sweep \
    --partition=cpu \
    --time=24:00:00 \
    --mem=8G \
    --cpus-per-task=2 \
    --array="0-$(( TOTAL_TASKS - 1 ))%50" \
    --output="${LOGS_DIR}/sweep_%A_%a.out" \
    --error="${LOGS_DIR}/sweep_%A_%a.err" \
    --export=ALL,LOG_ROOT="${LOG_ROOT}" \
    --wrap="apptainer exec \
        --bind ${PKG_DIR}:/opt/ros2_ws/src/geometry-real-world-controller \
        ${CONTAINER} \
        /opt/ros2_ws/src/geometry-real-world-controller/launch/run_experiments_cluster.sh"
