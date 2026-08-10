#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/configs/train_physics_7T_phive_simulation_snr9_gtstats.yaml"
RUN_NAME="Simulation_SNR9_PHIVE_GTStats"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"
DATA_FILE="${WORKSPACE_ROOT}/MRSISimulator/simulations/Simulation_SNR9/realization_000/simulation_fid_noisy.npy"
MASK_FILE="${WORKSPACE_ROOT}/MRSISimulator/simulations/Simulation_SNR9/realization_000/brain_mask.npy"
STATS_FILE="${PROJECT_ROOT}/configs/physics_parameter_stats_7T_simulation_gt.json"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "/home/hfischer/venvs/walinet/bin/python" ]]; then
    PYTHON="/home/hfischer/venvs/walinet/bin/python"
else
    PYTHON="$(command -v python3)"
fi

for required_file in "${CONFIG_PATH}" "${DATA_FILE}" "${MASK_FILE}" "${STATS_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file is missing: ${required_file}" >&2
        exit 1
    fi
done
mkdir -p "${LOG_DIR}"

{
    echo "[$(date --iso-8601=seconds)] Starting PhysicsConv3D PHIVE on SNR-9 simulation with GT-scale parameter statistics"
    echo "Project: ${PROJECT_ROOT}"
    echo "Config:  ${CONFIG_PATH}"
    echo "Python:  ${PYTHON}"
    echo "Run:     ${RUN_NAME}"
    echo "Input:   ${DATA_FILE}"
    echo "Mask:    ${MASK_FILE}"
    echo "Stats:   ${STATS_FILE}"
    echo "Target:  identical noisy SNR-9 spectrum"
    echo "Loss:    direct MSE over 1.8-4.2 ppm in brain voxels"
    echo "Normalization: preserve the existing shared clean-FID/GT-map scale"
    echo
} > "${LOG_FILE}"

nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PROJECT_ROOT}/src:${WORKSPACE_ROOT}/walinet/src:${WORKSPACE_ROOT}/mrs_utils${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -u "${PROJECT_ROOT}/scripts/train.py" \
    --config "${CONFIG_PATH}" \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"
sleep 1
if kill -0 "${PID}" 2>/dev/null; then
    echo "Simulation PHIVE GTStats training started with PID ${PID}"
    echo "Log: ${LOG_FILE}"
    echo "Follow: tail -f ${LOG_FILE}"
else
    echo "Training stopped during startup. Check: ${LOG_FILE}" >&2
    exit 1
fi
