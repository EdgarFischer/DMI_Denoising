#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/configs/train_physics_7T_phive_voigt.yaml"
RUN_NAME="MS_180_Phive_GlobalVoigt_BaselineReg_NoTwoHG"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
else
    PYTHON="/home/hfischer/venvs/walinet/bin/python"
fi
mkdir -p "${LOG_DIR}"

{
    echo "[$(date --iso-8601=seconds)] Starting PhysicsConv3D 7T PHIVE global Voigt"
    echo "Project:  ${PROJECT_ROOT}"
    echo "Config:   ${CONFIG_PATH}"
    echo "Python:   ${PYTHON}"
    echo "Run:      ${RUN_NAME}"
    echo "Model:    15 amplitudes + global shift/phase + global Lorentz/Gauss + 9+9 baseline"
    echo "Channels: 38 physical parameter maps per voxel"
    echo "Scaling:  global Lorentz/Gauss use in-vivo Z-score statistics"
    echo "Baseline: exact forD spline model; curvature weight=1000000; zero initialization"
    echo "Loss:     direct PHIVE MSE over 1.8-4.2 ppm"
    echo
} > "${LOG_FILE}"

nohup env PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PROJECT_ROOT}/src:${WORKSPACE_ROOT}/walinet/src:${WORKSPACE_ROOT}/mrs_utils${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -u "${PROJECT_ROOT}/scripts/train.py" --config "${CONFIG_PATH}" \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo "Training started with PID ${PID}"
echo "Follow: tail -f ${LOG_FILE}"
