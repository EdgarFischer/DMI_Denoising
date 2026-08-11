#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

CONFIG_PATH="${PROJECT_ROOT}/configs/train_physics_7T_phive.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_NAME="MS_180_N2S_AllReg0001"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "/home/hfischer/venvs/walinet/bin/python" ]]; then
    PYTHON="/home/hfischer/venvs/walinet/bin/python"
else
    PYTHON="$(command -v python3)"
fi

mkdir -p "${LOG_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

{
    echo "[$(date --iso-8601=seconds)] Starting regularized PhysicsConv3D 7T PHIVE control"
    echo "Project: ${PROJECT_ROOT}"
    echo "Config:  ${CONFIG_PATH}"
    echo "Python:  ${PYTHON}"
    echo "Run:     ${RUN_NAME}"
    echo "Data:    MS_180 / OriginalData/data_after_walinet.npy"
    echo "Input:   unchanged noisy spectrum"
    echo "Target:  identical full spectrum"
    echo "Loss:    |residual|^2 / sigma^2 from epoch 1; plain MSE logged separately"
    echo "Sigma:   per voxel std_f(|prediction-target|), detached (PHIVE-style)"
    echo "Context: encoder still receives the complete spectrum"
    echo "Masking: N2V/N2S masking disabled; masking config ignored"
    echo "Lineshape: one global shift + global Lorentz/Gauss FWHM per voxel"
    echo "Metabolite-specific shifts/FWHM: disabled"
    echo "Nuisance prior: 0.01 * sum(z^2) for global shift, Lorentz/Gauss FWHM, phase 0/1"
    echo "Nuisance statistics: original Vol1_Brisbane calibrated values"
    echo "Basis:    14 components; Glc and TwoHG disabled"
    echo "Curvature: kernel=0; complex 9-spline baseline=1000000"
    echo "Baseline init: exactly zero physical real/imaginary coefficients"
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
    echo "PHIVE control training started with PID ${PID}"
    echo "Log: ${LOG_FILE}"
    echo "Follow: tail -f ${LOG_FILE}"
else
    echo "Training stopped during startup. Check the log:" >&2
    echo "${LOG_FILE}" >&2
    exit 1
fi
