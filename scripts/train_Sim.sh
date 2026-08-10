#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

CONFIG_PATH="${PROJECT_ROOT}/configs/train_physics_7T_phive_Sim.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_NAME="Sim_AllReg_0001"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"

# Use the active environment, otherwise the known WALINET environment. This
# avoids silently falling back to /usr/bin/python3 without hlsvdpropy.
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
    echo "[$(date --iso-8601=seconds)] Starting PhysicsConv3D 7T global-Voigt + exact forD baseline N2S"
    echo "Project: ${PROJECT_ROOT}"
    echo "Config:  ${CONFIG_PATH}"
    echo "Python:  ${PYTHON}"
    echo "Run:     ${RUN_NAME}"
    echo "Data:    MS_180 / OriginalData/data_after_walinet.npy"
    echo "Baseline: forD 9 complex cubic B-splines (18 coefficients)"
    echo "Scale:    0.34802767666497547 (forD -> Denoising units)"
    echo "Lineshape: one global shift + global Lorentz/Gauss FWHM per voxel"
    echo "Metabolite-specific shifts/FWHM: disabled"
    echo "Priors:   no shift/FWHM penalty; calibrated global Z-score coordinates"
    echo "Curvature: no kernel; complex 9-spline baseline=1000000"
    echo "Baseline init: exactly zero physical real/imaginary coefficients"
    echo "Loss:     epochs 1-100 plain masked MSE; from 101 |residual|^2 / sigma^2"
    echo "Sigma:    per voxel std_f(|prediction-target|), detached (PHIVE-style)"
    echo "Basis:    14 components; Glc and TwoHG disabled"
    echo "Start:    new random network (no resume/pretraining)"
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
    echo "Training started with PID ${PID}"
    echo "Log: ${LOG_FILE}"
    echo "Follow: tail -f ${LOG_FILE}"
else
    echo "Training stopped during startup. Check the log:" >&2
    echo "${LOG_FILE}" >&2
    exit 1
fi
