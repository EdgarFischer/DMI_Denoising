#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

CONFIG_PATH="${PROJECT_ROOT}/configs/train_physics_7T.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/PhysicsConv3D_7T_N2V_HealthyActive16_LCModelPriors.log"
PID_FILE="${LOG_DIR}/PhysicsConv3D_7T_N2V_HealthyActive16_LCModelPriors.pid"

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
    echo "[$(date --iso-8601=seconds)] Starting PhysicsConv3D 7T LCModel-kernel + exact forD baseline N2V"
    echo "Project: ${PROJECT_ROOT}"
    echo "Config:  ${CONFIG_PATH}"
    echo "Python:  ${PYTHON}"
    echo "Run:     PhysicsConv3D_7T_N2V_HealthyActive16_LCModelPriors_MS_180"
    echo "Data:    MS_180 / OriginalData/data_after_walinet.npy"
    echo "Baseline: forD 9 complex cubic B-splines (18 coefficients)"
    echo "Scale:    0.34802767666497547 (forD -> Denoising units)"
    echo "FWHM:     softplus (positive, unbounded, no logarithmic damping)"
    echo "Metabolite shifts: linear Hz (unbounded, no tanh saturation)"
    echo "Scaling:  shift 0 +/- 1 Hz; FWHM 5 +/- 2.5 Hz (z coordinates)"
    echo "Priors:   LCModel-style relative shift 0 +/- 0.2 Hz; extra FWHM 0 +/- 1 Hz"
    echo "Loss:     masked N2V MSE + 0.01 * each parameter-prior term"
    echo "Basis:    16 Healthy-active components, including Glc and TwoHG"
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
