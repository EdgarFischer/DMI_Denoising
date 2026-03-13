#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="mrjo"
IMAGE_NAME="deepdenoising"

# Two levels above docker/  -> e.g. ~/Denoising
WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"

# Repo root = one level above docker/ -> e.g. ~/Denoising/DMI_Denoising
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"
REPO_IN_CONTAINER="/workspace/${REPO_NAME}"

read -rp "Activate Git integration? (y/N) " ENABLE_GIT

# Remove old container if it already exists
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

GPU_ARGS=(--gpus all)
BASE_ARGS=(
  -d
  --name "${CONTAINER_NAME}"
  -p 7017:8888
  -v "${WORKSPACE}:/workspace"
)

GIT_ARGS=()

if [[ "${ENABLE_GIT}" =~ ^[Yy]$ ]]; then
  echo "→ Enabling Git integration..."

  if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
    echo "⚠️ SSH_AUTH_SOCK is not set."
    echo "   Trying to start ssh-agent and load ~/.ssh/id_rsa ..."

    eval "$(ssh-agent -s)"

    if [[ -f "${HOME}/.ssh/id_rsa" ]]; then
      ssh-add "${HOME}/.ssh/id_rsa"
    else
      echo "✗ No SSH key found at ~/.ssh/id_rsa"
      exit 1
    fi
  fi

  if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
    echo "✗ SSH_AUTH_SOCK is still empty. Cannot enable Git integration."
    exit 1
  fi

  GIT_NAME="$(git config --get user.name || true)"
  GIT_EMAIL="$(git config --get user.email || true)"

  GIT_ARGS+=(
    -v "${SSH_AUTH_SOCK}:/ssh-agent.sock"
    -v "${HOME}/.ssh:/home/hostuser/.ssh:ro"
    -e SSH_AUTH_SOCK=/ssh-agent.sock
  )

  if [[ -n "${GIT_NAME}" ]]; then
    GIT_ARGS+=(-e GIT_AUTHOR_NAME="${GIT_NAME}" -e GIT_COMMITTER_NAME="${GIT_NAME}")
  fi

  if [[ -n "${GIT_EMAIL}" ]]; then
    GIT_ARGS+=(-e GIT_AUTHOR_EMAIL="${GIT_EMAIL}" -e GIT_COMMITTER_EMAIL="${GIT_EMAIL}")
  fi

  echo "✓ Git integration enabled"
else
  echo "→ Launching container without Git integration..."
fi

echo "Workspace mounted from: ${WORKSPACE}"
echo "Repo inside container:  ${REPO_IN_CONTAINER}"

echo "→ Trying to launch container with GPU support..."

if docker run "${BASE_ARGS[@]}" "${GPU_ARGS[@]}" "${GIT_ARGS[@]}" "${IMAGE_NAME}"; then
  echo "✅ Container '${CONTAINER_NAME}' started with GPU support."
else
  echo "⚠️ GPU not available, launching without GPU..."

  # Remove partially created container from failed GPU start
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  docker run "${BASE_ARGS[@]}" "${GIT_ARGS[@]}" "${IMAGE_NAME}"

  echo "✅ Container '${CONTAINER_NAME}' started without GPU."
fi

if [[ "${ENABLE_GIT}" =~ ^[Yy]$ ]]; then
  echo "→ Configuring Git safe.directory..."
  docker exec -u hostuser "${CONTAINER_NAME}" \
    git config --global --add safe.directory "${REPO_IN_CONTAINER}"
fi

echo "✅ Container '${CONTAINER_NAME}' is up and running."
echo "   Jupyter: http://localhost:7017"
echo "   Repo path inside container: ${REPO_IN_CONTAINER}"