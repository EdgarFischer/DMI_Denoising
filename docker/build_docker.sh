#!/usr/bin/env bash
set -e

# Move to the directory where this script lives
cd "$(dirname "$0")"

docker build \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  -t deepdenoising .
