#!/usr/bin/env bash

set -euo pipefail

# Simple initializer for a new uv-based project with Jupyter.
#
# Usage:
#   ./scripts/init_uv_project.sh my-project-uv "Python (my-project-uv)"
# or just:
#   ./scripts/init_uv_project.sh
# which will default to:
#   KERNEL_NAME="project-uv"
#   DISPLAY_NAME="Python (project-uv)"

KERNEL_NAME="${1:-project-uv}"
DISPLAY_NAME="${2:-Python (project-uv)}"

echo "==> Project directory: $(pwd)"
echo "==> Using kernel name: ${KERNEL_NAME}"
echo "==> Using display name: ${DISPLAY_NAME}"

echo "==> Creating uv virtual environment (.venv) if missing..."
uv venv

if [[ -f "pyproject.toml" ]]; then
  echo "==> Detected pyproject.toml — running 'uv sync'..."
  uv sync
elif [[ -f "requirements.txt" ]]; then
  echo "==> Detected requirements.txt — installing with 'uv pip install -r requirements.txt'..."
  uv pip install -r requirements.txt
else
  echo "==> No pyproject.toml or requirements.txt found. Skipping dependency install."
fi

echo "==> Installing Jupyter and ipykernel into the env..."
uv pip install jupyter ipykernel

echo "==> Registering Jupyter kernel '${KERNEL_NAME}'..."
uv run python -m ipykernel install --user \
  --name "${KERNEL_NAME}" \
  --display-name "${DISPLAY_NAME}"

echo "==> Done."
echo "Open your notebook and select the kernel named: ${DISPLAY_NAME}"

