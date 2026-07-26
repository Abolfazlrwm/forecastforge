#!/usr/bin/env bash
# bootstrap.sh — One-shot local environment setup for ForecastForge (POSIX).
#
# Responsibilities (and only these):
#   1. Create a virtual environment.
#   2. Install the package (editable, with dev dependencies).
#   3. Install the pre-commit git hook.
#   4. Copy .env.example to .env if .env does not already exist.
#   5. Print a success message.
#
# Usage:
#   ./scripts/bootstrap.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

echo "==> Using Python interpreter: ${PYTHON_BIN}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "==> Creating virtual environment in ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "==> Virtual environment already exists at ${VENV_DIR}, reusing it"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing ForecastForge (editable) with dev dependencies"
pip install -e ".[dev,docs]"

echo "==> Installing pre-commit git hook"
pre-commit install

if [ ! -f ".env" ]; then
  echo "==> Copying .env.example to .env"
  cp .env.example .env
else
  echo "==> .env already exists, leaving it unchanged"
fi

echo ""
echo "✅ ForecastForge development environment is ready."
echo "   Activate it in new shells with: source ${VENV_DIR}/bin/activate"
