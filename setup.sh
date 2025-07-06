#!/usr/bin/env bash
# setup.sh - Bootstrap a Python project using Poetry
# Usage: ./setup.sh
# Requires: bash, poetry

set -euo pipefail

# Constants
declare -r SCRIPT_NAME="${0##*/}"    # type: string - script name
declare -r POETRY_BIN="$(command -v poetry || true)"  # type: string - path to Poetry binary

# Function to print error and exit
error() {
  # $1: error message (string)
  echo "${SCRIPT_NAME}: ERROR: $1" >&2
  exit 1
}

# Verify Poetry is installed
if [[ -z "${POETRY_BIN}" ]]; then
  error "Poetry is not installed. Install it via 'pip install --user poetry'."
fi

# Ensure script runs from project root by checking for pyproject.toml
if [[ ! -f "pyproject.toml" ]]; then
  error "pyproject.toml not found. Run this script from the project root."
fi

# Install dependencies and create (or update) virtual environment
echo "Installing dependencies and setting up virtual environment..."
"${POETRY_BIN}" install

# Optional: install pre-commit hooks if configured
if [[ -f ".pre-commit-config.yaml" ]]; then
  echo "Found .pre-commit-config.yaml, installing hooks..."
  "${POETRY_BIN}" run pre-commit install
fi

echo "Python project setup complete."

exit 0
