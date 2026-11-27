#!/usr/bin/env bash
# Start the rloup Web Arena. Creates the venv and installs deps on first run.
# Usage: ./run_server.sh [--host 127.0.0.1] [--port 8765] [--device auto]
set -euo pipefail

cd "$(dirname "$0")"

VENV=.venv
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    echo "First run: creating $VENV and installing dependencies..." >&2
    command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
    uv venv --python 3.12 "$VENV"
    uv pip install -e .
fi

if ! "$PY" -c 'import rloup' >/dev/null 2>&1; then
    echo "Dependencies missing in $VENV, reinstalling..." >&2
    uv pip install -e .
fi

exec "$PY" -m rloup.web "$@"
