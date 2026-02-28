#!/usr/bin/env zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/dashboard/frontend"

if [[ -f "$ROOT_DIR/../../venv/bin/activate" ]]; then
  source "$ROOT_DIR/../../venv/bin/activate"
elif [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  source "$ROOT_DIR/venv/bin/activate"
else
  echo "No virtualenv activate script found (checked ../../venv and ./venv)." >&2
  exit 1
fi

cd "$FRONTEND_DIR"
npm run build

cd "$ROOT_DIR"
exec uvicorn dashboard.backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"
