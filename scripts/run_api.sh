#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

UVICORN_BIN="${UVICORN_BIN:-uvicorn}"
if [ -x ".venv/bin/uvicorn" ]; then
  UVICORN_BIN=".venv/bin/uvicorn"
fi

exec "$UVICORN_BIN" mini_ai_platform.main:app --host 127.0.0.1 --port 8000 --reload
