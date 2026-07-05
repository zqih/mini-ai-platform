#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec uvicorn mini_ai_platform.main:app --host 127.0.0.1 --port 8000 --reload

