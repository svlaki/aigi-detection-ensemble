#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="$PWD" python3 -m uvicorn demo.backend.main:app --reload --port 8000
