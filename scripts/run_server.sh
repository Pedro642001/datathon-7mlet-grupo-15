#!/usr/bin/env bash
# Script para rodar o servidor FastAPI com uvicorn (novo layout src.app.main:app)
set -euo pipefail
venv/bin/uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
