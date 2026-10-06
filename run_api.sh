#!/usr/bin/env bash
# Start the HR chatbot API locally. Apply the schema first (idempotent):
#   python3 -m vectorstore.setup_db
cd "$(dirname "$0")"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
