#!/bin/bash
export DYLD_LIBRARY_PATH="$(brew --prefix expat)/lib:$DYLD_LIBRARY_PATH"
cd "$(dirname "$0")"
if [ -z "${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}" ]; then
  echo "Set GOOGLE_API_KEY or GEMINI_API_KEY before running the voice agent."
  exit 1
fi
exec venv/bin/python3.12 agent.py
