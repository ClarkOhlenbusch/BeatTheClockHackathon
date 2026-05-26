#!/bin/bash
export DYLD_LIBRARY_PATH="$(brew --prefix expat)/lib:$DYLD_LIBRARY_PATH"
cd "$(dirname "$0")"
exec venv/bin/python3.12 agent.py
