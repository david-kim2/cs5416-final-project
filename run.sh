#!/bin/bash

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python3 pipeline.py