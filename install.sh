#!/bin/bash

echo "Running install.sh..."

# Create venv if needed
if [ ! -d ".venv" ]; then
    virtualenv -p python3 .venv
fi

source .venv/bin/activate

# Install numpy with correct version
pip install "numpy>=1.17.3,<1.25.0"

# Install other requirements
pip install -r requirements.txt

echo "Installation complete!"