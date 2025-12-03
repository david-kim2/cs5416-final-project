#!/bin/bash

# Run script for ML Inference Pipeline
# This script will be executed on each node

TOTAL_NODES=${TOTAL_NODES:-1}
NODE_NUMBER=${NODE_NUMBER:-0}

if [ "$TOTAL_NODES" -eq 1 ]; then
    # Monolithic mode
    echo "Starting monolithic pipeline on Node $NODE_NUMBER..."
    python3 pipeline.py
else
    # Distributed mode
    echo "Starting distributed pipeline on Node $NODE_NUMBER..."
    if [ "$NODE_NUMBER" -eq 0 ]; then
        python3 node0_distributed.py
    elif [ "$NODE_NUMBER" -eq 1 ]; then
        python3 node1_distributed.py
    elif [ "$NODE_NUMBER" -eq 2 ]; then
        python3 node2_distributed.py
    else
        echo "Invalid NODE_NUMBER. Must be 0, 1, or 2."
        exit 1
    fi
fi