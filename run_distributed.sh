#!/bin/bash

# Run script for Distributed ML Inference Pipeline
# This script will be executed on each node

# Validate environment variables
if [ -z "$NODE_NUMBER" ]; then
    echo "Error: NODE_NUMBER environment variable not set"
    exit 1
fi

if [ -z "$TOTAL_NODES" ]; then
    echo "Error: TOTAL_NODES environment variable not set"
    exit 1
fi

echo "Starting distributed pipeline on Node $NODE_NUMBER of $TOTAL_NODES nodes..."

# Route to appropriate service based on node number
case $NODE_NUMBER in
    0)
        echo "Starting Node 0 service (Orchestrator + Embedder)..."
        python3 node0_distributed.py
        ;;
    1)
        echo "Starting Node 1 service (FAISS + Retrieval + Reranker)..."
        python3 node1_distributed.py
        ;;
    2)
        echo "Starting Node 2 service (LLM + Sentiment + Safety)..."
        python3 node2_distributed.py
        ;;
    *)
        echo "Error: Invalid NODE_NUMBER: $NODE_NUMBER. Must be 0, 1, or 2."
        exit 1
        ;;
esac

