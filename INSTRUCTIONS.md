# Distributed ML Pipeline - Instructions

## Overview

This implementation transforms the monolithic pipeline into a distributed microservices architecture across 3 nodes:

- **Node 0**: Orchestrator + Embedder (receives client requests, generates embeddings)
- **Node 1**: FAISS + Document Retrieval + Reranker
- **Node 2**: LLM + Sentiment Analysis + Safety Filter

## Architecture

```
Client → Node 0 (Embedder) → Node 1 (FAISS+Rerank) → Node 2 (LLM+Sentiment+Safety) → Node 0 → Client
```

## Files Created

1. **node0_distributed.py** - Orchestrator and embedding service
2. **node1_distributed.py** - FAISS search, document retrieval, and reranking service
3. **node2_distributed.py** - LLM generation, sentiment analysis, and safety filtering service
4. **run_distributed.sh** - Script to run the distributed version
5. **compare_pipelines.py** - Script to compare monolithic vs distributed performance

## How to Run

### Prerequisites

1. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. Create test documents and FAISS index:
   ```bash
   python3 create_test_docs.py
   ```

3. Get your IP addresses for each node:
   ```bash
   ifconfig  # or ip addr on Linux
   ```

### Running the Distributed Pipeline

You need 3 separate machines or terminals (one for each node).

#### Terminal 1 - Node 0 (Orchestrator + Embedder):
```bash
TOTAL_NODES=3 NODE_NUMBER=0 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8001 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
./run_distributed.sh
```

#### Terminal 2 - Node 1 (FAISS + Retrieval + Reranker):
```bash
TOTAL_NODES=3 NODE_NUMBER=1 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8001 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
./run_distributed.sh
```

#### Terminal 3 - Node 2 (LLM + Sentiment + Safety):
```bash
TOTAL_NODES=3 NODE_NUMBER=2 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8001 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
./run_distributed.sh
```

**Note**: If running on different machines, replace `localhost` with the actual IP addresses.

### Running the Client

Once all 3 nodes are running, in a 4th terminal:

```bash
NODE_0_IP=localhost:8000 python3 client.py
```

## Configuration Options

You can configure batching behavior via environment variables:

- `BATCH_SIZE`: Number of requests to batch together (default: 4)
- `BATCH_TIMEOUT`: Maximum time to wait for batch to fill in seconds (default: 0.5)

Example with custom batching:
```bash
BATCH_SIZE=8 BATCH_TIMEOUT=1.0 TOTAL_NODES=3 NODE_NUMBER=0 ... ./run_distributed.sh
```

## How to Compare: Monolithic vs Distributed


A comparison script is provided to automate this process:

```bash
python3 compare_pipelines.py
```

This script will:
- Run both versions sequentially
- Measure latency, throughput, and memory usage
- Generate a comparison report

## Key Differences

### Monolithic Pipeline (`pipeline.py`)
- All stages run in one process
- Models loaded/unloaded per request (inefficient)
- Batch size of 1
- Single node

### Distributed Pipeline (`node*_distributed.py`)
- Stages split across 3 nodes
- Models loaded once per node (more efficient)
- Configurable batching (default: 4 requests)
- True microservices architecture
- Better memory utilization (each node only loads needed models)

## Troubleshooting

1. **Port already in use**: Make sure ports 8000, 8001, 8002 are free
   ```bash
   lsof -i :8000  # Check what's using the port
   kill <PID>     # Kill the process if needed
   ```

2. **Connection refused**: Make sure all 3 nodes are running before starting the client

3. **Model download issues**: First run will download models, which may take time

4. **Memory issues**: Reduce BATCH_SIZE if hitting memory limits

## Performance Profiling

To profile the system:

1. Monitor memory usage:
   ```bash
   watch -n 1 'ps aux | grep python3 | grep -E "node[0-2]_distributed"'
   ```

2. Monitor network traffic between nodes:
   ```bash
   # On Linux
   sudo tcpdump -i any -n port 8000 or port 8001 or port 8002
   ```

3. Measure latency:
   - The client script already measures and reports latency
   - Check the output for "took X.XXs" messages

## Expected Output

When running the client, you should see:
- Requests being sent every 10 seconds
- Responses with generated text, sentiment, and toxicity flags
- Summary with total time, successful/failed requests

The distributed version should show:
- Batch processing messages from Node 0
- Processing messages from Node 1 and Node 2
- Similar correctness to monolithic version
- Potentially better throughput due to batching

