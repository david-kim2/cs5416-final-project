# Distributed Pipeline Implementation Summary

## Overview

This document describes the distributed implementation of the ML inference pipeline, designed to meet Tier 3 (Excellent Grade) requirements. The implementation is based on bottleneck analysis that identified LLM Generation (55.7%) and FAISS Search (31.5%) as the primary bottlenecks.

## Architecture

### Node 0: Orchestrator + Embeddings + Sentiment + Safety
**Services:**
- Request orchestration and routing
- Embedding generation (BAAI/bge-base-en-v1.5)
- Sentiment analysis (nlptown/bert-base-multilingual-uncased-sentiment)
- Safety filtering (unitary/toxic-bert)

**Port:** 8000 (configurable via NODE_0_IP)

**Features:**
- Opportunistic batching with adaptive batch sizes based on queue depth
- Embedding caching (1000 entry cache with FIFO eviction)
- Parallel sentiment and safety analysis using threading
- Data compression for inter-node communication
- GPU acceleration support

### Node 1: FAISS + Document Retrieval + Reranking
**Services:**
- FAISS ANN search
- Document fetching from SQLite database
- Document reranking (BAAI/bge-reranker-base)

**Port:** 8001 (configurable via NODE_1_IP)

**Features:**
- FAISS index loaded once at startup (stays in memory)
- Document caching (2000 entry cache with FIFO eviction)
- Reranker model loaded once at startup
- GPU acceleration support for reranker
- Data compression for inter-node communication

### Node 2: LLM Generation
**Services:**
- LLM response generation (Qwen/Qwen2.5-0.5B-Instruct)

**Port:** 8002 (configurable via NODE_2_IP)

**Features:**
- LLM model loaded once at startup (stays in memory)
- GPU acceleration support
- Data compression for inter-node communication

## Key Features Implemented

### Tier 1 Requirements ✓
- Pipeline runs across 3 nodes
- Node 0 receives client requests and returns responses
- Correctness maintained
- Runs within 16GB RAM per node constraint

### Tier 2 Requirements ✓
- True microservices architecture with distinct services per node
- Orchestration and request routing across nodes
- Services communicate via HTTP REST APIs

### Tier 3 Requirements ✓
- **Opportunistic Batching**: Dynamic batch sizing based on queue depth
  - Low load (queue ≤ 2): MIN_BATCH_SIZE (default 1)
  - Medium load (queue 3-5): BASE_BATCH_SIZE (default 4)
  - High load (queue 6-10): BASE_BATCH_SIZE * 2 (default 8)
  - Very high load (queue > 10): MAX_BATCH_SIZE (default 8)
  - Adaptive timeouts: Longer timeouts when queue is deep
- **GPU Acceleration**: Configurable via USE_GPU environment variable (default: true)
- **Caching Strategies**: 
  - Embedding cache on Node 0 (1000 entries)
  - Document cache on Node 1 (2000 entries)
- **Parallel Execution**: Sentiment and safety analysis run in parallel on Node 0
- **Data Compression**: Gzip compression for inter-node communication (configurable via ENABLE_COMPRESSION)

## Environment Variables

### Required (provided by TAs)
- `TOTAL_NODES`: Number of nodes (3 for distributed)
- `NODE_NUMBER`: Node identifier (0, 1, or 2)
- `NODE_0_IP`, `NODE_1_IP`, `NODE_2_IP`: IP addresses for each node
- `FAISS_INDEX_PATH`: Path to FAISS index file
- `DOCUMENTS_DIR`: Directory containing documents database

### Optional (optimization controls)
- `USE_GPU`: Enable GPU acceleration (default: "true")
- `ENABLE_CACHING`: Enable caching (default: "true")
- `ENABLE_COMPRESSION`: Enable data compression (default: "true")
- `MIN_BATCH_SIZE`: Minimum batch size for opportunistic batching (default: 1)
- `MAX_BATCH_SIZE`: Maximum batch size (default: 8)
- `BASE_BATCH_SIZE`: Base batch size for medium load (default: 4)
- `BATCH_TIMEOUT`: Base timeout for batching in seconds (default: 0.5)
- `MAX_BATCH_TIMEOUT`: Maximum timeout for batching (default: 2.0)

## Running the Distributed Pipeline

### Step 1: Start Node 0 (Orchestrator)
```bash
TOTAL_NODES=3 NODE_NUMBER=0 \
NODE_0_IP=localhost:8000 NODE_1_IP=localhost:8001 NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin DOCUMENTS_DIR=documents/ \
./run.sh
```

### Step 2: Start Node 1 (FAISS + Reranking)
```bash
TOTAL_NODES=3 NODE_NUMBER=1 \
NODE_0_IP=localhost:8000 NODE_1_IP=localhost:8001 NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin DOCUMENTS_DIR=documents/ \
./run.sh
```

### Step 3: Start Node 2 (LLM)
```bash
TOTAL_NODES=3 NODE_NUMBER=2 \
NODE_0_IP=localhost:8000 NODE_1_IP=localhost:8001 NODE_2_IP=localhost:8002 \
./run.sh
```

### Step 4: Verify Health
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### Step 5: Run Client
```bash
NODE_0_IP=localhost:8000 python3 client.py
```

## Request Flow

1. Client sends request to Node 0 (`/query` endpoint)
2. Node 0 adds request to batch queue
3. Node 0 worker thread collects batch (opportunistic batching)
4. Node 0 generates embeddings (with caching)
5. Node 0 sends embeddings to Node 1 (`/search_and_rerank`)
6. Node 1 performs FAISS search
7. Node 1 fetches documents (with caching)
8. Node 1 reranks documents
9. Node 1 returns reranked documents to Node 0
10. Node 0 sends queries and documents to Node 2 (`/generate`)
11. Node 2 generates LLM responses
12. Node 2 returns responses to Node 0
13. Node 0 performs sentiment and safety analysis in parallel
14. Node 0 returns final response to client

## Profiling Experiments

See `PROFILING_EXPERIMENTS.md` for detailed instructions on:
- Measuring throughput vs batch size
- Measuring latency vs batch size
- Testing optimization impacts
- Memory profiling
- Sustained load testing

## Design Decisions

1. **Service Division**: Based on bottleneck analysis, LLM (55.7% of time) is isolated on Node 2, FAISS (31.5%) is on Node 1 with reranking, and lightweight stages are on Node 0.

2. **Model Loading**: Models are loaded once at startup and kept in memory to avoid repeated loading overhead (unlike the monolithic version).

3. **Opportunistic Batching**: Batch sizes adapt to queue depth, balancing latency (small batches) and throughput (large batches).

4. **Caching**: Embeddings and documents are cached to reduce redundant computation and I/O.

5. **Compression**: Inter-node communication uses gzip compression to reduce network bandwidth.

6. **Parallel Execution**: Sentiment and safety analysis run in parallel since they are independent.

## Performance Optimizations

- **GPU Acceleration**: All models can use GPU when available (configurable)
- **Caching**: Reduces redundant embedding generation and document fetching
- **Compression**: Reduces network transfer time between nodes
- **Model Persistence**: Models stay loaded in memory, avoiding reload overhead
- **Opportunistic Batching**: Maximizes throughput under load while maintaining low latency under light load

## Files

- `node0_distributed.py`: Node 0 service implementation
- `node1_distributed.py`: Node 1 service implementation
- `node2_distributed.py`: Node 2 service implementation
- `run.sh`: Entry point script (supports both monolithic and distributed modes)
- `PROFILING_EXPERIMENTS.md`: Instructions for running profiling experiments
- `bottleneck_analysis.tex`: LaTeX report section with bottleneck analysis

