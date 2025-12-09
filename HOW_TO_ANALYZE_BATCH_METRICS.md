# How to Analyze Batch Size vs Throughput vs Memory Usage

This guide provides step-by-step instructions to analyze the relationship between batch size, throughput, and memory usage in the distributed ML inference pipeline.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Start All Nodes](#step-1-start-all-nodes)
3. [Step 2: Run Experiments](#step-2-run-experiments)
4. [Step 3: Analyze Results](#step-3-analyze-results)
5. [Understanding the Results](#understanding-the-results)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- All 3 nodes must be running
- Test data must be created (`faiss_index.bin` and `documents/` directory)
- Python 3 with required packages installed

---

## Step 1: Start All Nodes

You need to start 3 nodes in **separate terminal windows**. The order matters: start Node 1 and Node 2 first, then Node 0.

### Terminal 1: Start Node 1 (Retrieval Service)

Open a new terminal and run:

```bash
cd /home/gw356/cs5416-final-project

TOTAL_NODES=3 \
NODE_NUMBER=1 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8004 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
./run.sh
```

**Wait for:** You should see `[Node1] Ready!` in the terminal.

**What Node 1 does:**
- Handles FAISS vector search
- Retrieves documents from database
- Performs document reranking

---

### Terminal 2: Start Node 2 (Generation Service)

Open a **second** terminal and run:

```bash
cd /home/gw356/cs5416-final-project

TOTAL_NODES=3 \
NODE_NUMBER=2 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8004 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
./run.sh
```

**Wait for:** You should see `[Node2] Ready!` in the terminal.

**What Node 2 does:**
- Generates LLM responses
- Performs sentiment analysis
- Performs safety checks

---

### Terminal 3: Start Node 0 (Frontend/Orchestrator)

Open a **third** terminal and run:

```bash
cd /home/gw356/cs5416-final-project

TOTAL_NODES=3 \
NODE_NUMBER=0 \
NODE_0_IP=localhost:8000 \
NODE_1_IP=localhost:8004 \
NODE_2_IP=localhost:8002 \
FAISS_INDEX_PATH=faiss_index.bin \
DOCUMENTS_DIR=documents/ \
./run.sh
```

**Wait for:** You should see `[Node0] Ready!` and opportunistic batching configuration printed.

**What Node 0 does:**
- Receives client requests
- Generates embeddings
- Orchestrates the pipeline
- **Records batch sizes, throughput, and memory usage**

**Important:** Keep this terminal visible - you'll see batch processing statistics here!

---

### Verify All Nodes Are Running

In a **fourth terminal**, verify all nodes are healthy:

```bash
curl http://localhost:8000/health && echo " - Node 0"
curl http://localhost:8004/health && echo " - Node 1"
curl http://localhost:8002/health && echo " - Node 2"
```

You should see:
```
{"node":0,"status":"healthy"} - Node 0
{"node":1,"status":"healthy"} - Node 1
{"node":2,"status":"healthy"} - Node 2
```

---

## Step 2: Run Experiments

Now that all nodes are running, you can run experiments to collect batch size, throughput, and memory data.

### Option A: Automated Collection (Recommended)

This will run all experiments automatically and collect data:

```bash
cd /home/gw356/cs5416-final-project
python3 collect_batch_metrics.py
```

**What this does:**
- Runs experiments at 4 different rates: slow, medium, fast, very_fast
- Each experiment sends 30 requests (configurable via `NUM_REQUESTS` environment variable)
- Automatically collects batch size, throughput, and memory data
- Displays a summary table at the end
- Saves results to `batch_metrics_results.json`

**Expected output:**
```
======================================================================
BATCH SIZE vs THROUGHPUT vs MEMORY ANALYSIS
======================================================================
This will run experiments at different rates to see how batch size
affects throughput and memory usage.

Rates to test: slow, medium, fast, very_fast
Requests per experiment: 30

Make sure all 3 nodes are running!
======================================================================
Press Enter to start experiments...
```

**Time required:** Approximately 5-10 minutes for all experiments.

---

### Option B: Manual Experiments

If you prefer to run experiments manually one at a time:

#### Experiment 1: Slow Rate (Small Batches)

In a **new terminal** (Terminal 4 or 5):

```bash
cd /home/gw356/cs5416-final-project

# Clear previous batch log (optional)
rm -f batch_sizes.jsonl

# Run slow rate experiment
NODE_0_IP=localhost:8000 python3 test_profiling.py rate 30 slow
```

**What to expect:**
- Requests sent every 10 seconds
- Small batch sizes (1-2 requests per batch)
- Lower throughput
- Lower memory usage

**Watch Node 0 terminal** for batch processing statistics.

After completion, save the log:
```bash
mv batch_sizes.jsonl batch_sizes_slow.jsonl
```

---

#### Experiment 2: Medium Rate (Medium Batches)

```bash
NODE_0_IP=localhost:8000 python3 test_profiling.py rate 30 medium
```

**What to expect:**
- Requests sent every 2 seconds
- Medium batch sizes (2-4 requests per batch)
- Medium throughput
- Medium memory usage

Save the log:
```bash
mv batch_sizes.jsonl batch_sizes_medium.jsonl
```

---

#### Experiment 3: Fast Rate (Large Batches)

```bash
NODE_0_IP=localhost:8000 python3 test_profiling.py rate 30 fast
```

**What to expect:**
- Requests sent every 0.5 seconds
- Large batch sizes (4-8 requests per batch)
- Higher throughput
- Higher memory usage

Save the log:
```bash
mv batch_sizes.jsonl batch_sizes_fast.jsonl
```

---

#### Experiment 4: Very Fast Rate (Maximum Batches)

```bash
NODE_0_IP=localhost:8000 python3 test_profiling.py rate 30 very_fast
```

**What to expect:**
- Requests sent every 0.1 seconds
- Maximum batch sizes (8-16 requests per batch, up to MAX_BATCH_SIZE)
- Highest throughput
- Highest memory usage

Save the log:
```bash
mv batch_sizes.jsonl batch_sizes_very_fast.jsonl
```

---

## Step 3: Analyze Results

### Method 1: Analyze All Experiments Together

If you used the automated script (`collect_batch_metrics.py`), results are already displayed. You can also view the saved JSON:

```bash
cat batch_metrics_results.json
```

---

### Method 2: Analyze Individual Log Files

For each experiment log file:

```bash
# Analyze slow rate
python3 show_batch_relationship.py batch_sizes_slow.jsonl

# Analyze medium rate
python3 show_batch_relationship.py batch_sizes_medium.jsonl

# Analyze fast rate
python3 show_batch_relationship.py batch_sizes_fast.jsonl

# Analyze very fast rate
python3 show_batch_relationship.py batch_sizes_very_fast.jsonl
```

---

### Method 3: Analyze Current Batch Log

If you just ran an experiment and haven't renamed the log:

```bash
python3 show_batch_relationship.py batch_sizes.jsonl
```

---

### Method 4: Detailed Batch Size Analysis

For detailed batch size statistics:

```bash
python3 analyze_batch_sizes.py batch_sizes.jsonl
```

This shows:
- Batch size distribution histogram
- Queue depth statistics
- Correlation between optimal and actual batch sizes

---

## Understanding the Results

### Output Table Explanation

When you run the analysis, you'll see a table like this:

```
BATCH SIZE vs THROUGHPUT vs MEMORY USAGE
==================================================================
Rate         Avg Batch    Min Batch    Max Batch    Throughput        Avg Memory      Peak Memory
             Size         Size         Size         (req/min)         (MB)            (MB)
------------------------------------------------------------------
slow         1.2          1            2            12.5              450.2           480.3
medium       2.8          2            4            28.3              460.1           490.2
fast         5.5          4            8            52.7              480.5           510.8
very_fast    12.3         8            16           95.1              520.3           560.1
==================================================================
```

**Column meanings:**
- **Rate**: Request rate used in the experiment
- **Avg Batch Size**: Average number of requests processed per batch
- **Min/Max Batch Size**: Smallest and largest batch sizes observed
- **Throughput (req/min)**: Requests processed per minute
- **Avg Memory (MB)**: Average memory usage during processing
- **Peak Memory (MB)**: Maximum memory usage observed

---

### Key Relationships to Observe

#### 1. Batch Size vs Throughput

**Expected pattern:** Larger batch sizes → Higher throughput

- **Why?** Processing multiple requests together is more efficient than processing them one-by-one
- **Example:** Batch size 1 might give 12 req/min, batch size 8 might give 80 req/min

#### 2. Batch Size vs Memory

**Expected pattern:** Larger batch sizes → Higher memory usage

- **Why?** Larger batches require more memory to hold embeddings, documents, and generated responses simultaneously
- **Example:** Batch size 1 might use 450 MB, batch size 8 might use 520 MB

#### 3. The Trade-off

**Key insight:** There's a trade-off between throughput and memory usage.

- **Small batches:** Lower memory, lower throughput
- **Large batches:** Higher memory, higher throughput

The **efficiency ratio** shows how much throughput gain you get per unit of memory increase:
- **Efficiency > 1**: Throughput increases faster than memory (good!)
- **Efficiency < 1**: Memory increases faster than throughput (trade-off)

---

### What the Numbers Mean

#### Throughput

- **Low throughput (10-20 req/min)**: System is processing requests slowly, likely due to small batches
- **Medium throughput (20-50 req/min)**: Moderate batching, balanced performance
- **High throughput (50+ req/min)**: Large batches, high efficiency

#### Memory Usage

- **Low memory (400-450 MB)**: Small batches, minimal memory overhead
- **Medium memory (450-500 MB)**: Moderate batches, reasonable memory usage
- **High memory (500+ MB)**: Large batches, higher memory footprint

#### Batch Size Distribution

The distribution shows how often each batch size was used:
- **Mostly size 1-2**: Low load, requests arrive slowly
- **Mix of sizes 2-8**: Medium load, adaptive batching working
- **Mostly size 8-16**: High load, maximum batching

---

## Step 4: Record Your Findings

After analyzing, create a summary table for your report:

| Rate | Avg Batch Size | Throughput (req/min) | Avg Memory (MB) | Peak Memory (MB) |
|------|----------------|---------------------|-----------------|------------------|
| Slow | ? | ? | ? | ? |
| Medium | ? | ? | ? | ? |
| Fast | ? | ? | ? | ? |
| Very Fast | ? | ? | ? | ? |

**Key observations to note:**
- How does batch size change with request rate?
- How does throughput scale with batch size?
- How does memory usage scale with batch size?
- What is the efficiency ratio (throughput gain / memory cost)?

---

## Troubleshooting

### Problem: "No batch data found"

**Solution:**
- Make sure Node 0 is running and processing requests
- Check that `batch_sizes.jsonl` file exists and has content
- Verify Node 0 terminal shows batch processing messages

### Problem: "No throughput/memory data found"

**Solution:**
- Make sure you're using the updated `pipeline.py` that logs memory and throughput
- Restart Node 0 to ensure it's using the latest code
- Check that batches are completing (look for "Batch complete" messages in Node 0)

### Problem: Nodes not responding

**Solution:**
- Check that all 3 nodes are running: `curl http://localhost:8000/health`
- Verify ports are not in use: `lsof -i :8000 -i :8004 -i :8002`
- Restart nodes if needed


## Quick Reference Commands

### Start Nodes
```bash
# Node 1
TOTAL_NODES=3 NODE_NUMBER=1 NODE_0_IP=localhost:8000 NODE_1_IP=localhost:8004 NODE_2_IP=localhost:8002 FAISS_INDEX_PATH=faiss_index.bin DOCUMENTS_DIR=documents/ ./run.sh

# Node 2
TOTAL_NODES=3 NODE_NUMBER=2 NODE_0_IP=localhost:8000 NODE_1_IP=localhost:8004 NODE_2_IP=localhost:8002 FAISS_INDEX_PATH=faiss_index.bin DOCUMENTS_DIR=documents/ ./run.sh

# Node 0
TOTAL_NODES=3 NODE_NUMBER=0 NODE_0_IP=localhost:8000 NODE_1_IP=localhost:8004 NODE_2_IP=localhost:8002 FAISS_INDEX_PATH=faiss_index.bin DOCUMENTS_DIR=documents/ ./run.sh
```

### Run Experiments
```bash
# Automated (all rates)
python3 collect_batch_metrics.py

# Manual (single rate)
NODE_0_IP=localhost:8000 python3 test_profiling.py rate 30 slow
```

### Analyze Results
```bash
# Show relationship table
python3 show_batch_relationship.py batch_sizes.jsonl

# Detailed batch statistics
python3 analyze_batch_sizes.py batch_sizes.jsonl
```

### Check Node Health
```bash
curl http://localhost:8000/health && echo " - Node 0"
curl http://localhost:8004/health && echo " - Node 1"
curl http://localhost:8002/health && echo " - Node 2"
```

---


