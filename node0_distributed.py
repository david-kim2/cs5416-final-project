#!/usr/bin/env python3
"""
Node 0 Service: Orchestrator + Embedder
- Receives client requests at /query endpoint
- Generates embeddings using SentenceTransformer
- Routes to Node 1 for FAISS/retrieval/reranking
- Routes to Node 2 for LLM/sentiment/safety
- Returns final response to client
"""

import os
import time
import requests
import numpy as np
import torch
import gc
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from queue import Queue
import threading
from sentence_transformers import SentenceTransformer

# Read environment variables
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8001')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8002')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
BATCH_TIMEOUT = float(os.environ.get('BATCH_TIMEOUT', '0.5'))

# Flask app
app = Flask(__name__)

# Request queue and results storage
results = {}
results_lock = threading.Lock()
batch_queue = []
batch_lock = threading.Lock()

class EmbedderService:
    """Service for generating embeddings"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        print(f"[Node 0] Initializing embedder on {self.device}")
        self.model = SentenceTransformer(self.embedding_model_name).to(self.device)
        print("[Node 0] Embedder initialized!")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of queries"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype('float32').tolist()  # Convert to list for JSON serialization


# Global services
embedder = None

def send_to_node1(embeddings: List[List[float]], queries: List[str], request_ids: List[str]) -> Dict[str, Any]:
    """Send embeddings to Node 1 for FAISS search and reranking"""
    try:
        url = f"http://{NODE_1_IP}/process"
        payload = {
            'embeddings': embeddings,
            'queries': queries,
            'request_ids': request_ids
        }
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Node 1 returned status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[Node 0] Error communicating with Node 1: {e}")
        raise

def send_to_node2(queries: List[str], documents_batch: List[List[Dict]], request_ids: List[str]) -> Dict[str, Any]:
    """Send queries and documents to Node 2 for LLM generation, sentiment, and safety"""
    try:
        url = f"http://{NODE_2_IP}/process"
        payload = {
            'queries': queries,
            'documents': documents_batch,
            'request_ids': request_ids
        }
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Node 2 returned status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[Node 0] Error communicating with Node 2: {e}")
        raise

def process_batch_worker():
    """Worker thread that processes batches from the queue"""
    global embedder
    while True:
        try:
            # Collect batch
            batch = []
            start_time = time.time()
            
            # Wait for batch to fill or timeout
            while True:
                with batch_lock:
                    queue_len = len(batch_queue)
                    elapsed = time.time() - start_time
                    
                    if queue_len >= BATCH_SIZE:
                        # Batch is full, take BATCH_SIZE items
                        batch = batch_queue[:BATCH_SIZE]
                        batch_queue[:] = batch_queue[BATCH_SIZE:]
                        break
                    elif queue_len > 0 and elapsed >= BATCH_TIMEOUT:
                        # Timeout reached, take what we have
                        batch = batch_queue[:]
                        batch_queue[:] = []
                        break
                
                # Sleep outside lock to avoid blocking
                if len(batch) == 0:
                    time.sleep(0.01)
                else:
                    break
            
            if not batch:
                time.sleep(0.1)
                continue
            
            print(f"\n[Node 0] Processing batch of {len(batch)} requests")
            
            # Extract data
            request_ids = [item['request_id'] for item in batch]
            queries = [item['query'] for item in batch]
            
            # Step 1: Generate embeddings
            print(f"[Node 0] Generating embeddings for {len(queries)} queries...")
            embeddings = embedder.generate_embeddings_batch(queries)
            
            # Step 2: Send to Node 1 for FAISS search and reranking
            print(f"[Node 0] Sending to Node 1 for FAISS search and reranking...")
            node1_result = send_to_node1(embeddings, queries, request_ids)
            
            # Step 3: Send to Node 2 for LLM generation, sentiment, and safety
            print(f"[Node 0] Sending to Node 2 for LLM generation, sentiment, and safety...")
            node2_result = send_to_node2(
                queries,
                node1_result['reranked_documents'],
                request_ids
            )
            
            # Step 4: Store results
            with results_lock:
                for idx, req_id in enumerate(request_ids):
                    results[req_id] = {
                        'request_id': req_id,
                        'generated_response': node2_result['responses'][idx],
                        'sentiment': node2_result['sentiments'][idx],
                        'is_toxic': node2_result['is_toxic'][idx]
                    }
            
            print(f"[Node 0] Batch processing complete for {len(batch)} requests")
            
        except Exception as e:
            print(f"[Node 0] Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            # Mark requests as failed
            with results_lock:
                for item in batch:
                    results[item['request_id']] = {
                        'request_id': item['request_id'],
                        'error': str(e)
                    }

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests from client"""
    try:
        data = request.json
        request_id = data.get('request_id')
        query = data.get('query')
        
        if not request_id or not query:
            return jsonify({'error': 'Missing request_id or query'}), 400
        
        # Check if result already exists
        with results_lock:
            if request_id in results:
                result = results.pop(request_id)
                if 'error' in result:
                    return jsonify(result), 500
                return jsonify(result), 200
        
        print(f"[Node 0] Queueing request {request_id}")
        
        # Add to batch queue
        with batch_lock:
            batch_queue.append({
                'request_id': request_id,
                'query': query,
                'timestamp': time.time()
            })
        
        # Wait for processing (with timeout)
        timeout = 300  # 5 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    if 'error' in result:
                        return jsonify(result), 500
                    return jsonify(result), 200
            
            if time.time() - start_wait > timeout:
                return jsonify({'error': 'Request timeout'}), 504
            
            time.sleep(0.1)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': 0,
        'service': 'orchestrator+embedder'
    }), 200

def main():
    """Main execution function"""
    global embedder
    
    print("="*60)
    print("NODE 0 SERVICE: ORCHESTRATOR + EMBEDDER")
    print("="*60)
    print(f"Node 0 IP: {NODE_0_IP}")
    print(f"Node 1 IP: {NODE_1_IP}")
    print(f"Node 2 IP: {NODE_2_IP}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batch timeout: {BATCH_TIMEOUT}s")
    print("="*60)
    
    # Initialize embedder
    print("\nInitializing embedder service...")
    embedder = EmbedderService()
    
    # Start batch processing worker thread
    worker_thread = threading.Thread(target=process_batch_worker, daemon=True)
    worker_thread.start()
    print("Batch processing worker thread started!")
    
    # Start Flask server
    print(f"\nStarting Flask server on {NODE_0_IP}")
    hostname = NODE_0_IP.split(':')[0]
    port = int(NODE_0_IP.split(':')[1]) if ':' in NODE_0_IP else 8000
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()

