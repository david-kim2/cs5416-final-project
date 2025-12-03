"""
Node 0 Service: Orchestrator + Embeddings + Sentiment + Safety
- Receives client requests
- Generates embeddings
- Routes to Node 1 (FAISS + Reranking) and Node 2 (LLM)
- Performs sentiment and safety analysis in parallel
- Returns final responses to client
"""

import os
import gc
import json
import time
import hashlib
import gzip
import base64
import threading
import numpy as np
import torch
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue
import requests

# Environment variables
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8001')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8002')
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'true').lower() == 'true'
ENABLE_COMPRESSION = os.environ.get('ENABLE_COMPRESSION', 'true').lower() == 'true'

# Opportunistic batching configuration
MIN_BATCH_SIZE = int(os.environ.get('MIN_BATCH_SIZE', '1'))
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '8'))
BASE_BATCH_SIZE = int(os.environ.get('BASE_BATCH_SIZE', '4'))
BATCH_TIMEOUT = float(os.environ.get('BATCH_TIMEOUT', '0.5'))
MAX_BATCH_TIMEOUT = float(os.environ.get('MAX_BATCH_TIMEOUT', '2.0'))

CONFIG = {
    'max_tokens': 128,
    'truncate_length': 512
}

app = Flask(__name__)

# Request queue and results storage
batch_queue = Queue()
results = {}
results_lock = threading.Lock()

# Embedding cache
embedding_cache = {}
embedding_cache_lock = threading.Lock()
cache_max_size = 1000

# Model loading lock for sentiment/safety
model_load_lock = threading.Lock()

@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float

def get_device():
    """Get device based on GPU availability and USE_GPU setting"""
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Device] GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        if USE_GPU:
            print(f"[Device] GPU requested but not available, using CPU")
        else:
            print(f"[Device] Using CPU (GPU disabled)")
    return device

def compress_data(data: bytes) -> str:
    """Compress data using gzip and return base64 encoded string"""
    if not ENABLE_COMPRESSION:
        return base64.b64encode(data).decode('utf-8')
    compressed = gzip.compress(data)
    return base64.b64encode(compressed).decode('utf-8')

def decompress_data(encoded_data: str) -> bytes:
    """Decompress base64 encoded gzip data"""
    data = base64.b64decode(encoded_data.encode('utf-8'))
    if not ENABLE_COMPRESSION:
        return data
    return gzip.decompress(data)

def calculate_optimal_batch_size(queue_depth: int) -> int:
    """Opportunistic batching: Calculate optimal batch size based on queue depth"""
    if queue_depth <= 2:
        return MIN_BATCH_SIZE
    elif queue_depth <= 5:
        return BASE_BATCH_SIZE
    elif queue_depth <= 10:
        return min(BASE_BATCH_SIZE * 2, MAX_BATCH_SIZE)
    else:
        return MAX_BATCH_SIZE

def calculate_adaptive_timeout(queue_depth: int) -> float:
    """Adaptive timeout: Longer timeout when queue is deep"""
    if queue_depth <= 2:
        return BATCH_TIMEOUT
    elif queue_depth <= 5:
        return BATCH_TIMEOUT * 1.5
    else:
        return min(BATCH_TIMEOUT * 2, MAX_BATCH_TIMEOUT)

class EmbedderService:
    """Service for generating embeddings with caching"""
    def __init__(self):
        self.device = get_device()
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        print("[Node 0] Loading embedding model...")
        self.model = SentenceTransformer(self.embedding_model_name).to(self.device)
        print("[Node 0] Embedding model loaded!")
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with caching"""
        if not ENABLE_CACHING:
            return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        
        # Check cache and generate only for uncached texts
        texts_to_encode = []
        text_indices = []
        cached_embeddings = {}
        new_embeddings_list = []
        
        with embedding_cache_lock:
            for idx, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in embedding_cache:
                    cached_embeddings[idx] = embedding_cache[text_hash]
                else:
                    texts_to_encode.append(text)
                    text_indices.append(idx)
        
        # Generate embeddings for uncached texts
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            new_embeddings_list = new_embeddings.tolist()
            
            # Update cache
            with embedding_cache_lock:
                for text, embedding in zip(texts_to_encode, new_embeddings_list):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    if len(embedding_cache) >= cache_max_size:
                        # Remove oldest entry (FIFO-like)
                        oldest_key = next(iter(embedding_cache))
                        del embedding_cache[oldest_key]
                    embedding_cache[text_hash] = embedding
        
        # Combine cached and new embeddings
        all_embeddings = []
        cache_idx = 0
        for idx in range(len(texts)):
            if idx in cached_embeddings:
                all_embeddings.append(cached_embeddings[idx])
            else:
                all_embeddings.append(new_embeddings_list[cache_idx])
                cache_idx += 1
        
        return np.array(all_embeddings, dtype=np.float32)

class SentimentSafetyService:
    """Service for sentiment analysis and safety filtering"""
    def __init__(self):
        self.device = get_device()
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'
        # Models loaded on-demand to save memory
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Analyze sentiment for batch"""
        with model_load_lock:
            classifier = hf_pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name
            )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        sentiments = [sentiment_map.get(result['label'], 'neutral') for result in raw_results]
        del classifier
        gc.collect()
        return sentiments
    
    def filter_response_safety_batch(self, texts: List[str]) -> List[str]:
        """Filter responses for safety"""
        with model_load_lock:
            classifier = hf_pipeline(
                "text-classification",
                model=self.safety_model_name
            )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        toxicity_flags = [result['score'] > 0.5 for result in raw_results]
        del classifier
        gc.collect()
        return toxicity_flags
    
    def analyze_sentiment_and_safety_parallel(self, texts: List[str]) -> tuple[List[str], List[str]]:
        """Run sentiment and safety analysis in parallel"""
        sentiments = []
        toxicity_flags = []
        errors = {'sentiment': None, 'safety': None}
        
        def run_sentiment():
            nonlocal sentiments, errors
            try:
                sentiments = self.analyze_sentiment_batch(texts)
            except Exception as e:
                errors['sentiment'] = str(e)
        
        def run_safety():
            nonlocal toxicity_flags, errors
            try:
                toxicity_flags = self.filter_response_safety_batch(texts)
            except Exception as e:
                errors['safety'] = str(e)
        
        thread1 = threading.Thread(target=run_sentiment)
        thread2 = threading.Thread(target=run_safety)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        if errors['sentiment']:
            raise Exception(f"Sentiment analysis failed: {errors['sentiment']}")
        if errors['safety']:
            raise Exception(f"Safety filtering failed: {errors['safety']}")
        
        return sentiments, toxicity_flags

# Global services
embedder = None
sentiment_safety = None

def send_to_node1(embeddings: np.ndarray, queries: List[str], request_ids: List[str]) -> Dict:
    """Send embeddings to Node 1 for FAISS search and reranking"""
    url = f"http://{NODE_1_IP}/search_and_rerank"
    data = {
        'embeddings': embeddings.tolist(),
        'queries': queries,
        'request_ids': request_ids
    }
    json_data = json.dumps(data).encode('utf-8')
    compressed_data = compress_data(json_data)
    
    response = requests.post(
        url,
        json={'data': compressed_data, 'compressed': ENABLE_COMPRESSION},
        timeout=300
    )
    response.raise_for_status()
    result = response.json()
    
    if result.get('compressed'):
        decompressed = decompress_data(result['data'])
        result = json.loads(decompressed.decode('utf-8'))
    else:
        result = json.loads(result['data'])
    
    return result

def send_to_node2(queries: List[str], documents_batch: List[List[Dict]], request_ids: List[str]) -> Dict:
    """Send queries and documents to Node 2 for LLM generation"""
    url = f"http://{NODE_2_IP}/generate"
    data = {
        'queries': queries,
        'documents_batch': documents_batch,
        'request_ids': request_ids
    }
    json_data = json.dumps(data).encode('utf-8')
    compressed_data = compress_data(json_data)
    
    response = requests.post(
        url,
        json={'data': compressed_data, 'compressed': ENABLE_COMPRESSION},
        timeout=600
    )
    response.raise_for_status()
    result = response.json()
    
    if result.get('compressed'):
        decompressed = decompress_data(result['data'])
        result = json.loads(decompressed.decode('utf-8'))
    else:
        result = json.loads(result['data'])
    
    return result

def process_batch_worker():
    """Worker thread that processes batches from the queue"""
    global embedder, sentiment_safety
    
    while True:
        try:
            # Collect batch with opportunistic batching
            batch = []
            queue_depth = batch_queue.qsize()
            optimal_batch_size = calculate_optimal_batch_size(queue_depth)
            adaptive_timeout = calculate_adaptive_timeout(queue_depth)
            
            # Wait for first request
            first_request = batch_queue.get(timeout=None)
            if first_request is None:
                break
            batch.append(first_request)
            
            # Collect more requests up to optimal batch size or timeout
            deadline = time.time() + adaptive_timeout
            while len(batch) < optimal_batch_size and time.time() < deadline:
                try:
                    request = batch_queue.get(timeout=max(0.01, deadline - time.time()))
                    if request is None:
                        break
                    batch.append(request)
                except:
                    break
            
            # Process batch
            request_ids = [req['request_id'] for req in batch]
            queries = [req['query'] for req in batch]
            
            print(f"\n[Node 0] Processing batch of {len(batch)} requests (queue depth: {queue_depth}, optimal batch: {optimal_batch_size})")
            
            # Step 1: Generate embeddings
            embeddings = embedder.generate_embeddings_batch(queries)
            
            # Step 2: Send to Node 1 for FAISS search and reranking
            node1_result = send_to_node1(embeddings, queries, request_ids)
            
            # Step 3: Send to Node 2 for LLM generation
            node2_result = send_to_node2(
                queries,
                node1_result['reranked_documents'],
                request_ids
            )
            
            # Step 4: Sentiment and safety analysis in parallel
            sentiments, toxicity_flags = sentiment_safety.analyze_sentiment_and_safety_parallel(
                node2_result['responses']
            )
            
            # Step 5: Store results
            with results_lock:
                for idx, req_id in enumerate(request_ids):
                    results[req_id] = {
                        'request_id': req_id,
                        'generated_response': node2_result['responses'][idx],
                        'sentiment': sentiments[idx],
                        'is_toxic': 'true' if toxicity_flags[idx] else 'false'
                    }
            
            print(f"[Node 0] Batch processing complete for {len(batch)} requests")
            
        except Exception as e:
            import traceback
            print(f"[Node 0] Error processing batch: {e}")
            print(traceback.format_exc())
            batch_queue.task_done()

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests"""
    try:
        data = request.json
        request_id = data.get('request_id')
        query = data.get('query')
        
        if not request_id or not query:
            return jsonify({'error': 'Missing request_id or query'}), 400
        
        # Check if result already exists
        with results_lock:
            if request_id in results:
                return jsonify(results[request_id]), 200
        
        # Add to queue
        batch_queue.put({
            'request_id': request_id,
            'query': query,
            'timestamp': time.time()
        })
        
        # Wait for processing
        timeout = 900  # 15 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
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
        'service': 'orchestrator+embedder+sentiment+safety',
        'node': 0
    }), 200

def main():
    """Main execution function"""
    global embedder, sentiment_safety
    
    print("="*60)
    print("NODE 0 SERVICE: ORCHESTRATOR + EMBEDDER + SENTIMENT + SAFETY")
    print("="*60)
    print(f"\nNode 0 IP: {NODE_0_IP}")
    print(f"Node 1 IP: {NODE_1_IP}")
    print(f"Node 2 IP: {NODE_2_IP}")
    print(f"\nOptimization Status:")
    print(f"  GPU Acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    print(f"  Caching: {'Enabled' if ENABLE_CACHING else 'Disabled'}")
    print(f"  Compression: {'Enabled' if ENABLE_COMPRESSION else 'Disabled'}")
    print(f"\nBatching Configuration:")
    print(f"  Min Batch Size: {MIN_BATCH_SIZE}")
    print(f"  Max Batch Size: {MAX_BATCH_SIZE}")
    print(f"  Base Batch Size: {BASE_BATCH_SIZE}")
    print(f"  Batch Timeout: {BATCH_TIMEOUT}s")
    print(f"  Max Batch Timeout: {MAX_BATCH_TIMEOUT}s")
    
    # Initialize services
    print("\nInitializing services...")
    embedder = EmbedderService()
    sentiment_safety = SentimentSafetyService()
    print("Services initialized!")
    
    # Start worker thread
    worker_thread = threading.Thread(target=process_batch_worker, daemon=True)
    worker_thread.start()
    print("Worker thread started!")
    
    # Start Flask server
    print(f"\nStarting Flask server on {NODE_0_IP}")
    hostname = NODE_0_IP.split(':')[0]
    port = int(NODE_0_IP.split(':')[1]) if ':' in NODE_0_IP else 8000
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()
