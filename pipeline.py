import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
import requests
import psutil
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue, Empty
import threading
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# ==================================================================
# MEMORY MONITORING UTILITIES
# ==================================================================

def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_memory_stats():
    """Get detailed memory statistics"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

# Global memory tracking
peak_memory_mb = 0
initial_memory_mb = 0

def update_peak_memory():
    """Update peak memory usage"""
    global peak_memory_mb
    current = get_memory_usage_mb()
    if current > peak_memory_mb:
        peak_memory_mb = current
    return current

# Environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 3))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8001')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8002')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768,
    'max_tokens': 128,
    'retrieval_k': 10,
    'truncate_length': 512,
    'batch_size': 8,
    'batch_timeout': 2.0,
}

@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float

@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str

# ==================================================================
# SERVICE CLASSES
# ==================================================================

class EmbeddingService:
    """Generates embeddings - Node 0"""
    def __init__(self):
        print("[EmbeddingService] Loading model...")
        mem_before = get_memory_usage_mb()
        
        self.device = torch.device('cpu')
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5').to(self.device)
        
        mem_after = get_memory_usage_mb()
        mem_used = mem_after - mem_before
        print(f"[EmbeddingService] Ready (loaded {mem_used:.1f} MB, total: {mem_after:.1f} MB)")
        update_peak_memory()
    
    def generate_batch(self, texts: List[str]) -> np.ndarray:
        start = time.time()
        result = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        elapsed = time.time() - start
        return result, elapsed


class RetrievalService:
    """FAISS + reranking - Node 1"""
    def __init__(self):
        print("[RetrievalService] Loading FAISS index...")
        mem_before = get_memory_usage_mb()
        
        self.device = torch.device('cpu')
        self.faiss_index = faiss.read_index(CONFIG['faiss_index_path'])
        
        mem_after = get_memory_usage_mb()
        mem_used = mem_after - mem_before
        print(f"[RetrievalService] FAISS loaded: {self.faiss_index.ntotal} vectors ({mem_used:.1f} MB, total: {mem_after:.1f} MB)")
        update_peak_memory()
        
        db_path = os.path.join(CONFIG['documents_path'], 'documents.db')
        self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.db_lock = threading.Lock()
        
        print("[RetrievalService] Loading reranker...")
        mem_before = get_memory_usage_mb()
        
        self.reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            'BAAI/bge-reranker-base'
        ).to(self.device)
        self.reranker_model.eval()
        
        mem_after = get_memory_usage_mb()
        mem_used = mem_after - mem_before
        print(f"[RetrievalService] Reranker loaded ({mem_used:.1f} MB, total: {mem_after:.1f} MB)")
        update_peak_memory()
    
    def process_batch(self, embeddings: List[List[float]], queries: List[str]) -> tuple:
        timings = {}
        
        # FAISS search
        start = time.time()
        embeddings_array = np.array(embeddings, dtype='float32')
        _, indices = self.faiss_index.search(embeddings_array, CONFIG['retrieval_k'])
        timings['faiss_search'] = time.time() - start
        
        # Fetch documents
        start = time.time()
        documents_batch = []
        for doc_ids in indices:
            documents = self._fetch_documents(doc_ids.tolist())
            documents_batch.append(documents)
        timings['document_fetch'] = time.time() - start
        
        # Rerank
        start = time.time()
        reranked_batch = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batch.append([])
                continue
            
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs, padding=True, truncation=True,
                    return_tensors='pt', max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            
            doc_scores = list(zip(documents, scores.cpu().numpy()))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batch.append([doc for doc, _ in doc_scores])
        timings['reranking'] = time.time() - start
        
        return reranked_batch, timings
    
    def _fetch_documents(self, doc_ids: List[int]) -> List[Dict]:
        documents = []
        with self.db_lock:
            cursor = self.db_conn.cursor()
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (int(doc_id),)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
        return documents


class GenerationService:
    """LLM generation - Node 2"""
    def __init__(self):
        print("[GenerationService] Loading LLM...")
        mem_before = get_memory_usage_mb()
        
        self.device = torch.device('cpu')
        self.model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct',
            torch_dtype=torch.float16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
        
        mem_after = get_memory_usage_mb()
        mem_used = mem_after - mem_before
        print(f"[GenerationService] LLM loaded ({mem_used:.1f} MB, total: {mem_after:.1f} MB)")
        update_peak_memory()
    
    def generate_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> tuple:
        start = time.time()
        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([
                f"- {doc['title']}: {doc['content'][:200]}" 
                for doc in documents[:3]
            ])
            
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        
        elapsed = time.time() - start
        return responses, elapsed


class SentimentService:
    """Sentiment analysis - Node 0"""
    def __init__(self):
        print("[SentimentService] Loading model...")
        mem_before = get_memory_usage_mb()
        
        self.device = torch.device('cpu')
        self.classifier = hf_pipeline(
            "sentiment-analysis",
            model='nlptown/bert-base-multilingual-uncased-sentiment',
            device=self.device
        )
        self.sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        
        mem_after = get_memory_usage_mb()
        mem_used = mem_after - mem_before
        print(f"[SentimentService] Ready ({mem_used:.1f} MB, total: {mem_after:.1f} MB)")
        update_peak_memory()
    
    def analyze_batch(self, texts: List[str]) -> tuple:
        start = time.time()
        truncated = [text[:CONFIG['truncate_length']] for text in texts]
        results = self.classifier(truncated)
        sentiments = [self.sentiment_map.get(r['label'], 'neutral') for r in results]
        elapsed = time.time() - start
        return sentiments, elapsed


class SafetyService:
    """Toxicity detection - Node 0"""
    def __init__(self):
        print("[SafetyService] Loading model...")
        mem_before = get_memory_usage_mb()
        
        self.device = torch.device('cpu')
        self.classifier = hf_pipeline(
            "text-classification",
            model='unitary/toxic-bert',
            device=self.device
        )
        
        mem_after = get_memory_usage_mb()
        mem_used = mem_after - mem_before
        print(f"[SafetyService] Ready ({mem_used:.1f} MB, total: {mem_after:.1f} MB)")
        update_peak_memory()
    
    def check_batch(self, texts: List[str]) -> tuple:
        start = time.time()
        truncated = [text[:CONFIG['truncate_length']] for text in texts]
        results = self.classifier(truncated)
        toxicity = [r['score'] > 0.5 for r in results]
        elapsed = time.time() - start
        return toxicity, elapsed


# ==================================================================
# BATCHING
# ==================================================================

class BatchAccumulator:
    def __init__(self, max_size: int, timeout: float):
        self.max_size = max_size
        self.timeout = timeout
        self.batch = []
        self.lock = threading.Lock()
        self.first_item_time = None
    
    def add(self, item) -> Optional[List]:
        with self.lock:
            if not self.batch:
                self.first_item_time = time.time()
            self.batch.append(item)
            
            # Process if batch is full
            if len(self.batch) >= self.max_size:
                return self._extract()
        return None
    
    def check(self) -> Optional[List]:
        with self.lock:
            if self.batch and self.first_item_time:
                # Process if timeout reached since first item
                if time.time() - self.first_item_time >= self.timeout:
                    return self._extract()
        return None
    
    def _extract(self) -> List:
        batch = self.batch
        self.batch = []
        self.first_item_time = None
        return batch


# ==================================================================
# NODE 0: FRONTEND
# ==================================================================

def create_node0_app():
    app = Flask(__name__)
    
    print("\n" + "="*60)
    print("NODE 0: FRONTEND + ORCHESTRATION")
    print("="*60)
    
    # Initialize services
    embedding_svc = EmbeddingService()
    sentiment_svc = SentimentService()
    safety_svc = SafetyService()
    
    # State
    results = {}
    results_lock = threading.Lock()
    batch_acc = BatchAccumulator(CONFIG['batch_size'], CONFIG['batch_timeout'])
    
    def process_batches():
        print("[Node0] Batch worker started")
        
        # Profiling data
        batch_count = 0
        total_requests = 0
        stage_times = {
            'embedding': [],
            'faiss_search': [],
            'document_fetch': [],
            'reranking': [],
            'llm_generation': [],
            'sentiment': [],
            'safety': [],
            'total_batch': []
        }
        
        while True:
            try:
                time.sleep(0.5)
                batch = batch_acc.check()
                if not batch:
                    continue
                
                batch_start = time.time()
                batch_count += 1
                batch_size = len(batch)
                total_requests += batch_size
                
                mem_before = get_memory_usage_mb()
                
                print(f"\n{'='*60}")
                print(f"[Node0] Processing batch #{batch_count} ({batch_size} requests)")
                print(f"[Node0] Memory: {mem_before:.1f} MB")
                print(f"{'='*60}")
                
                request_ids = [item['request_id'] for item in batch]
                queries = [item['query'] for item in batch]
                
                # Step 1: Embeddings
                print(f"[Node0] Step 1/7: Generating embeddings...")
                embeddings, embed_time = embedding_svc.generate_batch(queries)
                stage_times['embedding'].append(embed_time)
                print(f"  └─ Completed in {embed_time:.3f}s ({embed_time/batch_size:.3f}s per request)")
                
                # Step 2-4: Call Node 1 (FAISS + Docs + Rerank)
                print(f"[Node0] Step 2-4: Calling Node 1 for retrieval...")
                node1_start = time.time()
                response = requests.post(
                    f"http://{NODE_1_IP}/retrieve_batch",
                    json={'embeddings': embeddings.tolist(), 'queries': queries},
                    timeout=120
                )
                result = response.json()
                documents_batch = result['documents_batch']
                node1_timings = result['timings']
                node1_total = time.time() - node1_start
                
                stage_times['faiss_search'].append(node1_timings['faiss_search'])
                stage_times['document_fetch'].append(node1_timings['document_fetch'])
                stage_times['reranking'].append(node1_timings['reranking'])
                
                print(f"  ├─ FAISS search: {node1_timings['faiss_search']:.3f}s")
                print(f"  ├─ Document fetch: {node1_timings['document_fetch']:.3f}s")
                print(f"  ├─ Reranking: {node1_timings['reranking']:.3f}s")
                print(f"  └─ Total (incl. network): {node1_total:.3f}s")
                
                # Step 5: Call Node 2 (LLM)
                print(f"[Node0] Step 5/7: Calling Node 2 for generation...")
                node2_start = time.time()
                response = requests.post(
                    f"http://{NODE_2_IP}/generate_batch",
                    json={'queries': queries, 'documents_batch': documents_batch},
                    timeout=300
                )
                result = response.json()
                responses = result['responses']
                llm_time = result['timing']
                node2_total = time.time() - node2_start
                
                stage_times['llm_generation'].append(llm_time)
                print(f"  ├─ LLM generation: {llm_time:.3f}s ({llm_time/batch_size:.3f}s per request)")
                print(f"  └─ Total (incl. network): {node2_total:.3f}s")
                
                # Step 6-7: Sentiment + Safety (parallel)
                print(f"[Node0] Step 6-7: Analyzing sentiment and safety (parallel)...")
                parallel_start = time.time()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    sentiment_future = executor.submit(sentiment_svc.analyze_batch, responses)
                    safety_future = executor.submit(safety_svc.check_batch, responses)
                    
                    sentiments, sentiment_time = sentiment_future.result()
                    toxicity, safety_time = safety_future.result()
                
                parallel_total = time.time() - parallel_start
                stage_times['sentiment'].append(sentiment_time)
                stage_times['safety'].append(safety_time)
                
                print(f"  ├─ Sentiment analysis: {sentiment_time:.3f}s")
                print(f"  ├─ Safety check: {safety_time:.3f}s")
                print(f"  └─ Parallel execution: {parallel_total:.3f}s (max of both)")
                
                # Store results
                with results_lock:
                    for i, req_id in enumerate(request_ids):
                        results[req_id] = {
                            'request_id': req_id,
                            'generated_response': responses[i],
                            'sentiment': sentiments[i],
                            'is_toxic': 'true' if toxicity[i] else 'false'
                        }
                
                batch_total = time.time() - batch_start
                stage_times['total_batch'].append(batch_total)
                mem_after = get_memory_usage_mb()
                update_peak_memory()
                
                print(f"\n{'='*60}")
                print(f"[Node0] ✓ Batch #{batch_count} complete")
                print(f"  Total time: {batch_total:.3f}s")
                print(f"  Throughput: {batch_size/batch_total:.2f} req/s")
                print(f"  Avg latency: {batch_total/batch_size:.3f}s per request")
                print(f"  Memory: {mem_after:.1f} MB (peak: {peak_memory_mb:.1f} MB)")
                print(f"{'='*60}")
                
                # Print cumulative statistics every 5 batches
                if batch_count % 5 == 0:
                    print(f"\n{'='*60}")
                    print(f"CUMULATIVE STATISTICS (after {batch_count} batches, {total_requests} requests)")
                    print(f"{'='*60}")
                    print(f"Average Stage Times (per batch):")
                    print(f"  Embedding:        {np.mean(stage_times['embedding']):.3f}s")
                    print(f"  FAISS search:     {np.mean(stage_times['faiss_search']):.3f}s")
                    print(f"  Document fetch:   {np.mean(stage_times['document_fetch']):.3f}s")
                    print(f"  Reranking:        {np.mean(stage_times['reranking']):.3f}s")
                    print(f"  LLM generation:   {np.mean(stage_times['llm_generation']):.3f}s")
                    print(f"  Sentiment:        {np.mean(stage_times['sentiment']):.3f}s")
                    print(f"  Safety:           {np.mean(stage_times['safety']):.3f}s")
                    print(f"  Total batch:      {np.mean(stage_times['total_batch']):.3f}s")
                    print(f"\nAverage per-request times:")
                    avg_batch_size = total_requests / batch_count
                    print(f"  Embedding:        {np.mean(stage_times['embedding'])/avg_batch_size:.3f}s")
                    print(f"  FAISS search:     {np.mean(stage_times['faiss_search'])/avg_batch_size:.3f}s")
                    print(f"  Document fetch:   {np.mean(stage_times['document_fetch'])/avg_batch_size:.3f}s")
                    print(f"  Reranking:        {np.mean(stage_times['reranking'])/avg_batch_size:.3f}s")
                    print(f"  LLM generation:   {np.mean(stage_times['llm_generation'])/avg_batch_size:.3f}s")
                    print(f"  Sentiment:        {np.mean(stage_times['sentiment'])/avg_batch_size:.3f}s")
                    print(f"  Safety:           {np.mean(stage_times['safety'])/avg_batch_size:.3f}s")
                    print(f"\nOverall throughput: {total_requests/sum(stage_times['total_batch']):.2f} req/s")
                    print(f"Peak memory: {peak_memory_mb:.1f} MB")
                    print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"[Node0] Error: {e}")
                import traceback
                traceback.print_exc()
                if batch:
                    with results_lock:
                        for item in batch:
                            results[item['request_id']] = {'error': str(e)}
    
    threading.Thread(target=process_batches, daemon=True).start()
    
    @app.route('/query', methods=['POST'])
    def handle_query():
        try:
            data = request.json
            request_id = data.get('request_id')
            query = data.get('query')
            
            if not request_id or not query:
                return jsonify({'error': 'Missing request_id or query'}), 400
            
            batch_acc.add({'request_id': request_id, 'query': query, 'timestamp': time.time()})
            
            # Wait for result
            timeout = 300
            start = time.time()
            while True:
                with results_lock:
                    if request_id in results:
                        result = results.pop(request_id)
                        if 'error' in result:
                            return jsonify({'error': result['error']}), 500
                        return jsonify(result), 200
                
                if time.time() - start > timeout:
                    return jsonify({'error': 'Timeout'}), 504
                
                time.sleep(0.1)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'node': 0}), 200
    
    # Print memory summary
    final_mem = get_memory_usage_mb()
    print(f"\n{'='*60}")
    print(f"[Node0] Initialization Complete")
    print(f"  Total memory usage: {final_mem:.1f} MB")
    print(f"  Peak memory usage: {peak_memory_mb:.1f} MB")
    print(f"{'='*60}\n")
    
    print("\n[Node0] Ready!")
    return app


# ==================================================================
# NODE 1: RETRIEVAL
# ==================================================================

def create_node1_app():
    app = Flask(__name__)
    
    print("\n" + "="*60)
    print("NODE 1: RETRIEVAL SERVICE")
    print("="*60)
    
    retrieval_svc = RetrievalService()
    
    @app.route('/retrieve_batch', methods=['POST'])
    def retrieve_batch():
        try:
            data = request.json
            embeddings = data['embeddings']
            queries = data['queries']
            
            print(f"[Node1] Processing batch of {len(embeddings)}")
            documents_batch, timings = retrieval_svc.process_batch(embeddings, queries)
            
            return jsonify({
                'documents_batch': documents_batch,
                'timings': timings
            }), 200
        except Exception as e:
            print(f"[Node1] Error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'node': 1}), 200
    
    # Print memory summary
    final_mem = get_memory_usage_mb()
    print(f"\n{'='*60}")
    print(f"[Node1] Initialization Complete")
    print(f"  Total memory usage: {final_mem:.1f} MB")
    print(f"  Peak memory usage: {peak_memory_mb:.1f} MB")
    print(f"  Available headroom: {16384 - final_mem:.1f} MB (out of 16384 MB)")
    print(f"{'='*60}\n")
    
    print("\n[Node1] Ready!")
    return app


# ==================================================================
# NODE 2: GENERATION
# ==================================================================

def create_node2_app():
    app = Flask(__name__)
    
    print("\n" + "="*60)
    print("NODE 2: GENERATION SERVICE")
    print("="*60)
    
    generation_svc = GenerationService()
    
    @app.route('/generate_batch', methods=['POST'])
    def generate_batch():
        try:
            data = request.json
            queries = data['queries']
            documents_batch = data['documents_batch']
            
            print(f"[Node2] Generating for batch of {len(queries)}")
            responses, timing = generation_svc.generate_batch(queries, documents_batch)
            
            return jsonify({
                'responses': responses,
                'timing': timing
            }), 200
        except Exception as e:
            print(f"[Node2] Error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'node': 2}), 200
    
    # Print memory summary
    final_mem = get_memory_usage_mb()
    print(f"\n{'='*60}")
    print(f"[Node2] Initialization Complete")
    print(f"  Total memory usage: {final_mem:.1f} MB")
    print(f"  Peak memory usage: {peak_memory_mb:.1f} MB")
    print(f"{'='*60}\n")
    
    print("\n[Node2] Ready!")
    return app


# ==================================================================
# MAIN
# ==================================================================

def main():
    global initial_memory_mb, peak_memory_mb
    
    initial_memory_mb = get_memory_usage_mb()
    peak_memory_mb = initial_memory_mb
    
    print("\n" + "="*70)
    print("DISTRIBUTED ML INFERENCE PIPELINE")
    print("="*70)
    print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
    print(f"Initial memory: {initial_memory_mb:.1f} MB")
    print("="*70 + "\n")
    
    if NODE_NUMBER == 0:
        app = create_node0_app()
        port = int(NODE_0_IP.split(':')[1]) if ':' in NODE_0_IP else 8000
    elif NODE_NUMBER == 1:
        app = create_node1_app()
        port = int(NODE_1_IP.split(':')[1]) if ':' in NODE_1_IP else 8001
    elif NODE_NUMBER == 2:
        app = create_node2_app()
        port = int(NODE_2_IP.split(':')[1]) if ':' in NODE_2_IP else 8002
    else:
        raise ValueError(f"Invalid NODE_NUMBER: {NODE_NUMBER}")
    
    print(f"\nStarting server on 0.0.0.0:{port}\n")
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == "__main__":
    main()