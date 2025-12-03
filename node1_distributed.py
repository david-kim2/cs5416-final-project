"""
Node 1 Service: FAISS Search + Document Retrieval + Reranking
- Receives embeddings from Node 0
- Performs FAISS ANN search
- Fetches documents from disk
- Reranks documents
- Returns reranked documents to Node 0
"""

import os
import gc
import json
import gzip
import base64
import sqlite3
import threading
import numpy as np
import torch
import faiss
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify

# Environment variables
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8001')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'true').lower() == 'true'
ENABLE_COMPRESSION = os.environ.get('ENABLE_COMPRESSION', 'true').lower() == 'true'

CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768,
    'retrieval_k': 10,
    'truncate_length': 512
}

app = Flask(__name__)

# Document cache
document_cache = {}
document_cache_lock = threading.Lock()
cache_max_size = 2000  # Reduced from 5000 to stay within 16GB RAM

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

class FAISSService:
    """Service for FAISS search, document retrieval, and reranking"""
    def __init__(self):
        self.device = get_device()
        print("[Node 1] Loading FAISS index...")
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError(f"FAISS index not found at {CONFIG['faiss_index_path']}")
        self.index = faiss.read_index(CONFIG['faiss_index_path'])
        print(f"[Node 1] FAISS index loaded! (dim={self.index.d}, size={self.index.ntotal})")
        
        # Load reranker model
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        print("[Node 1] Loading reranker model...")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_model_name
        ).to(self.device)
        self.reranker_model.eval()
        print("[Node 1] Reranker model loaded!")
    
    def search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Perform FAISS ANN search for a batch of embeddings"""
        query_embeddings = query_embeddings.astype('float32')
        _, indices = self.index.search(query_embeddings, CONFIG['retrieval_k'])
        return [row.tolist() for row in indices]
    
    def fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Fetch documents for each query in the batch using SQLite with caching"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                # Check cache first
                if ENABLE_CACHING:
                    with document_cache_lock:
                        if doc_id in document_cache:
                            documents.append(document_cache[doc_id])
                            continue
                
                # Fetch from database
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    doc = {
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    }
                    documents.append(doc)
                    
                    # Update cache
                    if ENABLE_CACHING:
                        with document_cache_lock:
                            if len(document_cache) >= cache_max_size:
                                # Remove oldest entry (FIFO-like)
                                oldest_key = next(iter(document_cache))
                                del document_cache[oldest_key]
                            document_cache[doc_id] = doc
            
            documents_batch.append(documents)
        
        conn.close()
        return documents_batch
    
    def rerank_documents_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Rerank retrieved documents for each query in the batch"""
        reranked_batches = []
        
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            
            doc_scores = list(zip(documents, scores.cpu().numpy()))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        
        return reranked_batches

# Global service
faiss_service = None

@app.route('/search_and_rerank', methods=['POST'])
def search_and_rerank():
    """Handle FAISS search and reranking requests from Node 0"""
    try:
        data = request.json
        compressed_data = data.get('data')
        is_compressed = data.get('compressed', False)
        
        # Decompress if needed
        if is_compressed:
            decompressed = decompress_data(compressed_data)
            request_data = json.loads(decompressed.decode('utf-8'))
        else:
            request_data = json.loads(base64.b64decode(compressed_data.encode('utf-8')).decode('utf-8'))
        
        embeddings = np.array(request_data['embeddings'], dtype=np.float32)
        queries = request_data['queries']
        request_ids = request_data['request_ids']
        
        print(f"[Node 1] Processing batch of {len(queries)} queries")
        
        # Step 1: FAISS search
        doc_id_batches = faiss_service.search_batch(embeddings)
        
        # Step 2: Fetch documents
        documents_batch = faiss_service.fetch_documents_batch(doc_id_batches)
        
        # Step 3: Rerank documents
        reranked_docs_batch = faiss_service.rerank_documents_batch(queries, documents_batch)
        
        # Prepare response
        response_data = {
            'reranked_documents': reranked_docs_batch,
            'request_ids': request_ids
        }
        
        # Compress response
        json_data = json.dumps(response_data).encode('utf-8')
        compressed_response = compress_data(json_data)
        
        return jsonify({
            'data': compressed_response,
            'compressed': ENABLE_COMPRESSION
        }), 200
    
    except Exception as e:
        import traceback
        print(f"[Node 1] Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'faiss+retrieval+reranking',
        'node': 1
    }), 200

def main():
    """Main execution function"""
    global faiss_service
    
    print("="*60)
    print("NODE 1 SERVICE: FAISS + RETRIEVAL + RERANKING")
    print("="*60)
    print(f"\nNode 1 IP: {NODE_1_IP}")
    print(f"FAISS Index Path: {CONFIG['faiss_index_path']}")
    print(f"Documents Path: {CONFIG['documents_path']}")
    print(f"\nOptimization Status:")
    print(f"  GPU Acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    print(f"  Caching: {'Enabled' if ENABLE_CACHING else 'Disabled'}")
    print(f"  Compression: {'Enabled' if ENABLE_COMPRESSION else 'Disabled'}")
    
    # Initialize service
    print("\nInitializing service...")
    faiss_service = FAISSService()
    print("Service initialized!")
    
    # Start Flask server
    print(f"\nStarting Flask server on {NODE_1_IP}")
    hostname = NODE_1_IP.split(':')[0]
    port = int(NODE_1_IP.split(':')[1]) if ':' in NODE_1_IP else 8001
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()
