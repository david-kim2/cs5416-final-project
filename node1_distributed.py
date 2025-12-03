#!/usr/bin/env python3
"""
Node 1 Service: FAISS + Document Retrieval + Reranker
- Receives embeddings from Node 0
- Performs FAISS ANN search
- Fetches documents from disk
- Reranks documents
- Returns reranked documents
"""

import os
import gc
import sqlite3
import numpy as np
import torch
import faiss
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Read environment variables
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8001')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768,
    'retrieval_k': 10,
    'truncate_length': 512
}

# Flask app
app = Flask(__name__)

class FAISSService:
    """Service for FAISS search and document retrieval"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"[Node 1] Initializing FAISS service on {self.device}")
        print(f"[Node 1] FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"[Node 1] Documents path: {CONFIG['documents_path']}")
        
        # Load FAISS index
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError("FAISS index not found. Please create the index before running the pipeline.")
        
        print("[Node 1] Loading FAISS index...")
        self.index = faiss.read_index(CONFIG['faiss_index_path'])
        print("[Node 1] FAISS index loaded!")
        
        # Initialize reranker model
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        print("[Node 1] Loading reranker model...")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_model_name
        ).to(self.device)
        self.reranker_model.eval()
        print("[Node 1] Reranker model loaded!")
    
    def search_batch(self, embeddings: List[List[float]]) -> List[List[int]]:
        """Perform FAISS ANN search for a batch of embeddings"""
        embeddings_array = np.array(embeddings, dtype='float32')
        _, indices = self.index.search(embeddings_array, CONFIG['retrieval_k'])
        return [row.tolist() for row in indices]
    
    def fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
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
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        return reranked_batches


# Global service
faiss_service = None

@app.route('/process', methods=['POST'])
def handle_process():
    """Handle processing requests from Node 0"""
    try:
        data = request.json
        embeddings = data.get('embeddings')
        queries = data.get('queries')
        request_ids = data.get('request_ids')
        
        if not embeddings or not queries or not request_ids:
            return jsonify({'error': 'Missing required fields'}), 400
        
        print(f"\n[Node 1] Processing batch of {len(queries)} requests")
        
        # Step 1: FAISS search
        print(f"[Node 1] Performing FAISS search...")
        doc_id_batches = faiss_service.search_batch(embeddings)
        
        # Step 2: Fetch documents
        print(f"[Node 1] Fetching documents...")
        documents_batch = faiss_service.fetch_documents_batch(doc_id_batches)
        
        # Step 3: Rerank documents
        print(f"[Node 1] Reranking documents...")
        reranked_docs = faiss_service.rerank_documents_batch(queries, documents_batch)
        
        print(f"[Node 1] Processing complete for {len(queries)} requests")
        
        return jsonify({
            'reranked_documents': reranked_docs,
            'request_ids': request_ids
        }), 200
        
    except Exception as e:
        print(f"[Node 1] Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': 1,
        'service': 'faiss+retrieval+reranker'
    }), 200

def main():
    """Main execution function"""
    global faiss_service
    
    print("="*60)
    print("NODE 1 SERVICE: FAISS + RETRIEVAL + RERANKER")
    print("="*60)
    print(f"Node 1 IP: {NODE_1_IP}")
    print("="*60)
    
    # Initialize FAISS service
    print("\nInitializing FAISS service...")
    faiss_service = FAISSService()
    
    # Start Flask server
    print(f"\nStarting Flask server on {NODE_1_IP}")
    hostname = NODE_1_IP.split(':')[0]
    port = int(NODE_1_IP.split(':')[1]) if ':' in NODE_1_IP else 8001
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()

