"""
Node 2 Service: LLM Generation
- Receives queries and documents from Node 0
- Generates LLM responses
- Returns responses to Node 0
"""

import os
import gc
import json
import gzip
import base64
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify

# Environment variables
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8002')
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
ENABLE_COMPRESSION = os.environ.get('ENABLE_COMPRESSION', 'true').lower() == 'true'

CONFIG = {
    'max_tokens': 128,
    'truncate_length': 512
}

app = Flask(__name__)

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

class LLMService:
    """Service for LLM response generation"""
    def __init__(self):
        self.device = get_device()
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        print("[Node 2] Loading LLM model...")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
        ).to(self.device)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        print("[Node 2] LLM model loaded!")
    
    def generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
        """Generate LLM responses for each query in the batch"""
        responses = []
        
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]])
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)
            
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    **model_inputs,
                    max_new_tokens=CONFIG['max_tokens'],
                    temperature=0.01,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        
        return responses

# Global service
llm_service = None

@app.route('/generate', methods=['POST'])
def generate():
    """Handle LLM generation requests from Node 0"""
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
        
        queries = request_data['queries']
        documents_batch = request_data['documents_batch']
        request_ids = request_data['request_ids']
        
        print(f"[Node 2] Generating responses for {len(queries)} queries")
        
        # Generate LLM responses
        responses = llm_service.generate_responses_batch(queries, documents_batch)
        
        # Prepare response
        response_data = {
            'responses': responses,
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
        print(f"[Node 2] Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'llm-generation',
        'node': 2
    }), 200

def main():
    """Main execution function"""
    global llm_service
    
    print("="*60)
    print("NODE 2 SERVICE: LLM GENERATION")
    print("="*60)
    print(f"\nNode 2 IP: {NODE_2_IP}")
    print(f"\nOptimization Status:")
    print(f"  GPU Acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    print(f"  Compression: {'Enabled' if ENABLE_COMPRESSION else 'Disabled'}")
    
    # Initialize service
    print("\nInitializing service...")
    llm_service = LLMService()
    print("Service initialized!")
    
    # Start Flask server
    print(f"\nStarting Flask server on {NODE_2_IP}")
    hostname = NODE_2_IP.split(':')[0]
    port = int(NODE_2_IP.split(':')[1]) if ':' in NODE_2_IP else 8002
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()
