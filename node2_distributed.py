#!/usr/bin/env python3
"""
Node 2 Service: LLM + Sentiment Analysis + Safety Filter
- Receives queries and documents from Node 0
- Generates LLM responses
- Analyzes sentiment
- Applies safety filter
- Returns results
"""

import os
import gc
import torch
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline
)

# Read environment variables
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8002')
ONLY_CPU = os.environ.get('ONLY_CPU', 'true').lower() == 'true'

# Configuration
CONFIG = {
    'max_tokens': 128,
    'truncate_length': 512
}

# Flask app
app = Flask(__name__)

class LLMService:
    """Service for LLM generation, sentiment analysis, and safety filtering"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"[Node 2] Initializing LLM service on {self.device}")
        
        # Model names
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'
        
        # Load LLM
        print("[Node 2] Loading LLM model...")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16,
        ).to(self.device)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        print("[Node 2] LLM model loaded!")
        
        # Sentiment and safety models will be loaded on-demand to save memory
    
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
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Analyze sentiment for each generated response"""
        classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
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
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result['label'], 'neutral'))
        del classifier
        gc.collect()
        return sentiments
    
    def filter_response_safety_batch(self, texts: List[str]) -> List[str]:
        """Filter responses for safety for each entry in the batch"""
        classifier = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device
        )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append("true" if result['score'] > 0.5 else "false")
        del classifier
        gc.collect()
        return toxicity_flags


# Global service
llm_service = None

@app.route('/process', methods=['POST'])
def handle_process():
    """Handle processing requests from Node 0"""
    try:
        data = request.json
        queries = data.get('queries')
        documents = data.get('documents')
        request_ids = data.get('request_ids')
        
        if not queries or not documents or not request_ids:
            return jsonify({'error': 'Missing required fields'}), 400
        
        print(f"\n[Node 2] Processing batch of {len(queries)} requests")
        
        # Step 1: Generate LLM responses
        print(f"[Node 2] Generating LLM responses...")
        responses_text = llm_service.generate_responses_batch(queries, documents)
        
        # Step 2: Sentiment analysis
        print(f"[Node 2] Analyzing sentiment...")
        sentiments = llm_service.analyze_sentiment_batch(responses_text)
        
        # Step 3: Safety filter
        print(f"[Node 2] Applying safety filter...")
        toxicity_flags = llm_service.filter_response_safety_batch(responses_text)
        
        print(f"[Node 2] Processing complete for {len(queries)} requests")
        
        return jsonify({
            'responses': responses_text,
            'sentiments': sentiments,
            'is_toxic': toxicity_flags,
            'request_ids': request_ids
        }), 200
        
    except Exception as e:
        print(f"[Node 2] Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': 2,
        'service': 'llm+sentiment+safety'
    }), 200

def main():
    """Main execution function"""
    global llm_service
    
    print("="*60)
    print("NODE 2 SERVICE: LLM + SENTIMENT + SAFETY")
    print("="*60)
    print(f"Node 2 IP: {NODE_2_IP}")
    print(f"CPU only: {ONLY_CPU}")
    print("="*60)
    
    # Initialize LLM service
    print("\nInitializing LLM service...")
    llm_service = LLMService()
    
    # Start Flask server
    print(f"\nStarting Flask server on {NODE_2_IP}")
    hostname = NODE_2_IP.split(':')[0]
    port = int(NODE_2_IP.split(':')[1]) if ':' in NODE_2_IP else 8002
    app.run(host=hostname, port=port, threaded=True)

if __name__ == "__main__":
    main()

