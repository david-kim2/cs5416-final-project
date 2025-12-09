#!/usr/bin/env python3
"""
Test script for profiling the ML inference pipeline.
Can send many requests quickly to test opportunistic batching.
"""

import os
import time
import requests
import json
import threading
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Read NODE_0_IP from environment variable
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
SERVER_URL = f"http://{NODE_0_IP}/query"

# Test queries - expanded list for testing with many requests (100 queries)
TEST_QUERIES = [
    "How do I return a defective product?",
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change my shipping address after placing an order?",
    "What payment methods do you accept?",
    "How long does shipping typically take?",
    "Do you offer international shipping?",
    "What is your return policy?",
    "How can I track my order?",
    "What are your shipping options?",
    "Can I cancel my order?",
    "How do I contact customer service?",
    "What is your exchange policy?",
    "Do you have a loyalty program?",
    "What is your price match policy?",
    "How do I apply a discount code?",
    "Can I modify my order after placing it?",
    "What is your privacy policy?",
    "How do I create an account?",
    "What payment security measures do you have?",
    "Do you offer gift wrapping?",
    "What is your customer satisfaction guarantee?",
    "How do I unsubscribe from emails?",
    "Can I save items to a wishlist?",
    "What is your product review policy?",
    "How do I report a problem with my order?",
    "Do you have a mobile app?",
    "What are your store hours?",
    "How do I find a product?",
    "What is your shipping insurance policy?",
    "Can I schedule a delivery time?",
    "What happens if my package is damaged?",
    "How do I request a refund?",
    "What is your cancellation policy?",
    "Do you offer same-day delivery?",
    "How do I change my password?",
    "What is your data retention policy?",
    "Can I get a receipt by email?",
    "How do I add items to my cart?",
    "What is your return shipping policy?",
    "Can I use multiple discount codes?",
    "How do I check my order status?",
    "What is your product availability policy?",
    "Do you offer free shipping?",
    "How do I update my email address?",
    "What is your restocking fee policy?",
    "Can I return items without a receipt?",
    "How do I leave a product review?",
    "What is your gift card policy?",
    "Do you have a student discount?",
    "How do I track multiple orders?",
    "What is your damaged goods policy?",
    "Can I exchange for a different size?",
    "How do I set up auto-reorder?",
    "What is your subscription service?",
    "Do you offer bulk discounts?",
    "How do I manage my subscriptions?",
    "What is your price adjustment policy?",
    "Can I combine shipping on multiple orders?",
    "How do I find my order confirmation?",
    "What is your lost package policy?",
    "Do you offer expedited shipping?",
    "How do I update my phone number?",
    "What is your return window?",
    "Can I return opened items?",
    "How do I report fraudulent activity?",
    "What is your two-factor authentication?",
    "Do you have a rewards program?",
    "How do I redeem reward points?",
    "What is your referral program?",
    "Can I share my wishlist?",
    "How do I filter search results?",
    "What is your product comparison feature?",
    "Do you offer product recommendations?",
    "How do I save for later?",
    "What is your price drop alert?",
    "Can I set up price alerts?",
    "How do I view my purchase history?",
    "What is your order modification deadline?",
    "Do you offer pre-orders?",
    "How do I cancel a pre-order?",
    "What is your backorder policy?",
    "Can I reserve out-of-stock items?",
    "How do I get notified of restocks?",
    "What is your waitlist feature?",
    "Do you offer product bundles?",
    "How do I find related products?",
    "What is your cross-sell feature?",
    "Can I see recently viewed items?",
    "How do I clear my browsing history?",
    "What is your cookie policy?",
    "Do you track my browsing?",
    "How do I opt out of tracking?",
    "What is your GDPR compliance?",
    "Can I export my data?",
    "How do I delete my account?",
    "What is your account deletion policy?",
    "Do you store my credit card info?",
]

# Results storage
results = {}
results_lock = threading.Lock()
latencies = []


def send_request(request_id: str, query: str) -> Dict:
    """Send a single request to the server"""
    try:
        payload = {
            'request_id': request_id,
            'query': query
        }
        
        start_time = time.time()
        response = requests.post(SERVER_URL, json=payload, timeout=600)
        elapsed_time = time.time() - start_time
        
        with results_lock:
            latencies.append(elapsed_time)
        
        if response.status_code == 200:
            result = response.json()
            return {
                'request_id': request_id,
                'success': True,
                'latency': elapsed_time,
                'result': result
            }
        else:
            return {
                'request_id': request_id,
                'success': False,
                'latency': elapsed_time,
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            'request_id': request_id,
            'success': False,
            'latency': None,
            'error': str(e)
        }


def test_low_load(num_requests: int = 10, delay: float = 5.0):
    """Test under low load - send requests slowly"""
    print("="*70)
    print("LOW LOAD TEST")
    print("="*70)
    print(f"Sending {num_requests} requests with {delay}s delay between requests")
    print(f"Server: {SERVER_URL}")
    print("="*70)
    
    start_time = time.time()
    results_list = []
    
    for i in range(num_requests):
        request_id = f"low_load_{int(time.time())}_{i}"
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending request {i+1}/{num_requests}")
        print(f"Query: {query[:60]}...")
        
        result = send_request(request_id, query)
        results_list.append(result)
        
        if result['success']:
            print(f"  ✓ Success in {result['latency']:.2f}s")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        if i < num_requests - 1:
            time.sleep(delay)
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results_list if r['success']]
    failed = [r for r in results_list if not r['success']]
    
    if successful:
        avg_latency = sum(r['latency'] for r in successful) / len(successful)
        min_latency = min(r['latency'] for r in successful)
        max_latency = max(r['latency'] for r in successful)
    else:
        avg_latency = min_latency = max_latency = 0
    
    print("\n" + "="*70)
    print("LOW LOAD TEST RESULTS")
    print("="*70)
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Min latency: {min_latency:.2f}s")
    print(f"Max latency: {max_latency:.2f}s")
    print(f"Throughput: {len(successful)/total_time*60:.2f} req/min")
    print("="*70)
    
    return results_list


def test_high_load(num_requests: int = 50, max_workers: int = 20):
    """Test under high load - send requests quickly in parallel"""
    print("="*70)
    print("HIGH LOAD TEST (Maximum Throughput)")
    print("="*70)
    print(f"Sending {num_requests} requests in parallel (max {max_workers} concurrent)")
    print(f"Server: {SERVER_URL}")
    print("="*70)
    
    start_time = time.time()
    results_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_requests):
            request_id = f"high_load_{int(time.time())}_{i}"
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            
            future = executor.submit(send_request, request_id, query)
            futures.append((i+1, future, query))
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All {num_requests} requests submitted")
        print("Waiting for responses...\n")
        
        completed = 0
        for i, future, query in futures:
            result = future.result()
            results_list.append(result)
            completed += 1
            
            if result['success']:
                print(f"[{completed}/{num_requests}] ✓ Request {i} completed in {result['latency']:.2f}s")
            else:
                print(f"[{completed}/{num_requests}] ✗ Request {i} failed: {result.get('error', 'Unknown')}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results_list if r['success']]
    failed = [r for r in results_list if not r['success']]
    
    if successful:
        avg_latency = sum(r['latency'] for r in successful) / len(successful)
        min_latency = min(r['latency'] for r in successful)
        max_latency = max(r['latency'] for r in successful)
    else:
        avg_latency = min_latency = max_latency = 0
    
    print("\n" + "="*70)
    print("HIGH LOAD TEST RESULTS")
    print("="*70)
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Min latency: {min_latency:.2f}s")
    print(f"Max latency: {max_latency:.2f}s")
    print(f"Throughput: {len(successful)/total_time*60:.2f} req/min")
    print(f"Throughput: {len(successful)/total_time:.2f} req/s")
    print("="*70)
    
    return results_list


def test_variable_rate(num_requests: int = 50, rate: str = "slow"):
    """Test with variable request rates to see different batch sizes"""
    print("="*70)
    print("VARIABLE RATE TEST")
    print("="*70)
    print(f"Sending {num_requests} requests at {rate} rate")
    print(f"Server: {SERVER_URL}")
    print("="*70)
    
    # Define rate patterns
    rate_patterns = {
        "slow": 10.0,      # 10 seconds between requests (low load)
        "medium": 2.0,     # 2 seconds between requests (medium load)
        "fast": 0.5,       # 0.5 seconds between requests (high load)
        "very_fast": 0.1,  # 0.1 seconds between requests (very high load)
        "mixed": None      # Mixed rates
    }
    
    if rate not in rate_patterns and rate != "mixed":
        print(f"Unknown rate: {rate}. Using 'medium'")
        rate = "medium"
    
    start_time = time.time()
    results_list = []
    
    if rate == "mixed":
        # Mixed rate: start slow, speed up, then slow down
        delays = [5.0] * 10 + [1.0] * 20 + [0.2] * 10 + [1.0] * 10
        if len(delays) > num_requests:
            delays = delays[:num_requests]
    else:
        delay = rate_patterns[rate]
        delays = [delay] * (num_requests - 1)  # No delay after last request
    
    for i in range(num_requests):
        request_id = f"var_rate_{rate}_{int(time.time())}_{i}"
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending request {i+1}/{num_requests} (rate: {rate})")
        print(f"Query: {query[:60]}...")
        
        result = send_request(request_id, query)
        results_list.append(result)
        
        if result['success']:
            print(f"  ✓ Success in {result['latency']:.2f}s")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        if i < num_requests - 1:
            if rate == "mixed":
                time.sleep(delays[i])
            else:
                time.sleep(delay)
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results_list if r['success']]
    failed = [r for r in results_list if not r['success']]
    
    if successful:
        avg_latency = sum(r['latency'] for r in successful) / len(successful)
        min_latency = min(r['latency'] for r in successful)
        max_latency = max(r['latency'] for r in successful)
    else:
        avg_latency = min_latency = max_latency = 0
    
    print("\n" + "="*70)
    print("VARIABLE RATE TEST RESULTS")
    print("="*70)
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Min latency: {min_latency:.2f}s")
    print(f"Max latency: {max_latency:.2f}s")
    print(f"Throughput: {len(successful)/total_time*60:.2f} req/min")
    print("="*70)
    print(f"\nNOTE: Check Node 0 terminal for batch size distribution!")
    print(f"NOTE: Batch sizes are logged to batch_sizes.jsonl")
    print(f"      This experiment used rate: {rate}")
    print("="*70)
    
    # Save experiment metadata
    import json
    metadata = {
        'experiment_type': 'variable_rate',
        'rate': rate,
        'num_requests': num_requests,
        'start_time': start_time,
        'end_time': time.time(),
        'total_time': total_time,
        'successful': len(successful),
        'failed': len(failed),
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'throughput_req_per_min': len(successful)/total_time*60 if total_time > 0 else 0
    }
    metadata_file = f'experiment_metadata_{rate}_{int(start_time)}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Experiment metadata saved to: {metadata_file}")
    
    return results_list


def test_burst_load(num_requests: int = 30, burst_size: int = 10):
    """Test burst load - send requests in bursts"""
    print("="*70)
    print("BURST LOAD TEST")
    print("="*70)
    print(f"Sending {num_requests} requests in bursts of {burst_size}")
    print(f"Server: {SERVER_URL}")
    print("="*70)
    
    start_time = time.time()
    results_list = []
    
    num_bursts = (num_requests + burst_size - 1) // burst_size
    
    for burst_num in range(num_bursts):
        burst_start = burst_num * burst_size
        burst_end = min(burst_start + burst_size, num_requests)
        burst_size_actual = burst_end - burst_start
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Burst {burst_num+1}/{num_bursts}: Sending {burst_size_actual} requests")
        
        with ThreadPoolExecutor(max_workers=burst_size_actual) as executor:
            futures = []
            for i in range(burst_start, burst_end):
                request_id = f"burst_{int(time.time())}_{i}"
                query = TEST_QUERIES[i % len(TEST_QUERIES)]
                future = executor.submit(send_request, request_id, query)
                futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                results_list.append(result)
        
        if burst_num < num_bursts - 1:
            time.sleep(2)  # Wait 2 seconds between bursts
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results_list if r['success']]
    failed = [r for r in results_list if not r['success']]
    
    if successful:
        avg_latency = sum(r['latency'] for r in successful) / len(successful)
        min_latency = min(r['latency'] for r in successful)
        max_latency = max(r['latency'] for r in successful)
    else:
        avg_latency = min_latency = max_latency = 0
    
    print("\n" + "="*70)
    print("BURST LOAD TEST RESULTS")
    print("="*70)
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Min latency: {min_latency:.2f}s")
    print(f"Max latency: {max_latency:.2f}s")
    print(f"Throughput: {len(successful)/total_time*60:.2f} req/min")
    print("="*70)
    
    return results_list


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 test_profiling.py low [num_requests] [delay]")
        print("  python3 test_profiling.py high [num_requests] [max_workers]")
        print("  python3 test_profiling.py burst [num_requests] [burst_size]")
        print("  python3 test_profiling.py rate [num_requests] [rate]")
        print("\nTest Types:")
        print("  low    - Low load: send requests slowly (good for latency measurement)")
        print("  high   - High load: send many requests in parallel (good for max throughput)")
        print("  burst  - Burst load: send requests in bursts (good for adaptive batching)")
        print("  rate   - Variable rate: send at different rates (slow/medium/fast/very_fast/mixed)")
        print("\nExamples:")
        print("  python3 test_profiling.py low 10 5              # Low load: 10 requests, 5s delay")
        print("  python3 test_profiling.py high 50 20             # High load: 50 requests, 20 concurrent")
        print("  python3 test_profiling.py burst 30 10            # Burst: 30 requests in bursts of 10")
        print("  python3 test_profiling.py rate 50 slow           # Variable rate: 50 requests, slow (10s delay)")
        print("  python3 test_profiling.py rate 50 medium         # Variable rate: 50 requests, medium (2s delay)")
        print("  python3 test_profiling.py rate 50 fast           # Variable rate: 50 requests, fast (0.5s delay)")
        print("  python3 test_profiling.py rate 50 very_fast      # Variable rate: 50 requests, very fast (0.1s delay)")
        print("  python3 test_profiling.py rate 50 mixed          # Variable rate: 50 requests, mixed speeds")
        sys.exit(1)
    
    test_type = sys.argv[1].lower()
    
    if test_type == 'low':
        num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        delay = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        test_low_load(num_requests, delay)
    
    elif test_type == 'high':
        num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        test_high_load(num_requests, max_workers)
    
    elif test_type == 'burst':
        num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        burst_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        test_burst_load(num_requests, burst_size)
    
    elif test_type == 'rate':
        num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        rate = sys.argv[3] if len(sys.argv) > 3 else "medium"
        test_variable_rate(num_requests, rate)
    
    else:
        print(f"Unknown test type: {test_type}")
        print("Use 'low', 'high', 'burst', or 'rate'")
        sys.exit(1)


if __name__ == "__main__":
    main()

