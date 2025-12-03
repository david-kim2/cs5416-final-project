#!/usr/bin/env python3
"""
Comparison script for Monolithic vs Distributed Pipeline
Measures latency, throughput, and provides comparison metrics
"""

import os
import sys
import time
import subprocess
import requests
import json
from typing import Dict, List
from datetime import datetime

# Configuration
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
TEST_QUERIES = [
    "How do I return a defective product?",
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change my shipping address after placing an order?",
]

def wait_for_service(url: str, timeout: int = 60) -> bool:
    """Wait for a service to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def send_request(url: str, request_id: str, query: str) -> Dict:
    """Send a single request and measure latency"""
    payload = {'request_id': request_id, 'query': query}
    start_time = time.time()
    try:
        response = requests.post(f"{url}/query", json=payload, timeout=300)
        elapsed = time.time() - start_time
        if response.status_code == 200:
            return {
                'success': True,
                'latency': elapsed,
                'response': response.json()
            }
        else:
            return {
                'success': False,
                'latency': elapsed,
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'latency': elapsed,
            'error': str(e)
        }

def test_pipeline(pipeline_name: str, url: str, num_requests: int = 6) -> Dict:
    """Test a pipeline and collect metrics"""
    print(f"\n{'='*70}")
    print(f"Testing {pipeline_name} Pipeline")
    print(f"{'='*70}")
    
    # Wait for service to be ready
    print(f"Waiting for service at {url}...")
    if not wait_for_service(url):
        print(f"ERROR: Service at {url} did not become available")
        return {'error': 'Service not available'}
    
    print(f"Service is ready! Starting {num_requests} requests...")
    
    results = []
    start_time = time.time()
    
    for i in range(num_requests):
        request_id = f"test_{pipeline_name}_{int(time.time())}_{i}"
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        
        print(f"\n[{i+1}/{num_requests}] Sending: {query[:50]}...")
        result = send_request(url, request_id, query)
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Success in {result['latency']:.2f}s")
            print(f"  Sentiment: {result['response'].get('sentiment')}")
            print(f"  Is Toxic: {result['response'].get('is_toxic')}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Small delay between requests
        if i < num_requests - 1:
            time.sleep(2)
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    latencies = [r['latency'] for r in results if r['success']]
    
    metrics = {
        'pipeline': pipeline_name,
        'total_requests': num_requests,
        'successful_requests': successful,
        'failed_requests': num_requests - successful,
        'total_time': total_time,
        'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
        'min_latency': min(latencies) if latencies else 0,
        'max_latency': max(latencies) if latencies else 0,
        'throughput': successful / total_time if total_time > 0 else 0,  # requests per second
        'results': results
    }
    
    return metrics

def print_comparison(mono_metrics: Dict, dist_metrics: Dict):
    """Print comparison between two pipelines"""
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    
    if 'error' in mono_metrics or 'error' in dist_metrics:
        print("ERROR: One or both pipelines failed to run")
        return
    
    print(f"\n{'Metric':<30} {'Monolithic':<20} {'Distributed':<20} {'Difference':<20}")
    print("-" * 90)
    
    # Success rate
    mono_success = mono_metrics['successful_requests'] / mono_metrics['total_requests'] * 100
    dist_success = dist_metrics['successful_requests'] / dist_metrics['total_requests'] * 100
    print(f"{'Success Rate':<30} {mono_success:.1f}%{'':<15} {dist_success:.1f}%{'':<15} {dist_success-mono_success:+.1f}%")
    
    # Average latency
    mono_lat = mono_metrics['avg_latency']
    dist_lat = dist_metrics['avg_latency']
    print(f"{'Avg Latency (s)':<30} {mono_lat:.2f}{'':<15} {dist_lat:.2f}{'':<15} {dist_lat-mono_lat:+.2f}s")
    
    # Min latency
    mono_min = mono_metrics['min_latency']
    dist_min = dist_metrics['min_latency']
    print(f"{'Min Latency (s)':<30} {mono_min:.2f}{'':<15} {dist_min:.2f}{'':<15} {dist_min-mono_min:+.2f}s")
    
    # Max latency
    mono_max = mono_metrics['max_latency']
    dist_max = dist_metrics['max_latency']
    print(f"{'Max Latency (s)':<30} {mono_max:.2f}{'':<15} {dist_max:.2f}{'':<15} {dist_max-mono_max:+.2f}s")
    
    # Throughput
    mono_tp = mono_metrics['throughput'] * 60  # requests per minute
    dist_tp = dist_metrics['throughput'] * 60
    print(f"{'Throughput (req/min)':<30} {mono_tp:.2f}{'':<15} {dist_tp:.2f}{'':<15} {dist_tp-mono_tp:+.2f}")
    
    # Total time
    mono_total = mono_metrics['total_time']
    dist_total = dist_metrics['total_time']
    print(f"{'Total Time (s)':<30} {mono_total:.2f}{'':<15} {dist_total:.2f}{'':<15} {dist_total-mono_total:+.2f}s")
    
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    if dist_lat < mono_lat:
        improvement = ((mono_lat - dist_lat) / mono_lat) * 100
        print(f"✓ Distributed pipeline is {improvement:.1f}% faster on average")
    else:
        slowdown = ((dist_lat - mono_lat) / mono_lat) * 100
        print(f"⚠ Distributed pipeline is {slowdown:.1f}% slower on average")
    
    if dist_tp > mono_tp:
        improvement = ((dist_tp - mono_tp) / mono_tp) * 100
        print(f"✓ Distributed pipeline has {improvement:.1f}% higher throughput")
    else:
        reduction = ((mono_tp - dist_tp) / mono_tp) * 100
        print(f"⚠ Distributed pipeline has {reduction:.1f}% lower throughput")
    
    print(f"\nNote: These results are for sequential requests.")
    print(f"For better comparison, test with concurrent requests or use the client.py script.")

def main():
    """Main comparison function"""
    print("="*70)
    print("PIPELINE COMPARISON TOOL")
    print("="*70)
    print("\nThis script will test both monolithic and distributed pipelines.")
    print("Make sure you have:")
    print("  1. Monolithic pipeline running (TOTAL_NODES=1 NODE_NUMBER=0 ./run.sh)")
    print("  2. OR distributed pipeline running (all 3 nodes)")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Test monolithic (if running)
    print("\n" + "="*70)
    print("STEP 1: Testing Monolithic Pipeline")
    print("="*70)
    mono_url = f"http://{NODE_0_IP}"
    mono_metrics = test_pipeline("Monolithic", mono_url)
    
    if 'error' in mono_metrics:
        print("\n⚠ Monolithic pipeline test failed. Continuing with distributed only...")
        mono_metrics = None
    
    # Test distributed (if running)
    print("\n" + "="*70)
    print("STEP 2: Testing Distributed Pipeline")
    print("="*70)
    dist_url = f"http://{NODE_0_IP}"
    dist_metrics = test_pipeline("Distributed", dist_url)
    
    if 'error' in dist_metrics:
        print("\n⚠ Distributed pipeline test failed.")
        return
    
    # Compare if both succeeded
    if mono_metrics and not 'error' in mono_metrics:
        print_comparison(mono_metrics, dist_metrics)
    else:
        print("\n" + "="*70)
        print("DISTRIBUTED PIPELINE RESULTS")
        print("="*70)
        print(f"Successful requests: {dist_metrics['successful_requests']}/{dist_metrics['total_requests']}")
        print(f"Average latency: {dist_metrics['avg_latency']:.2f}s")
        print(f"Throughput: {dist_metrics['throughput']*60:.2f} requests/minute")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comparison_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'monolithic': mono_metrics,
            'distributed': dist_metrics,
            'timestamp': timestamp
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()

