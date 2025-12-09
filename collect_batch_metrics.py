#!/usr/bin/env python3
"""
Script to collect batch size, throughput, and memory usage data
Runs experiments at different rates and collects metrics
"""

import json
import os
import sys
import time
import subprocess
import re
from collections import defaultdict
from statistics import mean, median

def parse_batch_log(log_file='batch_sizes.jsonl'):
    """Parse batch_sizes.jsonl and extract metrics"""
    if not os.path.exists(log_file):
        return None
    
    batches = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                batches.append(json.loads(line))
    
    if not batches:
        return None
    
    # Calculate metrics
    batch_sizes = [b['batch_size'] for b in batches]
    timestamps = [b['timestamp'] for b in batches]
    
    # Extract throughput and memory from log entries if available
    throughputs_per_min = [b.get('throughput_per_min', 0) for b in batches if 'throughput_per_min' in b]
    memory_values = [b.get('memory_mb', 0) for b in batches if 'memory_mb' in b]
    peak_memory_values = [b.get('peak_memory_mb', 0) for b in batches if 'peak_memory_mb' in b]
    
    # Calculate overall throughput
    if len(timestamps) > 1:
        time_span = max(timestamps) - min(timestamps)
        total_requests = sum(batch_sizes)
        throughput_per_sec = total_requests / time_span if time_span > 0 else 0
        throughput_per_min = throughput_per_sec * 60
    else:
        throughput_per_sec = throughput_per_min = 0
        total_requests = sum(batch_sizes)
    
    # Use per-batch throughput if available, otherwise use overall
    if throughputs_per_min:
        avg_throughput_per_min = mean(throughputs_per_min)
    else:
        avg_throughput_per_min = throughput_per_min
    
    # Memory statistics
    avg_memory = mean(memory_values) if memory_values else None
    max_memory = max(memory_values) if memory_values else None
    peak_memory = max(peak_memory_values) if peak_memory_values else None
    
    return {
        'total_batches': len(batches),
        'total_requests': total_requests,
        'min_batch_size': min(batch_sizes),
        'max_batch_size': max(batch_sizes),
        'avg_batch_size': mean(batch_sizes),
        'median_batch_size': median(batch_sizes),
        'batch_size_distribution': dict(Counter(batch_sizes)),
        'throughput_per_sec': throughput_per_sec,
        'throughput_per_min': avg_throughput_per_min,  # Use per-batch average
        'avg_memory_mb': avg_memory,
        'max_memory_mb': max_memory,
        'peak_memory_mb': peak_memory,
        'time_span': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
    }

def extract_memory_from_node0_log():
    """Try to extract peak memory from Node 0 output"""
    # This is a placeholder - in practice, you'd need to capture Node 0 output
    # For now, we'll use psutil to get current memory
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except:
        return None

def run_experiment(rate, num_requests=30):
    """Run a single experiment and collect metrics"""
    print(f"\n{'='*70}")
    print(f"Running experiment: {rate} rate ({num_requests} requests)")
    print(f"{'='*70}")
    
    # Clear previous batch log or rename it
    batch_log = f'batch_sizes_{rate}.jsonl'
    if os.path.exists('batch_sizes.jsonl'):
        if os.path.exists(batch_log):
            os.remove(batch_log)
        os.rename('batch_sizes.jsonl', batch_log)
    
    # Set environment variable to use specific log file
    env = os.environ.copy()
    env['NODE_0_IP'] = 'localhost:8000'
    env['BATCH_LOG_FILE'] = 'batch_sizes.jsonl'
    
    # Run the test
    # Calculate timeout based on rate (slow rate needs more time)
    if rate == 'slow':
        timeout = 1200  # 20 minutes for slow rate (10s delay * 30 requests = 5 min + processing overhead)
    elif rate == 'medium':
        timeout = 600   # 10 minutes for medium rate
    else:
        timeout = 300   # 5 minutes for fast/very_fast rates
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ['python3', 'test_profiling.py', 'rate', str(num_requests), rate],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"Warning: Test returned non-zero exit code")
            print(f"Error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"Warning: Test timed out after {timeout/60:.0f} minutes")
        elapsed = timeout
    except Exception as e:
        print(f"Error running test: {e}")
        return None
    
    # Wait a bit for Node 0 to finish processing
    time.sleep(2)
    
    # Parse results
    metrics = parse_batch_log('batch_sizes.jsonl')
    if metrics:
        metrics['rate'] = rate
        metrics['num_requests'] = num_requests
        metrics['experiment_time'] = elapsed
        
        # Try to get memory (this is approximate)
        memory = extract_memory_from_node0_log()
        if memory:
            metrics['memory_mb'] = memory
        
        # Rename log file for this experiment
        if os.path.exists('batch_sizes.jsonl'):
            os.rename('batch_sizes.jsonl', batch_log)
            metrics['log_file'] = batch_log
        
        return metrics
    else:
        print(f"Warning: No batch data found for {rate} rate")
        return None

def print_results_table(results):
    """Print a formatted table of results"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*100)
    print("BATCH SIZE vs THROUGHPUT vs MEMORY USAGE")
    print("="*100)
    print(f"{'Rate':<12} {'Avg Batch':<12} {'Min Batch':<12} {'Max Batch':<12} {'Throughput':<20} {'Avg Memory':<15} {'Peak Memory':<15}")
    print(f"{'':<12} {'Size':<12} {'Size':<12} {'Size':<12} {'(req/min)':<20} {'(MB)':<15} {'(MB)':<15}")
    print("-"*100)
    
    for r in results:
        rate = r.get('rate', 'unknown')
        avg_batch = r.get('avg_batch_size', 0)
        min_batch = r.get('min_batch_size', 0)
        max_batch = r.get('max_batch_size', 0)
        throughput = r.get('throughput_per_min', 0)
        avg_memory = r.get('avg_memory_mb', 0) or 0
        peak_memory = r.get('peak_memory_mb', 0) or 0
        
        print(f"{rate:<12} {avg_batch:<12.1f} {min_batch:<12} {max_batch:<12} {throughput:<20.1f} {avg_memory:<15.1f} {peak_memory:<15.1f}")
    
    print("="*100)
    
    # Print detailed statistics
    print("\n" + "="*100)
    print("DETAILED STATISTICS")
    print("="*100)
    
    for r in results:
        rate = r.get('rate', 'unknown')
        print(f"\n{rate.upper()} RATE:")
        print(f"  Total batches: {r.get('total_batches', 0)}")
        print(f"  Total requests: {r.get('total_requests', 0)}")
        print(f"  Batch sizes: min={r.get('min_batch_size', 0)}, max={r.get('max_batch_size', 0)}, avg={r.get('avg_batch_size', 0):.2f}, median={r.get('median_batch_size', 0):.1f}")
        print(f"  Throughput: {r.get('throughput_per_min', 0):.1f} req/min ({r.get('throughput_per_sec', 0):.2f} req/s)")
        if r.get('avg_memory_mb'):
            print(f"  Average Memory: {r.get('avg_memory_mb', 0):.1f} MB")
        if r.get('peak_memory_mb'):
            print(f"  Peak Memory: {r.get('peak_memory_mb', 0):.1f} MB")
        if r.get('batch_size_distribution'):
            dist = r.get('batch_size_distribution', {})
            print(f"  Distribution: {dict(sorted(dist.items()))}")

def save_results_json(results, filename='batch_metrics_results.json'):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        rates = sys.argv[1:]
    else:
        rates = ['slow', 'medium', 'fast', 'very_fast']
    
    num_requests = int(os.environ.get('NUM_REQUESTS', 30))
    
    print("="*70)
    print("BATCH SIZE vs THROUGHPUT vs MEMORY ANALYSIS")
    print("="*70)
    print(f"This will run experiments at different rates to see how batch size")
    print(f"affects throughput and memory usage.")
    print(f"\nRates to test: {', '.join(rates)}")
    print(f"Requests per experiment: {num_requests}")
    print(f"\nMake sure all 3 nodes are running!")
    print("="*70)
    
    input("Press Enter to start experiments...")
    
    results = []
    for rate in rates:
        metrics = run_experiment(rate, num_requests)
        if metrics:
            results.append(metrics)
        time.sleep(1)  # Brief pause between experiments
    
    if results:
        print_results_table(results)
        save_results_json(results)
        
        # Print summary insights
        print("\n" + "="*100)
        print("KEY INSIGHTS")
        print("="*100)
        
        if len(results) >= 2:
            slow_result = next((r for r in results if r.get('rate') == 'slow'), None)
            fast_result = next((r for r in results if r.get('rate') == 'very_fast'), None)
            
            if slow_result and fast_result:
                batch_increase = (fast_result.get('avg_batch_size', 0) / slow_result.get('avg_batch_size', 1)) - 1
                throughput_increase = (fast_result.get('throughput_per_min', 0) / slow_result.get('throughput_per_min', 1)) - 1
                
                print(f"Batch size increase (slow → very_fast): {batch_increase*100:.1f}%")
                print(f"Throughput increase (slow → very_fast): {throughput_increase*100:.1f}%")
                
                if slow_result.get('avg_memory_mb') and fast_result.get('avg_memory_mb'):
                    memory_increase = (fast_result.get('avg_memory_mb', 0) / slow_result.get('avg_memory_mb', 1)) - 1
                    print(f"Average Memory increase (slow → very_fast): {memory_increase*100:.1f}%")
                if slow_result.get('peak_memory_mb') and fast_result.get('peak_memory_mb'):
                    peak_memory_increase = (fast_result.get('peak_memory_mb', 0) / slow_result.get('peak_memory_mb', 1)) - 1
                    print(f"Peak Memory increase (slow → very_fast): {peak_memory_increase*100:.1f}%")
    else:
        print("\nNo results collected. Check that:")
        print("  1. All 3 nodes are running")
        print("  2. Node 0 is processing batches")
        print("  3. batch_sizes.jsonl is being created")

if __name__ == '__main__':
    from collections import Counter
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user.")
        sys.exit(1)

