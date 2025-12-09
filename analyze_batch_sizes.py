#!/usr/bin/env python3
"""
Script to analyze batch sizes from batch_sizes.jsonl file
Shows statistics and distribution of batch sizes
"""

import json
import sys
from collections import Counter
from statistics import mean, median, stdev

def analyze_batch_sizes(log_file='batch_sizes.jsonl'):
    """Analyze batch sizes from log file"""
    batch_sizes = []
    queue_depths = []
    optimal_batch_sizes = []
    timestamps = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    batch_sizes.append(entry['batch_size'])
                    queue_depths.append(entry.get('queue_depth', 0))
                    optimal_batch_sizes.append(entry.get('optimal_batch_size', 0))
                    timestamps.append(entry['timestamp'])
    except FileNotFoundError:
        print(f"Error: File {log_file} not found.")
        print("Make sure Node 0 has processed some batches.")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing {log_file}: {e}")
        return
    
    if not batch_sizes:
        print(f"No batch data found in {log_file}")
        return
    
    # Calculate statistics
    print("="*70)
    print("BATCH SIZE ANALYSIS")
    print("="*70)
    print(f"Total batches processed: {len(batch_sizes)}")
    print(f"Total requests: {sum(batch_sizes)}")
    print()
    print("Batch Size Statistics:")
    print(f"  Min: {min(batch_sizes)}")
    print(f"  Max: {max(batch_sizes)}")
    print(f"  Mean: {mean(batch_sizes):.2f}")
    print(f"  Median: {median(batch_sizes):.2f}")
    if len(batch_sizes) > 1:
        print(f"  Std Dev: {stdev(batch_sizes):.2f}")
    print()
    
    # Distribution
    distribution = Counter(batch_sizes)
    print("Batch Size Distribution:")
    for size in sorted(distribution.keys()):
        count = distribution[size]
        percentage = (count / len(batch_sizes)) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {size:2d}: {count:3d} batches ({percentage:5.1f}%) {bar}")
    print()
    
    # Queue depth vs batch size correlation
    if queue_depths:
        print("Queue Depth Statistics:")
        print(f"  Min: {min(queue_depths)}")
        print(f"  Max: {max(queue_depths)}")
        print(f"  Mean: {mean(queue_depths):.2f}")
        print()
    
    # Optimal vs actual batch size
    if optimal_batch_sizes and all(opt > 0 for opt in optimal_batch_sizes):
        matches = sum(1 for i in range(len(batch_sizes)) 
                     if batch_sizes[i] == optimal_batch_sizes[i])
        print(f"Optimal batch size matches actual: {matches}/{len(batch_sizes)} ({matches/len(batch_sizes)*100:.1f}%)")
        print()
    
    # Time analysis
    if len(timestamps) > 1:
        time_span = max(timestamps) - min(timestamps)
        print(f"Time span: {time_span:.1f} seconds ({time_span/60:.1f} minutes)")
        if time_span > 0:
            avg_batches_per_sec = len(batch_sizes) / time_span
            print(f"Average batch rate: {avg_batches_per_sec:.2f} batches/second")
        print()
    
    print("="*70)
    print(f"Data file: {log_file}")
    print("="*70)

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'batch_sizes.jsonl'
    analyze_batch_sizes(log_file)

