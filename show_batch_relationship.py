#!/usr/bin/env python3
"""
Simple script to show the relationship between batch size, throughput, and memory
from existing batch_sizes.jsonl files
"""

import json
import os
import sys
from collections import defaultdict
from statistics import mean

def analyze_batch_relationship(log_file='batch_sizes.jsonl'):
    """Analyze relationship between batch size, throughput, and memory"""
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found")
        print("Run some experiments first to generate batch data.")
        return
    
    batches = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                batches.append(json.loads(line))
    
    if not batches:
        print(f"No data in {log_file}")
        return
    
    # Group by batch size
    by_batch_size = defaultdict(list)
    for b in batches:
        batch_size = b.get('batch_size', 0)
        if 'throughput_per_min' in b and 'memory_mb' in b:
            by_batch_size[batch_size].append({
                'throughput': b.get('throughput_per_min', 0),
                'memory': b.get('memory_mb', 0),
                'peak_memory': b.get('peak_memory_mb', 0)
            })
    
    if not by_batch_size:
        print("No throughput/memory data found in log file.")
        print("Make sure Node 0 has processed batches with the updated logging.")
        return
    
    # Calculate averages per batch size
    results = []
    for batch_size in sorted(by_batch_size.keys()):
        data = by_batch_size[batch_size]
        results.append({
            'batch_size': batch_size,
            'count': len(data),
            'avg_throughput': mean([d['throughput'] for d in data]),
            'avg_memory': mean([d['memory'] for d in data]),
            'max_peak_memory': max([d.get('peak_memory', d['memory']) for d in data]) if data else 0
        })
    
    # Print table
    print("\n" + "="*90)
    print("BATCH SIZE vs THROUGHPUT vs MEMORY USAGE")
    print("="*90)
    print(f"{'Batch Size':<12} {'Count':<10} {'Avg Throughput':<20} {'Avg Memory':<15} {'Max Peak Memory':<15}")
    print(f"{'':<12} {'':<10} {'(req/min)':<20} {'(MB)':<15} {'(MB)':<15}")
    print("-"*90)
    
    for r in results:
        print(f"{r['batch_size']:<12} {r['count']:<10} {r['avg_throughput']:<20.1f} {r['avg_memory']:<15.1f} {r['max_peak_memory']:<15.1f}")
    
    print("="*90)
    
    # Print insights
    if len(results) >= 2:
        print("\nKEY INSIGHTS:")
        print("-"*90)
        
        smallest = results[0]
        largest = results[-1]
        
        batch_increase = (largest['batch_size'] / smallest['batch_size']) - 1
        throughput_increase = (largest['avg_throughput'] / smallest['avg_throughput']) - 1 if smallest['avg_throughput'] > 0 else 0
        memory_increase = (largest['avg_memory'] / smallest['avg_memory']) - 1 if smallest['avg_memory'] > 0 else 0
        
        print(f"Batch size range: {smallest['batch_size']} → {largest['batch_size']} ({batch_increase*100:.1f}% increase)")
        print(f"Throughput range: {smallest['avg_throughput']:.1f} → {largest['avg_throughput']:.1f} req/min ({throughput_increase*100:.1f}% increase)")
        print(f"Memory range: {smallest['avg_memory']:.1f} → {largest['avg_memory']:.1f} MB ({memory_increase*100:.1f}% increase)")
        
        if throughput_increase > 0 and memory_increase > 0:
            efficiency = throughput_increase / memory_increase
            print(f"\nEfficiency ratio (throughput gain / memory cost): {efficiency:.2f}")
            if efficiency > 1:
                print("  → Throughput increases faster than memory (good!)")
            else:
                print("  → Memory increases faster than throughput (trade-off)")

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'batch_sizes.jsonl'
    analyze_batch_relationship(log_file)

