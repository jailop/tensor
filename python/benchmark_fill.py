#!/usr/bin/env python3
"""
Benchmark fill() operation with GPU acceleration
"""

import sys
sys.path.insert(0, '/home/jailop/devel/datainquiry/tensor/python')

import tensor4d as t4d
import numpy as np
import time

def benchmark_fill_cpu(size, iterations=100):
    """Benchmark CPU fill operation"""
    tensor = t4d.Matrixf([[0.0] * size for _ in range(size)])
    
    start = time.time()
    for _ in range(iterations):
        tensor.fill(3.14)
    end = time.time()
    
    return (end - start) / iterations

def benchmark_fill_gpu(size, iterations=100):
    """Benchmark GPU fill operation (if available)"""
    # Create GPU tensor
    tensor = t4d.Matrixf([[0.0] * size for _ in range(size)])
    # Note: Assuming tensor uses GPU automatically if available
    
    # Warm-up
    tensor.fill(1.0)
    
    start = time.time()
    for _ in range(iterations):
        tensor.fill(3.14)
    end = time.time()
    
    return (end - start) / iterations

def benchmark_numpy_fill(size, iterations=100):
    """Benchmark NumPy fill for comparison"""
    arr = np.zeros((size, size), dtype=np.float32)
    
    start = time.time()
    for _ in range(iterations):
        arr.fill(3.14)
    end = time.time()
    
    return (end - start) / iterations

def benchmark_batch_reuse():
    """Benchmark the batch tensor reuse pattern"""
    print("\n=== Batch Tensor Reuse Pattern ===")
    
    batch_size = 32
    num_classes = 10
    num_batches = 1000
    
    # Method 1: Create new tensor each batch (OLD WAY)
    start = time.time()
    for _ in range(num_batches):
        targets = t4d.Matrixf([[0.0] * num_classes for _ in range(batch_size)])
        # Simulate filling with one-hot
        for i in range(batch_size):
            t4d.nn.label_to_onehot_f(i % num_classes, targets, i, num_classes)
    time_old = time.time() - start
    
    # Method 2: Reuse tensor with fill (NEW WAY)
    targets = t4d.Matrixf([[0.0] * num_classes for _ in range(batch_size)])
    start = time.time()
    for _ in range(num_batches):
        targets.fill(0.0)
        # Simulate filling with one-hot
        for i in range(batch_size):
            t4d.nn.label_to_onehot_f(i % num_classes, targets, i, num_classes)
    time_new = time.time() - start
    
    print(f"Old way (create each batch): {time_old:.4f}s")
    print(f"New way (reuse with fill):   {time_new:.4f}s")
    print(f"Speedup: {time_old/time_new:.2f}x")
    print(f"Time saved: {(time_old - time_new)*1000:.1f}ms")

def main():
    print("=== Fill Operation Benchmark ===\n")
    print(f"tensor4d version: {t4d.__version__}")
    
    sizes = [100, 500, 1000, 2000]
    iterations = 50
    
    print(f"\nBenchmarking fill() with {iterations} iterations per size")
    print("-" * 60)
    print(f"{'Size':<10} {'tensor4d (ms)':<15} {'NumPy (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        t4d_time = benchmark_fill_gpu(size, iterations) * 1000
        numpy_time = benchmark_numpy_fill(size, iterations) * 1000
        speedup = numpy_time / t4d_time if t4d_time > 0 else 0
        
        print(f"{size:<10} {t4d_time:<15.4f} {numpy_time:<15.4f} {speedup:<10.2f}x")
    
    # Benchmark batch reuse pattern
    benchmark_batch_reuse()
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()
