#!/usr/bin/env python3
"""
Benchmark comparison between tensor4d and NumPy

This script compares the performance of various operations between
the tensor4d library and NumPy to demonstrate computational efficiency.
"""

import time
import numpy as np
import tensor4d as t4d
from typing import Callable, Tuple


class BenchmarkResult:
    """Store and display benchmark results"""
    def __init__(self, name: str, tensor4d_time: float, numpy_time: float, size: str,
                 tensor4d_std: float = 0.0, numpy_std: float = 0.0):
        self.name = name
        self.tensor4d_time = tensor4d_time
        self.numpy_time = numpy_time
        self.tensor4d_std = tensor4d_std
        self.numpy_std = numpy_std
        self.size = size
        self.speedup = numpy_time / tensor4d_time if tensor4d_time > 0 else float('inf')
    
    def __str__(self):
        return (f"{self.name:40s} | Size: {self.size:15s} | "
                f"tensor4d: {self.tensor4d_time*10000:7.2f}±{self.tensor4d_std*10000:5.2f}ms | "
                f"NumPy: {self.numpy_time*10000:7.2f}±{self.numpy_std*10000:5.2f}ms | "
                f"Speedup: {self.speedup:6.2f}x")


def timer(func: Callable, iterations: int = 1000) -> Tuple[float, float]:
    """Time a function over multiple iterations and return (mean, stddev)"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time


def create_t4d_from_numpy(np_array):
    """Helper to create tensor4d from numpy"""
    shape = np_array.shape
    if len(shape) == 1:
        t = t4d.Vectorf(list(shape))
    elif len(shape) == 2:
        t = t4d.Matrixf(list(shape))
    elif len(shape) == 3:
        t = t4d.Tensor3f(list(shape))
    else:
        t = t4d.Tensor4f(list(shape))
    return t.from_numpy(np_array)


def benchmark_creation(shape: Tuple[int, ...], iterations: int = 1000) -> BenchmarkResult:
    """Benchmark tensor/array creation"""
    size_str = 'x'.join(map(str, shape))
    
    # tensor4d
    if len(shape) == 1:
        t4d_time, t4d_std = timer(lambda: t4d.Vectorf(list(shape)), iterations)
    elif len(shape) == 2:
        t4d_time, t4d_std = timer(lambda: t4d.Matrixf(list(shape)), iterations)
    elif len(shape) == 3:
        t4d_time, t4d_std = timer(lambda: t4d.Tensor3f(list(shape)), iterations)
    else:
        t4d_time, t4d_std = timer(lambda: t4d.Tensor4f(list(shape)), iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: np.zeros(shape, dtype=np.float32), iterations)
    
    return BenchmarkResult("Creation (zeros)", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_from_numpy(shape: Tuple[int, ...], iterations: int = 1000) -> BenchmarkResult:
    """Benchmark conversion from numpy to tensor4d"""
    size_str = 'x'.join(map(str, shape))
    
    # Create numpy array once
    np_array = np.random.randn(*shape).astype(np.float32)
    
    # tensor4d conversion
    t4d_time, t4d_std = timer(lambda: create_t4d_from_numpy(np_array), iterations)
    
    # NumPy (just copy for comparison)
    np_time, np_std = timer(lambda: np_array.copy(), iterations)
    
    return BenchmarkResult("From NumPy conversion", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_elementwise_add(shape: Tuple[int, ...], iterations: int = 1000) -> BenchmarkResult:
    """Benchmark element-wise addition"""
    size_str = 'x'.join(map(str, shape))
    
    # Create numpy arrays
    a_np = np.random.randn(*shape).astype(np.float32)
    b_np = np.random.randn(*shape).astype(np.float32)
    
    # tensor4d
    a_t4d = create_t4d_from_numpy(a_np)
    b_t4d = create_t4d_from_numpy(b_np)
    t4d_time, t4d_std = timer(lambda: a_t4d + b_t4d, iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: a_np + b_np, iterations)
    
    return BenchmarkResult("Element-wise Add", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_elementwise_mul(shape: Tuple[int, ...], iterations: int = 1000) -> BenchmarkResult:
    """Benchmark element-wise multiplication"""
    size_str = 'x'.join(map(str, shape))
    
    # Create numpy arrays
    a_np = np.random.randn(*shape).astype(np.float32)
    b_np = np.random.randn(*shape).astype(np.float32)
    
    # tensor4d
    a_t4d = create_t4d_from_numpy(a_np)
    b_t4d = create_t4d_from_numpy(b_np)
    t4d_time, t4d_std = timer(lambda: a_t4d * b_t4d, iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: a_np * b_np, iterations)
    
    return BenchmarkResult("Element-wise Mul", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_matmul(m: int, n: int, k: int, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark matrix multiplication (m x n) @ (n x k)"""
    size_str = f"{m}x{n} @ {n}x{k}"
    
    # Create numpy arrays
    a_np = np.random.randn(m, n).astype(np.float32)
    b_np = np.random.randn(n, k).astype(np.float32)
    
    # tensor4d
    a_t4d = create_t4d_from_numpy(a_np)
    b_t4d = create_t4d_from_numpy(b_np)
    t4d_time, t4d_std = timer(lambda: a_t4d.matmul(b_t4d), iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: np.matmul(a_np, b_np), iterations)
    
    return BenchmarkResult("Matrix Multiplication", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_sum(shape: Tuple[int, ...], iterations: int = 1000) -> BenchmarkResult:
    """Benchmark sum reduction"""
    size_str = 'x'.join(map(str, shape))
    
    # Create numpy array
    a_np = np.random.randn(*shape).astype(np.float32)
    
    # tensor4d
    a_t4d = create_t4d_from_numpy(a_np)
    t4d_time, t4d_std = timer(lambda: a_t4d.sum(), iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: np.sum(a_np), iterations)
    
    return BenchmarkResult("Sum Reduction", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_mean(shape: Tuple[int, ...], iterations: int = 1000) -> BenchmarkResult:
    """Benchmark mean reduction"""
    size_str = 'x'.join(map(str, shape))
    
    # Create numpy array
    a_np = np.random.randn(*shape).astype(np.float32)
    
    # tensor4d
    a_t4d = create_t4d_from_numpy(a_np)
    t4d_time, t4d_std = timer(lambda: a_t4d.mean(), iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: np.mean(a_np), iterations)
    
    return BenchmarkResult("Mean Reduction", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_exp(shape: Tuple[int, ...], iterations: int = 1000) -> BenchmarkResult:
    """Benchmark exponential function"""
    size_str = 'x'.join(map(str, shape))
    
    # Create numpy array
    a_np = np.random.randn(*shape).astype(np.float32)
    
    # tensor4d
    a_t4d = create_t4d_from_numpy(a_np)
    t4d_time, t4d_std = timer(lambda: a_t4d.exp(), iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: np.exp(a_np), iterations)
    
    return BenchmarkResult("Exponential (exp)", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_transpose(shape: Tuple[int, int], iterations: int = 1000) -> BenchmarkResult:
    """Benchmark matrix transpose"""
    size_str = f"{shape[0]}x{shape[1]}"
    
    # Create numpy array
    a_np = np.random.randn(*shape).astype(np.float32)
    
    # tensor4d
    a_t4d = create_t4d_from_numpy(a_np)
    t4d_time, t4d_std = timer(lambda: a_t4d.transpose(), iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: np.transpose(a_np), iterations)
    
    return BenchmarkResult("Transpose", t4d_time, np_time, size_str, t4d_std, np_std)


def benchmark_reshape(original_shape: Tuple[int, ...], new_shape: Tuple[int, ...], 
                      iterations: int = 1000) -> BenchmarkResult:
    """Benchmark reshape operation"""
    size_str = f"{original_shape} -> {new_shape}"
    
    # Create numpy array
    a_np = np.random.randn(*original_shape).astype(np.float32)
    
    # tensor4d
    a_t4d = create_t4d_from_numpy(a_np)
    t4d_time, t4d_std = timer(lambda: a_t4d.reshape(list(new_shape)), iterations)
    
    # NumPy
    np_time, np_std = timer(lambda: np.reshape(a_np, new_shape), iterations)
    
    return BenchmarkResult("Reshape", t4d_time, np_time, str(size_str), t4d_std, np_std)


def print_header():
    """Print benchmark header"""
    print("\n" + "="*1000)
    print(" " * 30 + "tensor4d vs NumPy Benchmark")
    print("="*1000)
    print()


def print_section(title: str):
    """Print section header"""
    print(f"\n{title}")
    print("-" * 1000)


def main():
    """Run all benchmarks"""
    print_header()
    
    results = []
    
    # Small vectors/arrays
    print_section("Small Vectors (1D)")
    results.append(benchmark_creation((10000,)))
    results.append(benchmark_from_numpy((10000,)))
    results.append(benchmark_elementwise_add((10000,)))
    results.append(benchmark_elementwise_mul((10000,)))
    results.append(benchmark_sum((10000,)))
    results.append(benchmark_mean((10000,)))
    results.append(benchmark_exp((10000,)))
    
    for result in results[-7:]:
        print(result)
    
    # Medium vectors/arrays
    print_section("Medium Vectors (1D)")
    results.append(benchmark_creation((1000000,)))
    results.append(benchmark_from_numpy((1000000,)))
    results.append(benchmark_elementwise_add((1000000,)))
    results.append(benchmark_elementwise_mul((1000000,)))
    results.append(benchmark_sum((1000000,)))
    results.append(benchmark_mean((1000000,)))
    results.append(benchmark_exp((1000000,)))
    
    for result in results[-7:]:
        print(result)
    
    # Small matrices
    print_section("Small Matrices (2D)")
    results.append(benchmark_creation((1000, 1000)))
    results.append(benchmark_from_numpy((1000, 1000)))
    results.append(benchmark_elementwise_add((1000, 1000)))
    results.append(benchmark_elementwise_mul((1000, 1000)))
    results.append(benchmark_sum((1000, 1000)))
    results.append(benchmark_mean((1000, 1000)))
    results.append(benchmark_exp((1000, 1000)))
    results.append(benchmark_transpose((1000, 1000)))
    
    for result in results[-8:]:
        print(result)
    
    # Medium matrices
    print_section("Medium Matrices (2D)")
    results.append(benchmark_creation((500, 500)))
    results.append(benchmark_from_numpy((500, 500)))
    results.append(benchmark_elementwise_add((500, 500)))
    results.append(benchmark_elementwise_mul((500, 500)))
    results.append(benchmark_sum((500, 500)))
    results.append(benchmark_mean((500, 500)))
    results.append(benchmark_exp((500, 500)))
    results.append(benchmark_transpose((500, 500)))
    
    for result in results[-8:]:
        print(result)
    
    # Matrix multiplication
    print_section("Matrix Multiplication")
    results.append(benchmark_matmul(64, 64, 64))
    results.append(benchmark_matmul(128, 128, 128))
    results.append(benchmark_matmul(10006, 256, 256))
    results.append(benchmark_matmul(512, 512, 512))
    results.append(benchmark_matmul(1000, 10000, 1000))
    results.append(benchmark_matmul(10000, 1000, 1000))
    
    for result in results[-6:]:
        print(result)
    
    # Higher-dimensional tensors
    print_section("3D Tensors")
    results.append(benchmark_creation((50, 50, 50)))
    results.append(benchmark_from_numpy((50, 50, 50)))
    results.append(benchmark_elementwise_add((50, 50, 50)))
    results.append(benchmark_elementwise_mul((50, 50, 50)))
    results.append(benchmark_sum((50, 50, 50)))
    results.append(benchmark_mean((50, 50, 50)))
    results.append(benchmark_exp((50, 50, 50)))
    
    for result in results[-7:]:
        print(result)
    
    # Summary
    print_section("Summary Statistics")
    avg_speedup = sum(r.speedup for r in results) / len(results)
    min_speedup = min(r.speedup for r in results)
    max_speedup = max(r.speedup for r in results)
    
    faster_count = sum(1 for r in results if r.speedup > 1.0)
    slower_count = sum(1 for r in results if r.speedup < 1.0)
    
    print(f"Total benchmarks: {len(results)}")
    print(f"tensor4d faster: {faster_count} ({1000*faster_count/len(results):.1f}%)")
    print(f"NumPy faster: {slower_count} ({1000*slower_count/len(results):.1f}%)")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Min speedup: {min_speedup:.2f}x")
    print(f"Max speedup: {max_speedup:.2f}x")
    
    print("\n" + "="*1000)
    print("Note: Speedup > 1.0 means tensor4d is faster; < 1.0 means NumPy is faster")
    print("="*1000 + "\n")


if __name__ == "__main__":
    main()
