#!/usr/bin/env python3
"""
Benchmark comparison between tensor4d.nn and PyTorch
This script compares the performance of neural network layers between
the tensor4d library and PyTorch.
"""

import time
import sys
sys.path.insert(0, '/home/jailop/shared/guides/transformers/python')

import numpy as np
import tensor4d as t4d
from typing import Callable, Tuple

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
    PYTORCH_AVAILABLE = False
    sys.exit(1)


class BenchmarkResult:
    """Store and display benchmark results"""
    def __init__(self, name: str, tensor4d_time: float, pytorch_time: float, size: str,
                 tensor4d_std: float = 0.0, pytorch_std: float = 0.0):
        self.name = name
        self.tensor4d_time = tensor4d_time
        self.pytorch_time = pytorch_time
        self.tensor4d_std = tensor4d_std
        self.pytorch_std = pytorch_std
        self.size = size
        self.speedup = pytorch_time / tensor4d_time if tensor4d_time > 0 else float('inf')
    
    def __str__(self):
        return (f"{self.name:40s} | Size: {self.size:15s} | "
                f"tensor4d: {self.tensor4d_time*1000:7.2f}±{self.tensor4d_std*1000:5.2f}ms | "
                f"PyTorch: {self.pytorch_time*1000:7.2f}±{self.pytorch_std*1000:5.2f}ms | "
                f"Speedup: {self.speedup:6.2f}x")


def timer(func: Callable, iterations: int = 100) -> Tuple[float, float]:
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


def benchmark_linear_forward(batch_size: int, in_features: int, out_features: int, 
                             iterations: int = 100) -> BenchmarkResult:
    """Benchmark Linear layer forward pass"""
    size_str = f"({batch_size},{in_features})→{out_features}"
    
    # Create input data
    input_data = np.random.randn(batch_size, in_features).astype(np.float32)
    
    # tensor4d
    t4d_layer = t4d.nn.Linearf(in_features, out_features, True)
    t4d_input = t4d.Matrixf(input_data.tolist())
    t4d_time, t4d_std = timer(lambda: t4d_layer.forward(t4d_input), iterations)
    
    # PyTorch
    torch_layer = nn.Linear(in_features, out_features)
    torch_input = torch.from_numpy(input_data)
    with torch.no_grad():
        torch_time, torch_std = timer(lambda: torch_layer(torch_input), iterations)
    
    return BenchmarkResult("Linear Forward", t4d_time, torch_time, size_str, t4d_std, torch_std)


def benchmark_linear_backward(batch_size: int, in_features: int, out_features: int, 
                              iterations: int = 50) -> BenchmarkResult:
    """Benchmark Linear layer backward pass"""
    size_str = f"({batch_size},{in_features})→{out_features}"
    
    # Create input and gradient data
    input_data = np.random.randn(batch_size, in_features).astype(np.float32)
    grad_data = np.random.randn(batch_size, out_features).astype(np.float32)
    
    # tensor4d
    t4d_layer = t4d.nn.Linearf(in_features, out_features, True)
    t4d_input = t4d.Matrixf(input_data.tolist())
    t4d_grad = t4d.Matrixf(grad_data.tolist())
    t4d_output = t4d_layer.forward(t4d_input)
    t4d_time, t4d_std = timer(lambda: t4d_layer.backward(t4d_grad), iterations)
    
    # PyTorch
    torch_layer = nn.Linear(in_features, out_features)
    torch_input = torch.from_numpy(input_data).requires_grad_(True)
    torch_output = torch_layer(torch_input)
    torch_grad = torch.from_numpy(grad_data)
    
    def pytorch_backward():
        if torch_input.grad is not None:
            torch_input.grad.zero_()
        torch_output.backward(torch_grad, retain_graph=True)
    
    torch_time, torch_std = timer(pytorch_backward, iterations)
    
    return BenchmarkResult("Linear Backward", t4d_time, torch_time, size_str, t4d_std, torch_std)


def benchmark_relu_forward(batch_size: int, features: int, iterations: int = 100) -> BenchmarkResult:
    """Benchmark ReLU activation forward pass"""
    size_str = f"({batch_size},{features})"
    
    # Create input data with mix of positive and negative
    input_data = np.random.randn(batch_size, features).astype(np.float32)
    
    # tensor4d
    t4d_layer = t4d.nn.ReLUf()
    t4d_input = t4d.Matrixf(input_data.tolist())
    t4d_time, t4d_std = timer(lambda: t4d_layer.forward(t4d_input), iterations)
    
    # PyTorch
    torch_layer = nn.ReLU()
    torch_input = torch.from_numpy(input_data)
    with torch.no_grad():
        torch_time, torch_std = timer(lambda: torch_layer(torch_input), iterations)
    
    return BenchmarkResult("ReLU Forward", t4d_time, torch_time, size_str, t4d_std, torch_std)


def benchmark_sigmoid_forward(batch_size: int, features: int, iterations: int = 100) -> BenchmarkResult:
    """Benchmark Sigmoid activation forward pass"""
    size_str = f"({batch_size},{features})"
    
    input_data = np.random.randn(batch_size, features).astype(np.float32)
    
    # tensor4d
    t4d_layer = t4d.nn.Sigmoidf()
    t4d_input = t4d.Matrixf(input_data.tolist())
    t4d_time, t4d_std = timer(lambda: t4d_layer.forward(t4d_input), iterations)
    
    # PyTorch
    torch_layer = nn.Sigmoid()
    torch_input = torch.from_numpy(input_data)
    with torch.no_grad():
        torch_time, torch_std = timer(lambda: torch_layer(torch_input), iterations)
    
    return BenchmarkResult("Sigmoid Forward", t4d_time, torch_time, size_str, t4d_std, torch_std)


def benchmark_softmax_forward(batch_size: int, features: int, iterations: int = 100) -> BenchmarkResult:
    """Benchmark Softmax activation forward pass"""
    size_str = f"({batch_size},{features})"
    
    input_data = np.random.randn(batch_size, features).astype(np.float32)
    
    # tensor4d
    t4d_layer = t4d.nn.Softmaxf()
    t4d_input = t4d.Matrixf(input_data.tolist())
    t4d_time, t4d_std = timer(lambda: t4d_layer.forward(t4d_input), iterations)
    
    # PyTorch
    torch_layer = nn.Softmax(dim=1)
    torch_input = torch.from_numpy(input_data)
    with torch.no_grad():
        torch_time, torch_std = timer(lambda: torch_layer(torch_input), iterations)
    
    return BenchmarkResult("Softmax Forward", t4d_time, torch_time, size_str, t4d_std, torch_std)


def benchmark_dropout_forward(batch_size: int, features: int, p: float = 0.5, 
                              iterations: int = 100) -> BenchmarkResult:
    """Benchmark Dropout layer forward pass (inference mode)"""
    size_str = f"({batch_size},{features}), p={p}"
    
    input_data = np.random.randn(batch_size, features).astype(np.float32)
    
    # tensor4d (inference mode)
    t4d_layer = t4d.nn.Dropoutf(p)
    t4d_layer.train(False)
    t4d_input = t4d.Matrixf(input_data.tolist())
    t4d_time, t4d_std = timer(lambda: t4d_layer.forward(t4d_input), iterations)
    
    # PyTorch (eval mode)
    torch_layer = nn.Dropout(p)
    torch_layer.eval()
    torch_input = torch.from_numpy(input_data)
    with torch.no_grad():
        torch_time, torch_std = timer(lambda: torch_layer(torch_input), iterations)
    
    return BenchmarkResult("Dropout Inference", t4d_time, torch_time, size_str, t4d_std, torch_std)


def benchmark_batchnorm_forward(batch_size: int, features: int, iterations: int = 100) -> BenchmarkResult:
    """Benchmark BatchNorm layer forward pass"""
    size_str = f"({batch_size},{features})"
    
    input_data = np.random.randn(batch_size, features).astype(np.float32)
    
    # tensor4d
    t4d_layer = t4d.nn.BatchNorm1df(features, 1e-5, 0.1)
    t4d_input = t4d.Matrixf(input_data.tolist())
    t4d_time, t4d_std = timer(lambda: t4d_layer.forward(t4d_input), iterations)
    
    # PyTorch
    torch_layer = nn.BatchNorm1d(features)
    torch_input = torch.from_numpy(input_data)
    with torch.no_grad():
        torch_time, torch_std = timer(lambda: torch_layer(torch_input), iterations)
    
    return BenchmarkResult("BatchNorm Forward", t4d_time, torch_time, size_str, t4d_std, torch_std)


def benchmark_network_forward(batch_size: int, input_dim: int, hidden_dim: int, output_dim: int,
                              iterations: int = 50) -> BenchmarkResult:
    """Benchmark a simple 2-layer network forward pass"""
    size_str = f"({batch_size},{input_dim})→{hidden_dim}→{output_dim}"
    
    input_data = np.random.randn(batch_size, input_dim).astype(np.float32)
    
    # tensor4d
    t4d_fc1 = t4d.nn.Linearf(input_dim, hidden_dim, True)
    t4d_relu = t4d.nn.ReLUf()
    t4d_fc2 = t4d.nn.Linearf(hidden_dim, output_dim, True)
    t4d_softmax = t4d.nn.Softmaxf()
    t4d_input = t4d.Matrixf(input_data.tolist())
    
    def t4d_forward():
        h1 = t4d_fc1.forward(t4d_input)
        h2 = t4d_relu.forward(h1)
        h3 = t4d_fc2.forward(h2)
        output = t4d_softmax.forward(h3)
        return output
    
    t4d_time, t4d_std = timer(t4d_forward, iterations)
    
    # PyTorch
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x
    
    torch_net = SimpleNet()
    torch_input = torch.from_numpy(input_data)
    
    with torch.no_grad():
        torch_time, torch_std = timer(lambda: torch_net(torch_input), iterations)
    
    return BenchmarkResult("2-Layer Network", t4d_time, torch_time, size_str, t4d_std, torch_std)


def print_header():
    """Print benchmark header"""
    print("\n" + "="*100)
    print(" " * 25 + "tensor4d.nn vs PyTorch Benchmark")
    print("="*100)
    print()
    print(f"PyTorch version: {torch.__version__}")
    print(f"tensor4d version: {t4d.__version__}")
    print(f"Device: CPU")
    print()


def print_section(title: str):
    """Print section header"""
    print(f"\n{title}")
    print("-" * 100)


def main():
    """Run all benchmarks"""
    if not PYTORCH_AVAILABLE:
        return
    
    print_header()
    
    results = []
    
    # Linear layer benchmarks
    print_section("Linear Layer - Small")
    results.append(benchmark_linear_forward(32, 64, 128))
    results.append(benchmark_linear_forward(64, 128, 256))
    results.append(benchmark_linear_backward(32, 64, 128))
    results.append(benchmark_linear_backward(64, 128, 256))
    
    for result in results[-4:]:
        print(result)
    
    # Linear layer - Medium
    print_section("Linear Layer - Medium")
    results.append(benchmark_linear_forward(128, 256, 512))
    results.append(benchmark_linear_forward(256, 512, 1024))
    results.append(benchmark_linear_backward(128, 256, 512))
    results.append(benchmark_linear_backward(256, 512, 1024))
    
    for result in results[-4:]:
        print(result)
    
    # Activation functions - Small
    print_section("Activation Functions - Small")
    results.append(benchmark_relu_forward(32, 128))
    results.append(benchmark_sigmoid_forward(32, 128))
    results.append(benchmark_softmax_forward(32, 10))
    
    for result in results[-3:]:
        print(result)
    
    # Activation functions - Medium
    print_section("Activation Functions - Medium")
    results.append(benchmark_relu_forward(256, 512))
    results.append(benchmark_sigmoid_forward(256, 512))
    results.append(benchmark_softmax_forward(256, 100))
    
    for result in results[-3:]:
        print(result)
    
    # Activation functions - Large
    print_section("Activation Functions - Large")
    results.append(benchmark_relu_forward(1024, 2048))
    results.append(benchmark_sigmoid_forward(1024, 2048))
    results.append(benchmark_softmax_forward(1024, 1000))
    
    for result in results[-3:]:
        print(result)
    
    # Regularization layers
    print_section("Regularization Layers")
    results.append(benchmark_dropout_forward(32, 128, 0.5))
    results.append(benchmark_dropout_forward(256, 512, 0.5))
    results.append(benchmark_batchnorm_forward(32, 128))
    results.append(benchmark_batchnorm_forward(256, 512))
    
    for result in results[-4:]:
        print(result)
    
    # Full network benchmarks
    print_section("Full Networks")
    results.append(benchmark_network_forward(32, 784, 256, 10))  # MNIST-like
    results.append(benchmark_network_forward(64, 1024, 512, 100))
    results.append(benchmark_network_forward(128, 2048, 1024, 1000))
    
    for result in results[-3:]:
        print(result)
    
    # Summary
    print_section("Summary Statistics")
    avg_speedup = sum(r.speedup for r in results) / len(results)
    min_speedup = min(r.speedup for r in results)
    max_speedup = max(r.speedup for r in results)
    
    faster_count = sum(1 for r in results if r.speedup > 1.0)
    slower_count = sum(1 for r in results if r.speedup < 1.0)
    
    print(f"Total benchmarks: {len(results)}")
    print(f"tensor4d.nn faster: {faster_count} ({100*faster_count/len(results):.1f}%)")
    print(f"PyTorch faster: {slower_count} ({100*slower_count/len(results):.1f}%)")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Min speedup: {min_speedup:.2f}x")
    print(f"Max speedup: {max_speedup:.2f}x")
    
    # Best and worst cases
    best = max(results, key=lambda r: r.speedup)
    worst = min(results, key=lambda r: r.speedup)
    
    print(f"\nBest performance (tensor4d): {best.name} - {best.speedup:.2f}x faster")
    print(f"Worst performance (tensor4d): {worst.name} - {worst.speedup:.2f}x {'faster' if worst.speedup > 1 else 'slower'}")
    
    print("\n" + "="*100)
    print("Note: Speedup > 1.0 means tensor4d is faster; < 1.0 means PyTorch is faster")
    print("      PyTorch is highly optimized and uses MKL/OpenBLAS for CPU operations")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
