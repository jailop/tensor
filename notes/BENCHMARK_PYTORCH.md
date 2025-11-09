# Neural Network Benchmark: tensor4d vs PyTorch

This benchmark compares the performance of neural network layers between `tensor4d.nn` and PyTorch.

## Running the Benchmark

```bash
python3 benchmark_pytorch.py
```

## Requirements

- Python 3.6+
- PyTorch 2.0+
- NumPy
- tensor4d (built from source)

Install PyTorch:
```bash
pip install torch
```

## What is Benchmarked

### 1. **Linear Layers**
- Forward pass: input × weights^T + bias
- Backward pass: gradient computation
- Various sizes: small (32→128), medium (256→512), large (256→1024)

### 2. **Activation Functions**
- ReLU (Rectified Linear Unit)
- Sigmoid
- Softmax
- Various batch sizes: 32, 256, 1024

### 3. **Regularization Layers**
- Dropout (inference mode)
- Batch Normalization

### 4. **Full Networks**
- 2-layer networks with various architectures
- MNIST-like: 784→256→10
- Large: 2048→1024→1000

## Results Interpretation

**Speedup Metric:**
- `Speedup > 1.0`: tensor4d is faster
- `Speedup < 1.0`: PyTorch is faster
- `Speedup = 1.0`: equal performance

### Expected Results

**tensor4d advantages:**
- Simpler operations with small data sizes
- Specific backward pass computations
- Low overhead for small batches

**PyTorch advantages:**
- Matrix multiplications (highly optimized with MKL/OpenBLAS/BLAS)
- Large batch sizes
- Complex activation functions
- Full network inference
- GPU acceleration (not tested here)

## Why PyTorch is Generally Faster

PyTorch benefits from:

1. **Highly Optimized BLAS Libraries**: Uses Intel MKL, OpenBLAS, or Apple Accelerate
2. **CPU Vectorization**: AVX2, AVX512 SIMD instructions
3. **Multithreading**: Parallel computation across CPU cores
4. **Years of Optimization**: Industry-standard deep learning framework
5. **JIT Compilation**: TorchScript for runtime optimization

## tensor4d Goals

tensor4d is designed for:

- **Educational purposes**: Understanding how neural networks work
- **Research prototyping**: Quick experimentation with new layer types
- **Embedded systems**: Small footprint, no heavy dependencies
- **C interoperability**: Easy integration with C/C++ projects
- **Custom operations**: Easy to extend and modify

## Benchmark Methodology

- **Iterations**: 50-100 runs per operation
- **Warmup**: First few iterations excluded from timing
- **Statistics**: Mean ± standard deviation reported
- **Device**: CPU only (both libraries)
- **Precision**: float32
- **No GPU**: Fair comparison on CPU

## Sample Output

```
Linear Layer - Medium
----------------------------------------------------------------------------------------------------
Linear Forward                    | Size: (128,256)→512   | tensor4d:    9.16ms | PyTorch:    0.28ms | Speedup:   0.03x
Linear Backward                   | Size: (128,256)→512   | tensor4d:   16.76ms | PyTorch:    0.61ms | Speedup:   0.04x

Full Networks
----------------------------------------------------------------------------------------------------
2-Layer Network                   | Size: (32,784)→256→10 | tensor4d:    3.84ms | PyTorch:    0.14ms | Speedup:   0.04x
```

## Conclusion

PyTorch is a production-ready, highly optimized deep learning framework that significantly outperforms tensor4d for most neural network operations. This is expected and intentional.

**Use PyTorch when:**
- Training large models
- Production deployment
- GPU acceleration needed
- State-of-the-art performance required

**Use tensor4d when:**
- Learning how neural networks work internally
- Embedded/constrained environments
- Custom research operations
- C/C++ integration needed
- Educational purposes
- Prototyping novel architectures

## Notes

- Results may vary based on:
  - CPU architecture (AVX2/AVX512 support)
  - BLAS library installed
  - System load
  - Compiler optimizations
  
- PyTorch's advantage grows with:
  - Larger matrix sizes
  - Deeper networks
  - More complex operations
  - GPU usage (not tested)

## Future Improvements for tensor4d

Potential optimizations:
1. Use BLAS for matrix multiplication
2. SIMD vectorization for element-wise ops
3. Multithreading for batch processing
4. Kernel fusion for sequential operations
5. Memory pooling to reduce allocations

These optimizations are intentionally not implemented to keep the codebase simple and educational.
