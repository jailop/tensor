# GPU and BLAS Support for Tensor Operations

This document describes the GPU (CUDA) and BLAS acceleration support added to the tensor library.

## Overview

The tensor library now supports hardware acceleration through:
1. **CUDA GPU acceleration** - For NVIDIA GPUs using CUDA
2. **BLAS acceleration** - For optimized CPU matrix operations

Both are automatically detected and enabled at compile time if available.

## Build Configuration

### CMake Configuration

The build system automatically detects CUDA and BLAS:

```bash
cd build
cmake ..
make
```

Output messages:
- `CUDA found - GPU acceleration enabled` - CUDA support is available
- `BLAS found - optimized CPU matrix operations enabled` - BLAS support is available
- `CUDA not found - building CPU-only version` - No GPU support
- `BLAS not found - using standard CPU implementation` - No BLAS support

### Preprocessor Flags

- `USE_GPU` - Defined when CUDA is available
- `USE_BLAS` - Defined when BLAS is available

## Supported Operations

### Element-wise Operations (GPU + BLAS)

The following element-wise operations now support GPU acceleration:

#### Tensor-Tensor Operations
- **Addition**: `tensor1 + tensor2`, `tensor1 += tensor2`
- **Subtraction**: `tensor1 - tensor2`, `tensor1 -= tensor2`
- **Multiplication**: `tensor1 * tensor2`, `tensor1 *= tensor2`
- **Division**: `tensor1 / tensor2`, `tensor1 /= tensor2`

#### Tensor-Scalar Operations
- **Addition**: `tensor + scalar`, `tensor += scalar`
- **Subtraction**: `tensor - scalar`, `tensor -= scalar`
- **Multiplication**: `tensor * scalar`, `tensor *= scalar`
- **Division**: `tensor / scalar`, `tensor /= scalar`

### Mathematical Functions (GPU)

The following mathematical functions now support GPU acceleration:

- **Exponential**: `tensor.exp()`
- **Logarithm**: `tensor.log()`
- **Square Root**: `tensor.sqrt()`
- **Power**: `tensor.pow(exponent)`
- **Sine**: `tensor.sin()`
- **Cosine**: `tensor.cos()`
- **Hyperbolic Tangent**: `tensor.tanh()`
- **Sigmoid**: `tensor.sigmoid()` (with autograd support)
- **ReLU**: `tensor.relu()` (with autograd support)

### Reduction Operations (GPU)

- **Sum**: GPU-accelerated reduction with shared memory
- **Mean**: GPU-accelerated average calculation
- **Max**: GPU-accelerated maximum finding
- **Min**: GPU-accelerated minimum finding

### Matrix Operations (GPU + BLAS)

#### Dot Products
- **1D Dot Product**: `tensor1.dot(tensor2)` - Vector dot product
- **2D Matrix Multiplication**: `matrix1.dot(matrix2)` - Uses BLAS (GEMM) on CPU or custom CUDA kernel on GPU
- **N-D Tensor Contraction**: General tensor contraction with GPU support

## Usage Example

```cpp
#include "tensor.h"

// Create tensors with GPU support enabled (default)
Tensor<float, 2> a({1000, 1000}, true);  // use_gpu = true
Tensor<float, 2> b({1000, 1000}, true);

// Fill with random data
fill_random(a);
fill_random(b);

// Element-wise operations (automatically uses GPU if available)
auto c = a + b;        // GPU-accelerated addition
auto d = a * b;        // GPU-accelerated multiplication

// Mathematical functions (automatically uses GPU if available)
auto e = a.exp();      // GPU-accelerated exponential
auto f = a.sigmoid();  // GPU-accelerated sigmoid with autograd

// Matrix multiplication (uses BLAS on CPU or GPU kernel)
auto g = a.dot(b);     // GPU-accelerated or BLAS matrix multiply

// Reduction operations (uses GPU parallel reduction)
float sum_val = a.sum();     // GPU-accelerated sum
float mean_val = a.mean();   // GPU-accelerated mean
```

## Performance Characteristics

### GPU Acceleration

GPU operations are most efficient for:
- Large tensors (> 10,000 elements)
- Batch operations
- Matrix multiplications with large dimensions

GPU operations include memory transfer overhead (host ↔ device), so for small tensors, CPU operations may be faster.

### BLAS Acceleration

BLAS (e.g., OpenBLAS, MKL) provides optimized CPU implementations:
- Highly optimized matrix multiplication (GEMM)
- Efficient dot products
- Parallel execution on CPU cores
- No memory transfer overhead

## Implementation Details

### GPU Kernels

GPU operations are implemented in `src/tensor_gpu.cu`:

1. **Element-wise kernels**: Each thread processes one element
2. **Reduction kernels**: Use shared memory and parallel reduction
3. **Matrix multiplication**: Use 2D thread blocks with tiling

Example kernel:
```cuda
__global__ void add_kernel(const T* a, const T* b, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}
```

### BLAS Integration

BLAS functions are declared in `include/tensor.h`:

```cpp
#ifdef USE_BLAS
extern "C" {
    void cblas_sgemm(...);  // Single precision matrix multiply
    void cblas_dgemm(...);  // Double precision matrix multiply
    float cblas_sdot(...);  // Single precision dot product
    double cblas_ddot(...); // Double precision dot product
}
#endif
```

### Fallback Behavior

When GPU or BLAS is not available:
- Operations fall back to standard C++ implementations
- No functional changes, only performance impact
- All tests pass regardless of acceleration availability

## Benchmarking

Run performance benchmarks:

```bash
cd build
./tensor_perf
```

The benchmark will show performance differences between:
- CPU-only operations
- BLAS-accelerated operations (if available)
- GPU-accelerated operations (if available)

## Future Enhancements

Potential improvements:
1. **Persistent GPU memory** - Keep data on GPU to avoid transfer overhead
2. **Batched operations** - Process multiple tensors in a single kernel launch
3. **cuBLAS integration** - Use NVIDIA's optimized BLAS library on GPU
4. **Tensor cores** - Utilize Tensor Cores on newer GPUs for matrix ops
5. **Multi-GPU support** - Distribute operations across multiple GPUs
6. **Async operations** - Overlap computation with data transfer

## Compatibility

- **CUDA**: Requires CUDA 10.0 or later, compute capability 3.5+
- **BLAS**: Compatible with OpenBLAS, Intel MKL, ATLAS, or reference BLAS
- **Compiler**: GCC 7+, Clang 6+, or MSVC 2017+
- **CMake**: Version 3.14 or later

## Troubleshooting

### CUDA Not Detected
- Ensure CUDA toolkit is installed
- Set `CUDA_PATH` environment variable
- Check that `nvcc` is in PATH

### BLAS Not Found
- Install BLAS library: `sudo apt-get install libblas-dev libopenblas-dev`
- Or use Intel MKL for best CPU performance

### Runtime Errors
- Check GPU memory availability
- Verify CUDA architecture matches your GPU
- Ensure CUDA runtime is installed (not just toolkit)
