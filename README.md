# Tensor Library

**Disclaimer**: This is a personal project to learn about numerical
computing and optimization techniques. It is only intended to be used
for educative purposes and not at all for production use.

## Overview

A multi-dimensional tensor library for scientific computing and machine
learning, written in C++ with an in-progress C/Python API. It features
automatic GPU acceleration, optimized CPU operations, and support for
building basic neural networks.

- [User Guide](https://github.com/jailop/tensor/blob/main/userguide/00-index.md)
- [API Documentation](https://jailop.github.io/tensor/html)

## Acknowledgments

This project was created to explore:

- High-performance numerical computing
- GPU programming with CUDA
- Template metaprogramming in C++
- Neural network implementation from scratch
- Library design and API ergonomics

### Key Features

#### Core Tensor Operations

- Multi-dimensional tensor operations with arbitrary dimensions (1D to 4D and beyond)
- Type-safe template-based design with compile-time dimension checking
- Efficient memory management with move semantics and smart pointers
- Tensor views and slicing for memory-efficient operations
- Broadcasting for element-wise operations with shape compatibility
- Tensor reshaping, transposition, and dimension manipulation
- Element-wise arithmetic operations (+, -, , /, etc.)
- In-place operations for memory efficiency

#### Performance & Backends

- Automatic GPU acceleration via CUDA (when available)
- Optimized CPU operations via BLAS/LAPACK (when available)
- Smart backend selection: GPU → BLAS → CPU fallback
- Parallel execution support with Intel TBB
- Performance profiling and benchmarking tools

#### Mathematical Operations

- Mathematical functions: exp, log, sqrt, pow, abs
- Trigonometric functions: sin, cos, tan, asin, acos, atan
- Activation functions: sigmoid, relu, tanh, softmax
- Statistical operations: mean, variance, std, min, max, median, sum, product
- Normalization operations: standardization, min-max scaling
- Clipping and value constraints

#### Linear Algebra (wrapping third-party libraries)

- Matrix multiplication (matmul)
- Matrix decompositions: LU, QR, Cholesky, SVD, Eigenvalue
- Linear system solvers (LU-based, QR-based)
- Matrix properties: determinant, inverse, rank, trace, norm
- Advanced operations: kronecker product, cross product

#### Neural Networks & Deep Learning

- Automatic differentiation (autograd) for gradient computation
- Comprehensive gradient tracking and backpropagation
- Neural network layers: Linear, ReLU, Sigmoid, Softmax, Dropout, BatchNorm
- Loss functions: MSE, CrossEntropy
- Optimizers: SGD with momentum, Adam with adaptive learning rates
- Training utilities and network building blocks

#### Data I/O & Serialization

- Tensor serialization (save/load to binary format)
- NumPy array interoperability (Python bindings)
- CSV and text data loading utilities
- Cross-platform binary format support

#### Multi-Language Support

- C++ API: Header-only template library with full feature access (see `include/tensor.h`)
- C API: In-progress C bindings for foreign function interfaces
- Python API: In-progress Python bindings with NumPy interoperability (see `python/` directory)

### Backend Selection

The library automatically selects the best available backend at runtime:

1. GPU (CUDA): Used by default if compiled with `USE_GPU` and GPU hardware is available
2. BLAS: Used if GPU is unavailable but compiled with `USE_BLAS`
3. CPU: Fallback implementation with optimized C++ algorithms

## Building

### Requirements

- CMake 3.14 or higher
- C++ compiler with C++20 support (GCC 10+, Clang 12+, MSVC 2019+)
- Optional: CUDA Toolkit (for GPU support)
- Optional: BLAS/LAPACK (for optimized CPU operations)
- Optional: Intel TBB (for parallel execution)
- Optional: Doxygen (for documentation generation)

### Build Instructions

```bash
git clone git@github.com:jailop/tensor.git
cd tensor
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
ctest --output-on-failure
sudo cmake --install .
```

### Build Options

The build system automatically detects available features:

- CUDA: Automatically enabled if CUDA compiler is found
- BLAS/LAPACK: Automatically linked if found
- TBB: Automatically linked if found for parallel execution

To specify CUDA architecture:
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..  # For Ampere (RTX 30xx)
```

## Usage

### C++ API

#### Basic Tensor Operations

```cpp
#include "tensor.h"
#include "tensor_types.h"

using namespace tensor4d;

int main() {
    // Create a 2D tensor (matrix) - automatically uses GPU if available
    Matrixf A({3, 4});
    A.fill(1.0f);
    
    // Check which backend is being used
    std::cout << "Backend: " << backend_name(A.backend()) << std::endl;
    
    // Matrix operations
    Matrixf B = Matrixf::eye(4);
    Matrixf C = A.matmul(B);
    
    // Element-wise operations
    Matrixf D = C.exp().log().sqrt();
    
    // Statistical operations
    float mean = C.mean();
    float std = C.std();
    
    return 0;
}
```

#### Neural Network Example

```cpp
#include "nn_layers.h"
#include "optimizers.h"

using namespace tensor4d;
using namespace tensor4d::nn;

int main() {
    // Create a simple feedforward network (784 -> 128 -> 10)
    Linearf fc1(784, 128, true);
    ReLUf relu;
    Linearf fc2(128, 10, true);
    Softmaxf softmax;
    
    // Create input (batch_size=32, features=784)
    Matrixf input({32, 784});
    input.random_normal(0.0f, 1.0f);
    
    // Forward pass (automatically uses GPU if available)
    auto h1 = fc1.forward(input);
    auto a1 = relu.forward(h1);
    auto h2 = fc2.forward(a1);
    auto output = softmax.forward(h2);
    
    // Create optimizer
    SGD optimizer(0.01f, 0.9f);  // lr=0.01, momentum=0.9
    
    // Training loop would go here...
    
    return 0;
}
```

### C API

The library provides a in-progress C API for use with other languages:

```c
#include "tensor_c.h"

int main() {
    // Check GPU availability
    if (tensor_c_is_gpu_available()) {
        printf("GPU acceleration enabled!\n");
    }
    
    // Create matrices (automatically uses GPU if available)
    MatrixFloatHandle A, B, C;
    matrix_float_ones(3, 3, &A);
    matrix_float_eye(3, &B);
    
    // Matrix multiplication (GPU-accelerated)
    matrix_float_matmul(A, B, &C);
    
    // Print result
    matrix_float_print(C);
    
    // Clean up
    matrix_float_destroy(A);
    matrix_float_destroy(B);
    matrix_float_destroy(C);
    
    return 0;
}
```

### Python API

The library includes Python bindings with NumPy interoperability:

```python
import numpy as np
import tensor4d as t4d

# Create from Python lists or NumPy arrays
matrix = t4d.Matrixf([[1.0, 2.0], [3.0, 4.0]])
np_array = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
tensor = t4d.Matrixf.from_numpy(np_array)

# Perform operations (GPU-accelerated if available)
result = matrix.matmul(tensor)
activated = result.relu()

# Convert back to NumPy for visualization/further processing
np_result = activated.numpy()

# Automatic differentiation
x = t4d.Vectorf([2.0, 3.0, 4.0])
x.set_requires_grad(True)
y = x  x
z = y.sum()
z.backward()
gradients = x.grad().numpy()

# Neural network training
fc1 = t4d.nn.Linearf(784, 128, use_bias=True)
optimizer = t4d.nn.SGD(0.01, 0.9)
# ... training loop
```

Installation & Documentation:

For detailed Python API documentation, installation instructions, and examples, see:
- [python/README.md](python/README.md) - Comprehensive Python guide
- `python/example_.py` - Working examples
- `python/test_.py` - Test suite demonstrating features

## Library Architecture

### Core Components

#### `include/tensor.h` - Main Template Header

The primary header file containing the full `Tensor<T, N>` template class implementation. This is a header-only template library that provides:

- Complete template class definition with all member functions
- Compile-time dimension checking and type safety
- Inline implementations for optimal performance
- Automatic backend selection (GPU/BLAS/CPU)
- All mathematical, statistical, and linear algebra operations
- Autograd functionality for automatic differentiation

Usage: Include this header directly in your C++ code. All functionality is available at compile time through templates.

```cpp
#include "tensor.h"
using namespace tensor4d;

// Template instantiation happens at compile time
Tensor<float, 2> matrix({3, 4});  // Creates a float matrix
Tensor<double, 3> tensor3d({2, 3, 4});  // Creates a double 3D tensor
```

#### `src/tensor_instantiations.cc` - Explicit Template Instantiations

Provides explicit template instantiations for common tensor types, enabling the library to be compiled into static/shared libraries. This file:

- Pre-compiles template code for common types (float/double, rank 2-4)
- Reduces compilation time for client code using these common types
- Enables binary distribution without exposing full implementation
- Supports the following instantiated types:
  - `Tensor<float, 2>`, `Tensor<double, 2>` (Matrixf, Matrixd)
  - `Tensor<float, 3>`, `Tensor<double, 3>` (Tensor3f, Tensor3d)
  - `Tensor<float, 4>`, `Tensor<double, 4>` (Tensor4f, Tensor4d)

Key Difference: While `tensor.h` contains all template code for any type and dimension, `tensor_instantiations.cc` pre-compiles specific commonly-used instantiations. Users can still instantiate other types (e.g., `Tensor<int, 5>`) by including the header, but pre-instantiated types link against the compiled library.

Note: 1D tensors (Vectors) are not explicitly instantiated because some operations (transpose, vstack, hstack) require N ≥ 2. Vectors remain header-only.

### Additional Headers

#### Type Aliases and Utilities

- `tensor_types.h` - Convenient type aliases (Vectorf, Matrixf, Tensor3f, Tensor4f, etc.)
- `tensor_ops.h` - Advanced tensor operations and broadcasting utilities
- `tensor_io.h` - Tensor serialization (save/load binary format)
- `tensor_perf.h` - Performance profiling and benchmarking utilities

#### Linear Algebra

- `linalg.h` - Basic linear algebra (matrix multiplication, inverse, determinant, transpose)
- `linalg_advanced.h` - Matrix decompositions (LU, QR, Cholesky, SVD, Eigenvalue)

#### Neural Networks

- `nn_layers.h` - Layer implementations (Linear, ReLU, Sigmoid, Softmax, Dropout, BatchNorm)
- `loss_functions.h` - Loss functions (MSE, CrossEntropy)
- `optimizers.h` - Optimization algorithms (SGD, Adam)

#### Multi-Language Interfaces

##### C API (`include/tensor_c.h`)

C bindings for all tensor operations, enabling use from:

- C programs
- Languages with C FFI (Python ctypes, Rust, Go, etc.)
- Systems without C++ support

Features:

- Opaque handle-based API (MatrixFloatHandle, etc.)
- GPU availability checking
- Memory management functions

Build: Produces `libtensor_c.so` shared library

##### Python Bindings (`python/` directory)

Python interface using pybind11 with:

- NumPy interoperability (seamless conversion to/from NumPy arrays)
- Pythonic API with operator overloading
- All tensor types: Vectorf/d, Matrixf/d, Tensor3f/d, Tensor4f/d
- Neural network layers and optimizers
- Automatic differentiation support

Key Files:

- `tensor_wrapper.cc` - pybind11 binding implementation
- `python/README.md` - Python-specific documentation and examples
- `setup.py` - Python package build configuration
- `build.sh` - Build script for Python module

Build: Produces `tensor4d.so` Python extension module

#### GPU Support

- `tensor_gpu.cuh` - CUDA kernel implementations for GPU acceleration
- `tensor_gpu.cu` - CUDA kernel definitions


## Documentation

Generate API documentation with Doxygen (if installed):

```bash
cd build
make doc        # Generate documentation
make doc_open   # Generate and open in browser
```

Documentation will be generated in the `docs/html/` directory.

## Testing

Run the test suite:

```bash
cd build
ctest --output-on-failure
```

Or run tests directly:

```bash
./tensor_test              # Main test suite
./tensor_nn_enhancements_test  # Neural network tests
./tensor_c_test            # C API tests
```

## Performance Benchmarks

Run performance benchmarks:

```bash
cd build
./tensor_perf
# or
make run_perf
```

This will benchmark various operations across CPU, BLAS, and GPU (if available).

## References

- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- BLAS Reference: http://www.netlib.org/blas/
- LAPACK Reference: http://www.netlib.org/lapack/

