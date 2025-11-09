# Tensor Library - User Guide

Welcome to the comprehensive user guide for the Tensor Library!

## Structure

This guide is organized into 18 sections, each focusing on a specific aspect of the library:

| Section | Title | Description |
|---------|-------|-------------|
| [00](00-index.md) | **Index** | Overview and table of contents |
| [01](01-getting-started.md) | **Getting Started** | Installation, build, and first examples |
| [02](02-core-tensor-operations.md) | **Core Tensor Operations** | Creation, indexing, arithmetic, broadcasting |
| [03](03-shape-manipulation.md) | **Shape Manipulation** | Reshape, transpose, permute, repeat |
| [04](04-mathematical-operations.md) | **Mathematical Operations** | Element-wise functions, reductions, statistics |
| [05](05-linear-algebra.md) | **Linear Algebra** | Matrix operations, decompositions, specialized types |
| [06](06-autograd.md) | **Automatic Differentiation** | Gradient tracking, computational graphs |
| [07](07-machine-learning.md) | **Machine Learning Features** | Neural network layers, loss functions, optimizers |
| [08](08-advanced-indexing.md) | **Advanced Indexing** | Fancy indexing, boolean masks, conditionals |
| [09](09-io-operations.md) | **I/O Operations** | Save/load, NumPy interop, printing |
| [10](10-performance-optimization.md) | **Performance Optimization** | GPU, BLAS, memory pooling, multi-threading |
| [11](11-normalization-views.md) | **Normalization and Views** | Data normalization, submatrix views |
| [12](12-random-sampling.md) | **Random Sampling** | Distributions, permutations, random choice |
| [13](13-sorting-searching.md) | **Sorting and Searching** | Sort, argsort, topk, unique, binary search |
| [14](14-stacking-concatenation.md) | **Stacking and Concatenation** | Stack, concatenate, split, chunk |
| [15](15-best-practices.md) | **Best Practices** | Error handling, memory, performance, patterns |
| [16](16-api-reference.md) | **API Reference** | Quick reference for all functions |
| [17](17-python-integration.md) | **Python Integration** | Python bindings and NumPy interop |
| [18](18-c-interface.md) | **C Interface** | Using the library from C code |

## How to Use This Guide

### For Beginners

Start with these sections in order:
1. [Getting Started](01-getting-started.md) - Setup and installation
2. [Core Tensor Operations](02-core-tensor-operations.md) - Basic operations
3. [Shape Manipulation](03-shape-manipulation.md) - Working with tensor shapes
4. [Mathematical Operations](04-mathematical-operations.md) - Common computations

### For ML Practitioners

Focus on these sections:
1. [Getting Started](01-getting-started.md) - Quick setup
2. [Automatic Differentiation](06-autograd.md) - Understanding gradients
3. [Machine Learning Features](07-machine-learning.md) - Loss functions and optimizers
4. [Performance Optimization](10-performance-optimization.md) - GPU acceleration
5. [Python Integration](17-python-integration.md) - Using from Python

### For Scientists/Engineers

These sections are most relevant:
1. [Linear Algebra](05-linear-algebra.md) - Matrix operations and decompositions
2. [Mathematical Operations](04-mathematical-operations.md) - Statistical functions
3. [I/O Operations](09-io-operations.md) - Data import/export
4. [Normalization and Views](11-normalization-views.md) - Data preprocessing
5. [C Interface](18-c-interface.md) - Using from legacy C code

### As a Reference

Use the [API Reference](16-api-reference.md) for quick lookup of function names and signatures.

## Quick Examples

### Basic Tensor Creation and Operations

```cpp
#include "tensor.h"

auto A = Matrix<float>::randn({3, 4});
auto B = Matrix<float>::ones({3, 4});
auto C_var = A + B;
auto C = std::get<Matrix<float>>(C_var);
C.print();
```

### Linear Algebra

```cpp
#include "linalg.h"

auto A = Matrix<float>::randn({100, 50});
auto B = Matrix<float>::randn({50, 20});
auto C_var = matmul(A, B);  // Result: 100x20
```

### Machine Learning

```cpp
#include "tensor.h"
#include "loss_functions.h"
#include "optimizers.h"

// Define parameters
auto W = Matrix<float>::randn({784, 10}, true);  // requires_grad=true
auto b = Vector<float>::zeros({10}, true);

// Setup
auto params = {&W, &b};
Adam optimizer(params, 0.001f);
CrossEntropyLoss loss_fn;

// Training loop
for (int epoch = 0; epoch < 10; epoch++) {
    auto logits_var = matmul(X, W) + b;
    auto logits = std::get<Matrix<float>>(logits_var);
    auto loss = loss_fn.forward(logits, y);
    
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

## Additional Resources

- **API Documentation**: Full Doxygen documentation in `docs/html/`
- **Test Files**: Comprehensive examples in `tests/` directory
- **Feature List**: Complete feature tracking in `features.md`
- **Source Code**: All headers in `include/` directory

## Library Features

### Core Capabilities
✅ Multi-dimensional tensors with compile-time ranks  
✅ Automatic differentiation (autograd)  
✅ GPU acceleration via CUDA  
✅ CPU optimization via BLAS  
✅ NumPy-like API  

### Performance
✅ Automatic backend selection (GPU → BLAS → CPU)  
✅ Memory pooling  
✅ Multi-threading  
✅ Mixed precision (FP16, BF16)  
✅ Lazy evaluation  

### Machine Learning
✅ Neural network layers (Linear, ReLU, Sigmoid, Tanh, Softmax)  
✅ Regularization (Dropout, Batch Normalization)  
✅ Loss functions (MSE, Cross Entropy, BCE, L1, Smooth L1)  
✅ Optimizers (SGD, Adam, AdamW, RMSprop)  
✅ Activation functions  
✅ Gradient tracking and backpropagation  
✅ Complete training workflows (see MNIST example)  

### Linear Algebra
✅ Matrix multiplication, decompositions  
✅ SVD, QR, Cholesky, LU, Eigenvalues  
✅ Linear solvers (LU, QR, Cholesky)
✅ Least squares (QR, SVD)
✅ Matrix inverse, determinant, rank, pseudo-inverse
✅ Kronecker product, covariance
✅ Specialized Matrix and Vector types  

### Data Operations
✅ Advanced indexing and slicing  
✅ Broadcasting  
✅ I/O (save/load, NumPy .npy format)  
✅ Sorting, searching, stacking  
✅ Normalization functions (L1, L2, Z-score, Min-Max)
✅ Statistical operations (correlation, covariance, quantiles)

### Language Bindings
✅ C++ native API
✅ Python bindings via pybind11 (zero-copy NumPy interop)
✅ C interface for legacy code integration  

## Version Information

- **Current Version**: 1.5.0
- **Status**: Production Ready
- **Test Coverage**: 468 tests, 100% passing
- **Lines of Code**: ~12,600+
- **Python Bindings**: Available (pybind11)

## Contributing

[Information about contributing to the library]

## License

[Your License Here]

---

**Start Reading**: [Getting Started →](01-getting-started.md)
