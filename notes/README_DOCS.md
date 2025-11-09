# Tensor Library Documentation

## Overview

This is a comprehensive C++ tensor library for machine learning and scientific computing with Python and C bindings.

## 📚 Documentation Structure

| Document | Purpose | Audience |
|----------|---------|----------|
| [DOCUMENTATION_OVERVIEW.md](DOCUMENTATION_OVERVIEW.md) | Complete documentation map | Everyone |
| [USERGUIDE_SUMMARY.md](USERGUIDE_SUMMARY.md) | Quick reference for user guide | New users |
| [userguide/](userguide/) | 18-section comprehensive guide | All users |
| [c_interop.md](c_interop.md) | C interface specification | C developers |
| [features.md](features.md) | Feature tracking | Contributors |
| [docs/html/](docs/html/) | API reference (Doxygen) | Developers |

## 🚀 Quick Start

### For C++ Users
```cpp
#include "tensor.h"

auto A = Matrix<float>::randn({100, 50});
auto B = Matrix<float>::randn({50, 20});
auto C_var = matmul(A, B);
auto C = std::get<Matrix<float>>(C_var);
```

### For Python Users
```python
import tensor4d as t4d

A = t4d.Matrixf.randn([100, 50])
B = t4d.Matrixf.randn([50, 20])
C = t4d.matmul(A, B)
```

### For C Users
```c
#include "tensor_c.h"

MatrixFloatHandle A, B, C;
matrix_float_random_normal((size_t[]){100, 50}, &A);
matrix_float_random_normal((size_t[]){50, 20}, &B);
matrix_float_matmul(A, B, &C);

matrix_float_destroy(A);
matrix_float_destroy(B);
matrix_float_destroy(C);
```

## 📖 User Guide Sections

The complete user guide is in `./userguide/`:

1. **Getting Started** - Setup and installation
2. **Core Operations** - Tensor basics
3. **Shape Manipulation** - Reshape, transpose, etc.
4. **Math Operations** - Element-wise and reductions
5. **Linear Algebra** - Matrix operations and decompositions
6. **Autograd** - Automatic differentiation
7. **Machine Learning** - Loss functions and optimizers
8. **Advanced Indexing** - Fancy indexing and masking
9. **I/O Operations** - Save/load, NumPy format
10. **Performance** - GPU, BLAS, optimization
11. **Normalization** - Data preprocessing
12. **Random Sampling** - Statistical distributions
13. **Sorting & Searching** - Data organization
14. **Stacking** - Combining tensors
15. **Best Practices** - Guidelines and patterns
16. **API Reference** - Quick function lookup
17. **Python Integration** - Python bindings
18. **C Interface** - C API usage

## ✨ Key Features

### Multi-Language Support
- ✅ Native C++ API
- ✅ Python bindings (pybind11)
- ✅ C interface (opaque handles)

### Performance
- ✅ GPU acceleration (CUDA)
- ✅ BLAS/LAPACK optimization
- ✅ Automatic backend selection: GPU → BLAS → CPU
- ✅ Memory pooling
- ✅ Multi-threading
- ✅ Mixed precision (FP16/BF16)
- ✅ Lazy evaluation

### Machine Learning
- ✅ Automatic differentiation (autograd)
- ✅ Loss functions: MSE, Cross Entropy, BCE, L1, Smooth L1, Hinge, KL, Focal
- ✅ Optimizers: SGD, Adam, AdamW, RMSprop
- ✅ Gradient tracking and backpropagation

### Linear Algebra
- ✅ Matrix operations: matmul, transpose, inverse, determinant
- ✅ Decompositions: SVD, QR, Cholesky, LU, Eigenvalue
- ✅ Solvers: LU, QR, Cholesky, Least Squares
- ✅ Advanced: Pseudo-inverse, rank, Kronecker product

### Data Operations
- ✅ NumPy-compatible I/O (.npy format)
- ✅ Broadcasting
- ✅ Advanced indexing and slicing
- ✅ Statistical functions (correlation, covariance, quantiles)
- ✅ Normalization (L1, L2, Z-score, Min-Max)
- ✅ Random sampling (uniform, normal, exponential, etc.)

## 🔧 Building

### C++ Library
```bash
mkdir build && cd build
cmake .. -DUSE_GPU=ON -DUSE_BLAS=ON
make -j$(nproc)
./tensor_test
```

### Python Bindings
```bash
cd python
./build.sh
python examples/basic_operations.py
```

### Documentation
```bash
cd build
make doc
# Open docs/html/index.html
```

## 📊 Status

- **Version**: 1.4.2
- **Status**: Production Ready
- **Tests**: 411 passing (100%)
- **Documentation**: Complete
- **Python Bindings**: Available
- **C Interface**: Specified (implementation pending)

## 🎯 Use Cases

### Machine Learning
Train neural networks with autograd:
```cpp
auto W = Matrix<float>::randn({784, 10}, true);
Adam optimizer({&W}, 0.001f);

for (int epoch = 0; epoch < 100; epoch++) {
    auto loss = compute_loss(W);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

### Scientific Computing
Solve linear systems:
```cpp
auto A = Matrix<float>::randn({100, 100});
auto b = Vector<float>::randn({100});
auto x_var = solve_lu(A, b);
auto x = std::get<Vector<float>>(x_var);
```

### Data Analysis
Statistical operations:
```python
import tensor4d as t4d

data = t4d.Matrixf.randn([1000, 20])
mean = data.mean(axis=0)
std = data.std(axis=0)
corr = t4d.correlation_matrix(data)
```

## 📝 Documentation Files

```
.
├── README_DOCS.md              # This file - documentation entry point
├── DOCUMENTATION_OVERVIEW.md   # Complete documentation map
├── USERGUIDE_SUMMARY.md        # User guide summary
├── c_interop.md               # C interface specification
├── features.md                # Feature tracking
├── userguide/                 # 18-section user guide
│   ├── 00-index.md
│   ├── 01-getting-started.md
│   ├── ...
│   └── 18-c-interface.md
├── docs/html/                 # Doxygen API docs (generated)
├── tests/                     # Example code (411+ tests)
├── python/                    # Python bindings and examples
└── include/                   # Well-commented headers
```

## 🔍 Finding Information

| I want to... | Read this... |
|--------------|--------------|
| Get started | `userguide/01-getting-started.md` |
| Learn basics | `userguide/02-core-tensor-operations.md` |
| Use from Python | `userguide/17-python-integration.md` |
| Use from C | `userguide/18-c-interface.md` or `c_interop.md` |
| Look up a function | `userguide/16-api-reference.md` |
| See examples | `tests/` directory or user guide sections |
| Check features | `features.md` |
| Train ML models | `userguide/07-machine-learning.md` |
| Optimize performance | `userguide/10-performance-optimization.md` |
| Understand autograd | `userguide/06-autograd.md` |

## 🤝 Contributing

When adding features:
1. Update relevant user guide section
2. Add Doxygen comments to code
3. Add test cases
4. Update `features.md`
5. Update API reference if needed

## 📄 License

[Your License Here]

## 📧 Contact

[Your Contact Info Here]

---

**Start Here**: Read [DOCUMENTATION_OVERVIEW.md](DOCUMENTATION_OVERVIEW.md) for a complete documentation map, or jump straight to [userguide/01-getting-started.md](userguide/01-getting-started.md) to begin using the library.
