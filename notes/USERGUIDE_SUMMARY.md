# User Guide Summary

The comprehensive user guide has been created and is located in `./userguide/`.

## Structure

The guide consists of 18 sections covering all aspects of the tensor library:

### Core Documentation (Sections 1-9)
1. **Getting Started** - Installation and basic setup
2. **Core Tensor Operations** - Fundamentals of tensor manipulation
3. **Shape Manipulation** - Reshaping and transforming tensors
4. **Mathematical Operations** - Element-wise operations and reductions
5. **Linear Algebra** - Matrix operations and decompositions
6. **Automatic Differentiation** - Gradient computation and backpropagation
7. **Machine Learning Features** - Loss functions and optimizers
8. **Advanced Indexing** - Fancy indexing and masking
9. **I/O Operations** - Loading and saving tensors

### Advanced Topics (Sections 10-15)
10. **Performance Optimization** - GPU, BLAS, threading, memory pooling
11. **Normalization and Views** - Data preprocessing and submatrix views
12. **Random Sampling** - Statistical distributions and sampling
13. **Sorting and Searching** - Data organization operations
14. **Stacking and Concatenation** - Combining and splitting tensors
15. **Best Practices** - Guidelines and common patterns

### Reference & Integration (Sections 16-18)
16. **API Reference** - Quick function reference
17. **Python Integration** - Using the library from Python with NumPy interop
18. **C Interface** - Using the library from C code

## Key Highlights

### Language Support
- **C++**: Native implementation with modern C++17 features
- **Python**: Bindings via pybind11 with zero-copy NumPy conversion
- **C**: Full-featured opaque handle API for legacy code integration

### Performance Features
- Automatic backend selection: GPU → BLAS → CPU
- Memory pooling for reduced allocations
- Multi-threading support
- Mixed precision (FP16, BF16)
- Lazy evaluation

### Machine Learning
- Comprehensive autograd system
- Loss functions: MSE, Cross Entropy, BCE, L1, Smooth L1, Hinge, KL Divergence, Focal
- Optimizers: SGD, Adam, AdamW, RMSprop

### Linear Algebra
- BLAS/LAPACK-backed operations
- Decompositions: SVD, QR, Cholesky, LU, Eigenvalue
- Solvers: LU, QR, Cholesky, Least Squares
- Matrix operations: Inverse, Determinant, Rank, Pseudo-inverse

### Data Operations
- NumPy-compatible I/O (.npy format)
- Advanced indexing and broadcasting
- Statistical functions (correlation, covariance, quantiles)
- Normalization (L1, L2, Z-score, Min-Max)

## Usage Examples

### C++ Example
```cpp
#include "tensor.h"
#include "optimizers.h"

auto W = Matrix<float>::randn({784, 10}, true);
Adam optimizer({&W}, 0.001f);

for (int epoch = 0; epoch < 100; epoch++) {
    auto loss = compute_loss(W);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

### Python Example
```python
import tensor4d as t4d

W = t4d.Matrixf.randn([784, 10])
W.set_requires_grad(True)
optimizer = t4d.Adam([W], lr=0.001)

for epoch in range(100):
    loss = compute_loss(W)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### C Example
```c
#include "tensor_c.h"

MatrixFloatHandle W;
OptimizerHandle optimizer;

matrix_float_random_normal((size_t[]){784, 10}, &W);
tensor_float_requires_grad(W, true);
optimizer_adam_create(0.001f, 0.9f, 0.999f, &optimizer);
optimizer_add_parameter(optimizer, W);

for (int epoch = 0; epoch < 100; epoch++) {
    // compute loss
    optimizer_zero_grad(optimizer);
    tensor_float_backward(loss);
    optimizer_step(optimizer);
}

optimizer_destroy(optimizer);
matrix_float_destroy(W);
```

## Documentation Generation

The library includes comprehensive Doxygen documentation:

```bash
cd build
make doc
```

Documentation will be generated in `docs/html/index.html`.

## Testing

The guide examples are backed by 411+ passing tests covering:
- Core tensor operations
- Mathematical functions
- Linear algebra
- Autograd
- I/O operations
- Shape manipulation
- Random sampling
- Sorting and searching
- Broadcasting
- Normalization

Run tests with:
```bash
cd build
./tensor_test
```

## Getting Started

To start using the library:

1. Read `userguide/01-getting-started.md` for installation
2. Follow examples in `userguide/02-core-tensor-operations.md`
3. Explore specific topics based on your use case
4. Reference `userguide/16-api-reference.md` for function lookups

## Integration Paths

### For Python Users
- Use `python/build.sh` to build Python bindings
- Import as `import tensor4d`
- Seamless NumPy conversion via `.from_numpy()` and `.numpy()`

### For C Users
- Link against `libtensor_c.so`
- Include `tensor_c.h`
- Use opaque handles with explicit create/destroy functions

### For C++ Users
- Link against `libtensor_static.a` or `libtensor_shared.so`
- Include appropriate headers from `include/`
- Use template-based API directly

## File Locations

- **User Guide**: `./userguide/` (18 markdown files)
- **C Interface Spec**: `./c_interop.md`
- **Feature Tracking**: `./features.md`
- **API Docs**: `./docs/html/` (after building)
- **Examples**: `./tests/` (test files serve as examples)
- **Python Bindings**: `./python/`

## Version

- Current Version: 1.4.2
- Status: Production Ready
- Test Coverage: 411 tests (100% passing)
- Documentation: Complete

---

For more information, start with `userguide/00-index.md` or `userguide/README.md`.
