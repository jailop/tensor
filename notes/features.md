# Tensor Library Features

**Version**: 1.4.2 | **Status**: Production Ready | **Tests**: 411 passing (100%)

## Core Capabilities

### Computation Backend
- **GPU**: CUDA acceleration (automatic selection)
- **CPU**: BLAS/LAPACK optimization (OpenBLAS, MKL, Accelerate)
- **Fallback**: Pure C++ implementation
- **Auto-selection**: GPU → BLAS → CPU

### Python Integration
- pybind11 bindings with NumPy interoperability (zero-copy)
- Complete API wrapper: tensors, autograd, losses, optimizers, I/O
- NPY format support for data exchange

### Performance
- Memory pooling, multi-threading (ThreadPool)
- Mixed precision (FP16, BF16)
- Lazy evaluation with operation fusion

### Automatic Differentiation
- Gradient tracking, computational graph
- Backward pass, gradient accumulation
- `requires_grad()`, `backward()`, `zero_grad()`, `detach()`

### Data Operations
- **Indexing**: Fancy indexing, boolean masks, take/put/select/where
- **Shape**: reshape, flatten, squeeze, unsqueeze, permute, repeat, tile
- **Stacking**: concatenate, stack, split, chunk, vstack, hstack
- **Broadcasting**: NumPy-compatible automatic broadcasting

### Mathematics
- **Element-wise**: exp, log, sqrt, pow, sin, cos, tan, abs, ceil, floor, clamp
- **Activations**: sigmoid, tanh, relu, leaky_relu, softmax
- **Reductions**: sum, mean, variance, std, min, max, median, prod
- **Cumulative**: cumsum, cumprod, argmin, argmax
- **Statistical**: Pearson/Spearman correlation, covariance, quantile

### Linear Algebra
- **Operations**: matmul, dot, cross (3D), transpose
- **Decompositions**: SVD, QR, Cholesky, LU (with pivoting), eigenvalues/vectors
- **Solvers**: Linear systems (LU/Cholesky/QR), least squares (QR/SVD), pseudo-inverse
- **Advanced**: Determinant, inverse, matrix rank, Kronecker product
- **Types**: Specialized Matrix<T> and Vector<T> classes

### Normalization & Views
- **Normalization**: L1, L2, Z-score, min-max (axis-aware)
- **Submatrix views**: row(), col(), diag(), block(), head(), tail(), topRows(), bottomRows(), leftCols(), rightCols()

### Machine Learning
- **Losses**: MSE, CrossEntropy, BinaryCrossEntropy, L1, SmoothL1
- **Optimizers**: SGD (momentum, Nesterov), Adam, AdamW, RMSprop

### I/O Operations
- Save/load (binary, text, NPY format)
- NumPy .npy format compatibility
- Pretty printing with formatting

### Random Operations
- **Distributions**: uniform, normal, exponential, gamma, beta, chi-square, Cauchy, multinomial
- **Sampling**: randperm, choice (with/without replacement)
- **Utilities**: seed() for reproducibility

### Sorting & Searching
- sort, argsort, topk, unique, searchsorted (binary search)

## Test Coverage

17 test suites, 411 tests (100% passing):
- TensorTest (167), IndexingTest (20), ReductionTest (19), IOTest (19)
- ShapeTest (8), RandomTest (16), SortingTest (15), StackingTest (19)
- MathTest (6), LinalgTest (30), LossFunctionTest (12), OptimizerTest (9)
- PerfTest (30), MatrixTest (11), VectorTest (9), BroadcastingTest (26)
- NormalizationTest (15)

## Documentation

- ✅ **Doxygen**: Comprehensive API docs with examples (docs/)
- ✅ **User Guide**: 17-chapter guide covering basics to advanced (userguide/)
- ✅ **Instantiations**: Pre-compiled Matrix/Tensor types documented

## Project Stats

- **Headers**: 10 (tensor.h, linalg.h, tensor_views.h, normalization.h, loss_functions.h, optimizers.h, tensor_io.h, tensor_perf.h, matrix.h, vector.h)
- **Tests**: 17 files (~411 tests)
- **LOC**: ~20,000+
- **Libraries**: libtensor.a (static), libtensor.so (dynamic)

## Future Roadmap

### Planned v1.5 (High Priority Linear Algebra)
All core linear algebra complete! Focus on optimization and new domains.

### Future v2.0+ (Domain Extensions)
- Sparse matrices (COO, CSR/CSC formats)
- Geometry module (quaternions, rotations)
- FFT support (1D/2D/3D)
- Schur decomposition
- Matrix exponential/logarithm
- Iterative solvers (CG, BiCGSTAB, GMRES)
- Special functions (Bessel, error functions)
- Batch operations, distributed computing

### Already Beyond Armadillo/Eigen
- ✅ Autograd (automatic differentiation)
- ✅ Mixed precision & memory pooling
- ✅ ML optimizers & loss functions
- ✅ NumPy I/O interoperability
- ✅ Python bindings

---

**Last Updated**: 2025-01-09

