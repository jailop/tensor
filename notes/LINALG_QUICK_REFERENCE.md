# Linear Algebra Quick Reference

## Include Header
```cpp
#include "linalg.h"
```

## Specialized Types
```cpp
Vector<float> v({100});          // 1D tensor
Matrix<float> m({10, 20});       // 2D tensor
```

## Vector Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Norm | `linalg::norm(v)` | L2 norm |
| Normalize | `linalg::normalize(v)` | Unit vector |
| Dot Product | `linalg::dot(a, b)` | Inner product |
| Outer Product | `linalg::outer(a, b)` | Returns matrix |

## Matrix Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Mat-Vec Multiply | `linalg::matvec(A, x)` | Returns vector |
| Mat-Mat Multiply | `linalg::matmul(A, B)` | Matrix product |
| Transpose | `linalg::transpose(A)` | A^T |
| Trace | `linalg::trace(A)` | Sum of diagonal |
| Diagonal Extract | `linalg::diag(A)` | Returns vector |
| Diagonal Create | `linalg::diag(v)` | Returns matrix |
| Identity | `linalg::eye<T>(n)` | n×n identity |
| Frobenius Norm | `linalg::frobenius_norm(A)` | Matrix norm |

## Tensor Views

### 1D Slicing
```cpp
auto view = TensorSlice<T, 1>::slice(tensor, dim, start, end);
```

### Matrix Views
```cpp
auto row = TensorSlice<T, 2>::row(matrix, row_idx);
auto col = TensorSlice<T, 2>::col(matrix, col_idx);
auto block = TensorSlice<T, 2>::block(matrix, r0, r1, c0, c1);
```

### View Operations
```cpp
view.fill(value);               // Fill with value
Tensor<T, N> t = view.to_tensor();  // Copy to new tensor
view[{i}] = value;              // Modify (affects parent)
```

## Performance Optimization

### BLAS Acceleration (Automatic)
- Enabled with `-DUSE_BLAS` at compile time
- Used for float and double types
- Falls back to CPU implementation for other types

### GPU Acceleration
```cpp
// Enable GPU for tensor
Vector<float> v({1000}, true);   // use_gpu = true
Matrix<float> m({100, 100}, true);

// Disable GPU (CPU only)
Vector<float> v({1000}, false);  // use_gpu = false
```

## Common Patterns

### Linear System Verification
```cpp
Matrix<float> A({n, n});
Vector<float> x({n});
Vector<float> b = linalg::matvec(A, x);  // b = Ax
```

### Matrix Chain
```cpp
auto AB = linalg::matmul(A, B);
auto ABC = linalg::matmul(AB, C);
```

### Sub-Matrix Operations
```cpp
Matrix<float> large({1000, 1000});
auto block = TensorSlice<float, 2>::block(large, 0, 100, 0, 100);
block.fill(0.0f);  // Zero out 100×100 block
```

### Normalize Vector
```cpp
Vector<float> v({n});
auto unit = linalg::normalize(v);
```

## Build and Test

```bash
# Build with BLAS and GPU
cmake -B build -S . -DUSE_BLAS=ON -DUSE_GPU=ON
cmake --build build -j$(nproc)

# Run tests
./build/tensor_test
./build/linalg_test

# Run benchmarks
./build/tensor_perf
```

## Documentation

- **Full Guide**: `LINALG_GUIDE.md`
- **Implementation Details**: `LINALG_IMPLEMENTATION_SUMMARY.md`
- **Tests**: `tests/linalg_test.cc`
- **Benchmarks**: `perf/tensor_perf.cc`
