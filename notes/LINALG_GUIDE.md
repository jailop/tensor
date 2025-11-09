# Linear Algebra Operations for Tensors

## Overview

This document describes the linear algebra operations and tensor view capabilities added to the tensor library. These features provide specialized types (`Vector`, `Matrix`) and operations optimized for common linear algebra tasks, with support for both CPU (with BLAS acceleration) and GPU execution.

## Table of Contents

1. [Specialized Types](#specialized-types)
2. [Vector Operations](#vector-operations)
3. [Matrix Operations](#matrix-operations)
4. [Tensor Views](#tensor-views)
5. [Performance Optimization](#performance-optimization)
6. [Examples](#examples)

## Specialized Types

### Vector

A `Vector<T>` is an alias for `Tensor<T, 1>`, providing a convenient type for 1D tensors:

```cpp
#include "linalg.h"

Vector<float> v({100});  // Create a vector with 100 elements
v[{0}] = 1.0f;
v[{1}] = 2.0f;
```

### Matrix

A `Matrix<T>` is an alias for `Tensor<T, 2>`, providing a convenient type for 2D tensors:

```cpp
Matrix<float> m({3, 4});  // Create a 3x4 matrix
m[{0, 0}] = 1.0f;
m[{1, 2}] = 3.5f;
```

## Vector Operations

All vector operations are in the `linalg` namespace.

### Norm

Compute the L2 (Euclidean) norm of a vector:

```cpp
Vector<float> v({3});
v[{0}] = 3.0f;
v[{1}] = 4.0f;
v[{2}] = 0.0f;

float norm_val = linalg::norm(v);  // Returns 5.0
```

**Performance**: Uses BLAS `dot` when available, GPU acceleration when enabled.

### Normalize

Normalize a vector to unit length:

```cpp
Vector<float> v({3});
v[{0}] = 3.0f;
v[{1}] = 4.0f;
v[{2}] = 0.0f;

auto normalized = linalg::normalize(v);
// normalized[{0}] = 0.6, normalized[{1}] = 0.8, normalized[{2}] = 0.0
```

### Dot Product

Compute the dot product of two vectors:

```cpp
Vector<float> a({3});
a[{0}] = 1.0f;
a[{1}] = 2.0f;
a[{2}] = 3.0f;

Vector<float> b({3});
b[{0}] = 4.0f;
b[{1}] = 5.0f;
b[{2}] = 6.0f;

float dot_product = linalg::dot(a, b);  // Returns 32.0 (1*4 + 2*5 + 3*6)
```

**Performance**: Uses BLAS `dot` when available, GPU acceleration when enabled.

### Outer Product

Compute the outer product of two vectors:

```cpp
Vector<float> a({2});
a[{0}] = 1.0f;
a[{1}] = 2.0f;

Vector<float> b({3});
b[{0}] = 3.0f;
b[{1}] = 4.0f;
b[{2}] = 5.0f;

Matrix<float> result = linalg::outer(a, b);
// result is a 2x3 matrix:
// [[3.0, 4.0, 5.0],
//  [6.0, 8.0, 10.0]]
```

## Matrix Operations

### Matrix-Vector Multiplication

Multiply a matrix by a vector:

```cpp
Matrix<float> mat({2, 3});
mat[{0, 0}] = 1.0f; mat[{0, 1}] = 2.0f; mat[{0, 2}] = 3.0f;
mat[{1, 0}] = 4.0f; mat[{1, 1}] = 5.0f; mat[{1, 2}] = 6.0f;

Vector<float> vec({3});
vec[{0}] = 1.0f;
vec[{1}] = 2.0f;
vec[{2}] = 3.0f;

Vector<float> result = linalg::matvec(mat, vec);
// result[{0}] = 14.0, result[{1}] = 32.0
```

**Performance**: Uses BLAS when available, GPU acceleration when enabled.

### Matrix-Matrix Multiplication

Multiply two matrices:

```cpp
Matrix<float> a({2, 3});
Matrix<float> b({3, 2});
// ... initialize matrices ...

Matrix<float> result = linalg::matmul(a, b);  // Result is 2x2
```

**Performance**: Uses BLAS `gemm` for optimal performance, GPU acceleration available.

### Transpose

Transpose a matrix:

```cpp
Matrix<float> mat({2, 3});
// ... initialize matrix ...

Matrix<float> transposed = linalg::transpose(mat);  // Result is 3x2
```

### Trace

Compute the trace (sum of diagonal elements) of a square matrix:

```cpp
Matrix<float> mat({3, 3});
// ... initialize matrix ...

float trace_val = linalg::trace(mat);
```

### Diagonal Operations

Extract diagonal elements from a matrix:

```cpp
Matrix<float> mat({3, 3});
// ... initialize matrix ...

Vector<float> diag_vec = linalg::diag(mat);  // Extract diagonal
```

Create a diagonal matrix from a vector:

```cpp
Vector<float> vec({3});
vec[{0}] = 1.0f;
vec[{1}] = 2.0f;
vec[{2}] = 3.0f;

Matrix<float> diag_mat = linalg::diag(vec);
// Creates a 3x3 matrix with vec values on diagonal, zeros elsewhere
```

### Identity Matrix

Create an identity matrix:

```cpp
Matrix<float> I = linalg::eye<float>(3);  // 3x3 identity matrix
```

### Frobenius Norm

Compute the Frobenius norm of a matrix:

```cpp
Matrix<float> mat({2, 2});
// ... initialize matrix ...

float frob_norm = linalg::frobenius_norm(mat);
```

## Tensor Views

Tensor views provide non-owning references to portions of tensor data, allowing efficient access to sub-tensors without copying.

### 1D Tensor Slice

Extract a contiguous slice from a 1D tensor:

```cpp
Tensor<float, 1> tensor({10});
// ... initialize tensor ...

auto view = TensorSlice<float, 1>::slice(tensor, 0, 2, 5);
// View elements [2, 5) (indices 2, 3, 4)

view[{0}] = 99.0f;  // Modifies tensor[{2}]
```

### Matrix Row View

Access a specific row of a matrix:

```cpp
Matrix<float> mat({3, 4});
// ... initialize matrix ...

auto row1 = TensorSlice<float, 2>::row(mat, 1);
// row1 is a view of the second row (index 1)

row1[{0}] = 99.0f;  // Modifies mat[{1, 0}]
```

### Matrix Column View

Access a specific column of a matrix:

```cpp
Matrix<float> mat({3, 4});
// ... initialize matrix ...

auto col2 = TensorSlice<float, 2>::col(mat, 2);
// col2 is a view of the third column (index 2)

col2[{0}] = 99.0f;  // Modifies mat[{0, 2}]
```

### Matrix Block View

Access a rectangular sub-matrix:

```cpp
Matrix<float> mat({4, 5});
// ... initialize matrix ...

auto block = TensorSlice<float, 2>::block(mat, 1, 3, 2, 4);
// Block from rows [1, 3) and columns [2, 4)
// This is a 2x2 sub-matrix

block[{0, 0}] = 99.0f;  // Modifies mat[{1, 2}]
```

### Converting Views to Tensors

Views can be converted to new tensors (copies data):

```cpp
auto view = TensorSlice<float, 2>::block(mat, 0, 2, 0, 2);
Tensor<float, 2> new_tensor = view.to_tensor();
// new_tensor is an independent copy
```

### Filling Views

Views can be filled with a value:

```cpp
Matrix<float> mat({4, 4});
mat.fill(0.0f);

auto block = TensorSlice<float, 2>::block(mat, 1, 3, 1, 3);
block.fill(5.0f);
// Only the 2x2 block is filled with 5.0, rest remains 0.0
```

## Performance Optimization

### BLAS Acceleration

When compiled with BLAS support (`-DUSE_BLAS`), the following operations use optimized BLAS routines:

- `linalg::dot()` - uses `cblas_sdot/ddot`
- `linalg::matvec()` - uses BLAS dot products
- `linalg::matmul()` - uses `cblas_sgemm/dgemm`
- `linalg::norm()` - uses BLAS dot product

### GPU Acceleration

When compiled with CUDA support (`-DUSE_GPU`) and tensors created with `use_gpu=true`:

- Matrix multiplication uses GPU kernels
- Vector dot products use GPU kernels
- Matrix-vector multiplication uses GPU kernels

### CPU-only Mode

When creating tensors with `use_gpu=false`, operations use optimized CPU implementations:

```cpp
Vector<float> v({1000}, false);  // CPU-only
Matrix<float> m({100, 100}, false);  // CPU-only
```

## Examples

### Solving Linear Systems (Verification)

```cpp
// Verify Ax = b
Matrix<float> A({2, 2});
A[{0, 0}] = 3.0f; A[{0, 1}] = 1.0f;
A[{1, 0}] = 1.0f; A[{1, 1}] = 2.0f;

Vector<float> x({2});
x[{0}] = 2.0f;
x[{1}] = 3.0f;

Vector<float> b = linalg::matvec(A, x);
// b[{0}] should be 9.0, b[{1}] should be 8.0
```

### Matrix Chain Multiplication

```cpp
Matrix<float> A({2, 3});
Matrix<float> B({3, 2});
Matrix<float> C({2, 2});
// ... initialize matrices ...

auto AB = linalg::matmul(A, B);
auto result = linalg::matmul(AB, C);
```

### Working with Sub-matrices

```cpp
Matrix<float> large_mat({1000, 1000});
// ... initialize large_mat ...

// Extract top-left 100x100 block
auto block = TensorSlice<float, 2>::block(large_mat, 0, 100, 0, 100);

// Compute operations on the block
volatile float sum = 0;
for (size_t i = 0; i < 100; ++i) {
    auto row = TensorSlice<float, 2>::row(large_mat, i);
    sum += row[{0}];
}
```

### Unit Vector in Direction

```cpp
Vector<float> direction({3});
direction[{0}] = 1.0f;
direction[{1}] = 2.0f;
direction[{2}] = 2.0f;

auto unit_vec = linalg::normalize(direction);
// unit_vec now has length 1.0 in the same direction
```

## Build Configuration

To enable BLAS support:

```bash
cmake -B build -S . -DUSE_BLAS=ON
```

To enable GPU support:

```bash
cmake -B build -S . -DUSE_GPU=ON
```

Both can be enabled simultaneously for maximum performance.

## Testing

Linear algebra operations are tested in `tests/linalg_test.cc`. Run tests with:

```bash
./build/linalg_test
```

## Performance Benchmarking

Performance benchmarks for linear algebra operations are in `perf/tensor_perf.cc`. Run with:

```bash
./build/tensor_perf
```

This generates CSV files with detailed performance metrics for:
- Vector operations (norm, dot product, outer product)
- Matrix operations (matrix-vector, matrix-matrix multiplication, transpose)
- Tensor views (slicing, row/column/block access)
- GPU-accelerated operations (when available)
- BLAS-accelerated operations (when available)
