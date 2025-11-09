# Python Wrapper Update Summary

## Version 1.4.2

### New Linear Algebra Module

Added comprehensive `tensor4d.linalg` submodule with the following functions:

#### Vector Operations
- `norm(v)` - Euclidean norm
- `normalize(v)` - Normalize to unit length
- `dot(a, b)` - Dot product
- `outer(a, b)` - Outer product

#### Matrix Operations
- `matmul(A, B)` - Matrix multiplication
- `matvec(mat, vec)` - Matrix-vector multiplication
- `transpose(mat)` - Matrix transpose
- `trace(mat)` - Matrix trace
- `diag(mat)` - Extract diagonal from matrix
- `diag(vec)` - Create diagonal matrix from vector
- `eye(n, use_gpu=True)` - Identity matrix

#### Matrix Norms and Properties
- `frobenius_norm(mat)` - Frobenius norm
- `norm_l1(mat)` - L1 matrix norm
- `norm_inf(mat)` - L-infinity matrix norm
- `matrix_rank(mat, tol=-1)` - Matrix rank
- `condition_number(mat)` - Condition number

#### Linear Solvers
- `lstsq(A, b)` - Least squares solution

### Enhanced Matrix Methods

Added to `Matrixf` and `Matrixd` classes:
- `rank(tol=-1)` - Matrix rank
- `trace()` - Matrix trace
- `frobenius_norm()` - Frobenius norm
- `condition_number()` - Condition number

### Enhanced Vector Methods

Added to `Vectorf` and `Vectord` classes:
- `normalize()` - Normalize to unit length

### Updated Examples

- `example_linalg.py` - Comprehensive demonstration of all linear algebra features

### All Functions Support Both float and double Precision

Every function in the `linalg` module has overloads for both:
- `Vectorf` / `Matrixf` (float precision)
- `Vectord` / `Matrixd` (double precision)

### Testing

All new features have been tested and verified working:
- Vector operations (norm, normalize, dot, outer)
- Matrix operations (matmul, transpose, trace, diag)
- Matrix-vector multiplication
- Matrix norms and properties
- Least squares solving
- Identity matrix creation

### Usage Example

```python
import tensor4d as t4d

# Vector operations
v1 = t4d.Vectorf([1.0, 2.0, 3.0])
v2 = t4d.Vectorf([4.0, 5.0, 6.0])
dot_prod = t4d.linalg.dot(v1, v2)
v1_norm = t4d.linalg.norm(v1)
v1_normalized = t4d.linalg.normalize(v1)

# Matrix operations
A = t4d.Matrixf([[1.0, 2.0], [3.0, 4.0]])
B = t4d.Matrixf([[5.0, 6.0], [7.0, 8.0]])
C = t4d.linalg.matmul(A, B)
# or using @ operator
C = A @ B

# Matrix properties
trace = t4d.linalg.trace(A)
rank = t4d.linalg.matrix_rank(A)
frob_norm = t4d.linalg.frobenius_norm(A)

# Identity matrix
I = t4d.linalg.eye(3, use_gpu=False)

# Least squares
A = t4d.Matrixf([[1, 1], [1, 2], [1, 3]])
b = t4d.Vectorf([1, 2, 3])
x = t4d.linalg.lstsq(A, b)
```

### Build Status

✅ Successfully compiled with pybind11
✅ All tests passing
✅ Examples working correctly
✅ Compatible with Python 3.12+

### Files Modified

1. `python/tensor_wrapper.cc` - Added linalg submodule and enhanced bindings
2. `python/example_linalg.py` - Updated with comprehensive examples
3. `include/linalg.h` - Fixed minor bugs (requires_grad typo, determinant issue)

### Version

Python wrapper version updated to 1.4.2 to match library version.
