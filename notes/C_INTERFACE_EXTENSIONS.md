# C Interface Extensions

## Overview

The C interface has been extended to include advanced features from the C++ tensor library, including:

1. **Advanced Linear Algebra Operations**
2. **Statistical Operations**
3. **Mathematical Functions**
4. **Slicing and Views**
5. **Random Number Generation**
6. **Utility Functions**

## Status

### Fully Implemented

- **Basic Operations**: Vector/Matrix creation, destruction, arithmetic
- **I/O Operations**: Save/load for vectors and matrices
- **Basic Statistics**: mean, variance, std, sum, min, max
- **Optimizer Interface**: SGD and Adam optimizers
- **Some Linear Algebra**: solve, lstsq, pinv, rank, kron, cross product

### Partially Implemented (Need Refinement)

- **Correlation/Covariance**: Functions exist but need result variant handling
- **Normalization**: standardize, normalize methods
- **Mathematical Functions**: exp, log, sqrt, sin, cos, tan, tanh, sigmoid, relu
- **Random Generation**: uniform and normal distributions
- **Slicing**: get_row, get_col, submatrix, slice operations

### Placeholder (Not Yet Implemented)

- **LU Decomposition**
- **QR Decomposition**  
- **Cholesky Decomposition**
- **SVD**
- **Eigenvalue/Eigenvector computation**
- **Triangular solve**

## Header File Location

`include/tensor_c.h` - Complete function declarations

## Implementation File

`src/tensor_c.cpp` - Implementation with error handling

## Usage Example

```c
#include "tensor_c.h"

int main() {
    // Create a matrix
    MatrixFloatHandle A;
    float data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrix_float_create(3, 3, data, &A);
    
    // Create a vector
    VectorFloatHandle b;
    float b_data[3] = {1, 2, 3};
    vector_float_create(3, b_data, &b);
    
    // Solve Ax = b
    VectorFloatHandle x;
    TensorErrorCode status = matrix_float_solve(A, b, &x);
    
    if (status == TENSOR_SUCCESS) {
        // Print solution
        vector_float_print(x);
        vector_float_destroy(x);
    } else {
        printf("Error: %s\n", tensor_c_last_error());
    }
    
    // Cleanup
    matrix_float_destroy(A);
    vector_float_destroy(b);
    
    return 0;
}
```

## Next Steps

1. Fix compilation issues with missing tensor methods (get_row, get_col, submatrix, get_diagonal)
2. Implement decomposition functions (LU, QR, Cholesky, SVD, EIG)
3. Add comprehensive test suite for C interface
4. Update Python wrapper to expose C interface functions
5. Add examples demonstrating each feature category

## Documentation

Full C interface documentation is available in:
- `c_interop.md` - Guide for creating C interfaces
- `userguide/` - User guide with C examples
