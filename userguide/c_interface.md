
## Advanced Features in C Interface

The C interface now includes many advanced features beyond basic vector and matrix operations.

### Linear Algebra Operations

#### Solving Linear Systems

```c
// Solve Ax = b
MatrixFloatHandle A;
VectorFloatHandle b, x;

// ... create and populate A and b ...

TensorErrorCode status = matrix_float_solve(A, b, &x);
if (status == TENSOR_SUCCESS) {
    printf("Solution found\n");
    vector_float_print(x);
}
```

#### Least Squares

```c
// Solve overdetermined system (more equations than unknowns)
MatrixFloatHandle A;  // m x n matrix, m > n
VectorFloatHandle b, x;

status = matrix_float_lstsq(A, b, &x);
```

#### Pseudo-Inverse

```c
// Compute Moore-Penrose pseudo-inverse
MatrixFloatHandle A, A_pinv;

status = matrix_float_pinv(A, &A_pinv);
```

#### Matrix Rank

```c
// Get numerical rank of a matrix
size_t rank;
status = matrix_float_rank(A, &rank);
printf("Matrix rank: %zu\n", rank);
```

#### Kronecker Product

```c
// Compute Kronecker product of two matrices
MatrixFloatHandle A, B, C;
status = matrix_float_kron(A, B, &C);
```

#### Cross Product (3D vectors only)

```c
// Compute cross product of 3D vectors
VectorFloatHandle v1, v2, result;

// ... create 3-element vectors ...

status = vector_float_cross(v1, v2, &result);
```

### Statistical Operations

#### Correlation

```c
// Pearson correlation coefficient
VectorFloatHandle x, y;
float corr;

status = vector_float_correlation(x, y, &corr);
printf("Correlation: %f\n", corr);
```

#### Covariance

```c
// Covariance between two vectors
float cov;
status = vector_float_covariance(x, y, &cov);
```

#### Spearman Rank Correlation

```c
// Spearman rank correlation
float spearman;
status = vector_float_spearman(x, y, &spearman);
```

#### Median and Quantiles

```c
// Compute median
float median;
status = vector_float_median(vec, &median);

// Compute arbitrary quantile (e.g., 75th percentile)
float q75;
status = vector_float_quantile(vec, 0.75f, &q75);
```

#### Standardization and Normalization

```c
// Z-score standardization (mean=0, std=1)
VectorFloatHandle vec, standardized;
status = vector_float_standardize(vec, &standardized);

// Min-max normalization (scale to [0, 1])
VectorFloatHandle normalized;
status = vector_float_normalize(vec, &normalized);

// For matrices, specify axis:
MatrixFloatHandle mat, mat_std;
int axis = 0;  // standardize along rows
status = matrix_float_standardize(mat, axis, &mat_std);
```

### Mathematical Functions

Apply element-wise math functions to vectors and matrices:

```c
VectorFloatHandle vec, result;

// Exponential
status = vector_float_exp(vec, &result);

// Natural logarithm
status = vector_float_log(vec, &result);

// Square root
status = vector_float_sqrt(vec, &result);

// Trigonometric
status = vector_float_sin(vec, &result);
status = vector_float_cos(vec, &result);
status = vector_float_tan(vec, &result);

// Hyperbolic tangent
status = vector_float_tanh(vec, &result);

// Sigmoid activation
status = vector_float_sigmoid(vec, &result);

// ReLU activation
status = vector_float_relu(vec, &result);
```

All of these work similarly for matrices:

```c
MatrixFloatHandle mat, result;
status = matrix_float_exp(mat, &result);
status = matrix_float_sigmoid(mat, &result);
// ... etc
```

### Slicing and Views

```c
// Get a specific row from a matrix
MatrixFloatHandle mat;
VectorFloatHandle row;
size_t row_index = 2;
status = matrix_float_get_row(mat, row_index, &row);

// Get a specific column
VectorFloatHandle col;
size_t col_index = 1;
status = matrix_float_get_col(mat, col_index, &col);

// Get diagonal elements
VectorFloatHandle diag;
status = matrix_float_get_diag(mat, &diag);

// Extract submatrix
MatrixFloatHandle submat;
status = matrix_float_submatrix(mat, 
    0, 2,   // rows 0 to 2
    1, 3,   // cols 1 to 3
    &submat);

// Slice a vector
VectorFloatHandle vec, slice;
status = vector_float_slice(vec, 5, 10, &slice);  // elements [5, 10)
```

### Random Number Generation

```c
// Create vector with uniform random values
VectorFloatHandle random_vec;
size_t size = 100;
float low = 0.0f, high = 1.0f;
status = vector_float_random_uniform(size, low, high, &random_vec);

// Create matrix with normal (Gaussian) random values
MatrixFloatHandle random_mat;
size_t rows = 10, cols = 20;
float mean = 0.0f, std = 1.0f;
status = matrix_float_random_normal(rows, cols, mean, std, &random_mat);
```

### Utility Functions

```c
// Copy a vector or matrix
VectorFloatHandle vec, vec_copy;
status = vector_float_copy(vec, &vec_copy);

MatrixFloatHandle mat, mat_copy;
status = matrix_float_copy(mat, &mat_copy);

// Print to stdout (for debugging)
status = vector_float_print(vec);
status = matrix_float_print(mat);

// Get raw data pointer (read-only)
const float* data_ptr;
status = vector_float_data(vec, &data_ptr);
// Use data_ptr for interfacing with other libraries

// Reshape matrix
MatrixFloatHandle reshaped;
status = matrix_float_reshape(mat, new_rows, new_cols, &reshaped);
```

### Device Management

```c
// Set computation device (CPU or GPU)
status = tensor_c_set_device(TENSOR_DEVICE_GPU);

// Query current device
TensorDevice device;
status = tensor_c_get_device(&device);
if (device == TENSOR_DEVICE_GPU) {
    printf("Using GPU\n");
} else {
    printf("Using CPU\n");
}
```

### Error Handling

Always check return codes and use `tensor_c_last_error()` for detailed messages:

```c
TensorErrorCode status = vector_float_create(size, data, &vec);

if (status != TENSOR_SUCCESS) {
    fprintf(stderr, "Error code %d: %s\n", status, tensor_c_last_error());
    return -1;
}

// Or use a switch for specific handling
switch (status) {
    case TENSOR_SUCCESS:
        break;
    case TENSOR_ERROR_ALLOCATION:
        fprintf(stderr, "Memory allocation failed\n");
        break;
    case TENSOR_ERROR_SHAPE:
        fprintf(stderr, "Shape mismatch\n");
        break;
    case TENSOR_ERROR_INDEX:
        fprintf(stderr, "Index out of bounds\n");
        break;
    case TENSOR_ERROR_COMPUTATION:
        fprintf(stderr, "Computational error: %s\n", tensor_c_last_error());
        break;
    default:
        fprintf(stderr, "Unknown error\n");
}
```

### Library Version

```c
const char* version = tensor_c_version();
printf("Tensor library version: %s\n", version);
```

## Complete Example: Linear Regression

Here's a complete example using the C interface to perform linear regression:

```c
#include "tensor_c.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Generate some sample data: y = 2*x + 1 + noise
    size_t n = 100;
    
    // X matrix (n x 2): column of ones and x values
    MatrixFloatHandle X;
    VectorFloatHandle y;
    
    float* X_data = malloc(n * 2 * sizeof(float));
    float* y_data = malloc(n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        float x = (float)i / n;
        X_data[i * 2] = 1.0f;      // intercept
        X_data[i * 2 + 1] = x;      // x value
        y_data[i] = 2.0f * x + 1.0f + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
    
    matrix_float_create(n, 2, X_data, &X);
    vector_float_create(n, y_data, &y);
    
    // Solve least squares: beta = (X^T X)^{-1} X^T y
    VectorFloatHandle beta;
    TensorErrorCode status = matrix_float_lstsq(X, y, &beta);
    
    if (status == TENSOR_SUCCESS) {
        printf("Linear regression coefficients:\n");
        vector_float_print(beta);
        
        // Get the actual values
        float intercept, slope;
        vector_float_get(beta, 0, &intercept);
        vector_float_get(beta, 1, &slope);
        
        printf("\ny = %.4f + %.4f * x\n", intercept, slope);
        printf("(Expected: y = 1.0 + 2.0 * x)\n");
        
        vector_float_destroy(beta);
    } else {
        fprintf(stderr, "Error: %s\n", tensor_c_last_error());
    }
    
    // Cleanup
    matrix_float_destroy(X);
    vector_float_destroy(y);
    free(X_data);
    free(y_data);
    
    return 0;
}
```

## Notes on Implementation Status

Some advanced features are declared in the header but not fully implemented yet:
- LU, QR, Cholesky decompositions
- SVD
- Eigenvalue/eigenvector computation

These will return `TENSOR_ERROR_INVALID_OPERATION` with an appropriate error message.

For the most up-to-date status, see `C_INTERFACE_EXTENSIONS.md`.
