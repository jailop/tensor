# Tensor4D C API - Implemented Features

This document lists all features implemented in the Tensor4D C API (`tensor_c.h`). The C API provides complete bindings to the high-performance tensor4d C++ library, enabling usage from C, Python (via ctypes/CFFI), and other languages with C FFI support.

## Table of Contents
- [Backend Support](#backend-support)
- [Data Types](#data-types)
- [Vector Operations](#vector-operations)
- [Matrix Operations](#matrix-operations)
- [Advanced Linear Algebra](#advanced-linear-algebra)
- [Statistical Operations](#statistical-operations)
- [Mathematical Functions](#mathematical-functions)
- [Neural Network Layers](#neural-network-layers)
- [Optimizers](#optimizers)
- [I/O Operations](#io-operations)
- [Utility Functions](#utility-functions)

---

## Backend Support

The library automatically selects the best available backend at runtime:

1. **GPU (CUDA)**: Automatically used if compiled with `USE_GPU` and GPU hardware is available
2. **BLAS**: Used if GPU unavailable but compiled with `USE_BLAS`
3. **CPU**: Fallback implementation

### Backend Query Functions
- `tensor_c_is_gpu_available()` - Check if GPU acceleration is available
- `matrix_float_get_backend()` - Get the backend used by a float matrix
- `matrix_double_get_backend()` - Get the backend used by a double matrix
- `tensor_c_backend_name()` - Get backend name as string ("CPU", "BLAS", or "GPU")

### Automatic Backend Selection
No manual configuration needed - all tensor operations automatically use GPU when available, with transparent fallback to BLAS or CPU.

---

## Data Types

### Opaque Handles
- `VectorFloatHandle` - 1D float tensor
- `VectorDoubleHandle` - 1D double tensor
- `MatrixFloatHandle` - 2D float tensor
- `MatrixDoubleHandle` - 2D double tensor
- `LayerHandle` - Neural network layer
- `OptimizerHandle` - Optimizer for training

### Error Handling
- `TensorErrorCode` - Return codes for all functions
  - `TENSOR_SUCCESS`
  - `TENSOR_ERROR_ALLOCATION`
  - `TENSOR_ERROR_SHAPE`
  - `TENSOR_ERROR_INDEX`
  - `TENSOR_ERROR_COMPUTATION`
  - `TENSOR_ERROR_NULL_POINTER`
  - `TENSOR_ERROR_INVALID_OPERATION`
  - `TENSOR_ERROR_FILE_IO`
- `tensor_c_last_error()` - Get detailed error message (thread-local)

---

## Vector Operations

### Creation and Destruction (Float & Double)
- `vector_float_create()` - Create from data array
- `vector_float_zeros()` - Create zero-filled vector
- `vector_float_ones()` - Create one-filled vector
- `vector_float_destroy()` - Free vector memory

### Element Access (Float & Double)
- `vector_float_get()` - Get element at index
- `vector_float_set()` - Set element at index
- `vector_float_size()` - Get vector size

### Arithmetic Operations (Float & Double)
- `vector_float_add()` - Element-wise addition
- `vector_float_subtract()` - Element-wise subtraction
- `vector_float_multiply()` - Element-wise multiplication
- `vector_float_divide()` - Element-wise division
- `vector_float_dot()` - Dot product
- `vector_float_norm()` - L2 norm (Euclidean)

### Statistical Operations (Float & Double)
- `vector_float_mean()` - Mean value
- `vector_float_variance()` - Variance
- `vector_float_std()` - Standard deviation
- `vector_float_sum()` - Sum of all elements
- `vector_float_min()` - Minimum value
- `vector_float_max()` - Maximum value
- `vector_float_median()` - Median value
- `vector_float_quantile()` - Quantile computation

### Advanced Statistics (Float & Double)
- `vector_float_correlation()` - Pearson correlation coefficient
- `vector_float_covariance()` - Covariance
- `vector_float_spearman()` - Spearman rank correlation
- `vector_float_standardize()` - Z-score normalization
- `vector_float_normalize()` - Min-max scaling to [0, 1]

### Mathematical Functions (Float & Double)
- `vector_float_exp()` - Element-wise exponential
- `vector_float_log()` - Element-wise natural logarithm
- `vector_float_sqrt()` - Element-wise square root
- `vector_float_sin()` - Element-wise sine
- `vector_float_cos()` - Element-wise cosine
- `vector_float_tan()` - Element-wise tangent
- `vector_float_tanh()` - Element-wise hyperbolic tangent
- `vector_float_sigmoid()` - Element-wise sigmoid (1/(1+e^-x))
- `vector_float_relu()` - Element-wise ReLU (max(0, x))

### Geometric Operations (Float & Double)
- `vector_float_cross()` - 3D cross product

### Slicing (Float & Double)
- `vector_float_slice()` - Extract subvector [start, end)

### Random Generation (Float & Double)
- `vector_float_random_uniform()` - Uniform distribution [low, high]
- `vector_float_random_normal()` - Normal distribution N(mean, std)

### Utility (Float & Double)
- `vector_float_copy()` - Deep copy
- `vector_float_print()` - Print to stdout
- `vector_float_data()` - Get raw data pointer (read-only)

**Note**: All operations listed above are available for both `float` and `double` types with corresponding `_float` and `_double` suffixes.

---

## Matrix Operations

### Creation and Destruction (Float & Double)
- `matrix_float_create()` - Create from data array (row-major)
- `matrix_float_zeros()` - Create zero matrix
- `matrix_float_ones()` - Create ones matrix
- `matrix_float_eye()` - Create identity matrix
- `matrix_float_destroy()` - Free matrix memory

### Element Access (Float & Double)
- `matrix_float_get()` - Get element at (row, col)
- `matrix_float_set()` - Set element at (row, col)
- `matrix_float_shape()` - Get dimensions (rows, cols)

### Arithmetic Operations (Float & Double)
- `matrix_float_add()` - Element-wise addition (or broadcast)
- `matrix_float_subtract()` - Element-wise subtraction
- `matrix_float_multiply()` - Element-wise multiplication (Hadamard product)
- `matrix_float_matmul()` - Matrix multiplication (A @ B)
- `matrix_float_transpose()` - Transpose

### Linear Algebra (Float & Double)
- `matrix_float_inverse()` - Matrix inverse
- `matrix_float_determinant()` - Determinant
- `matrix_float_trace()` - Trace (sum of diagonal)
- `matrix_float_norm()` - Frobenius norm

### Matrix-Vector Operations (Float & Double)
- `matrix_float_matvec()` - Matrix-vector multiplication

### Statistical Operations (Float & Double)
- `matrix_float_mean()` - Mean of all elements
- `matrix_float_sum()` - Sum of all elements
- `matrix_float_min()` - Minimum value
- `matrix_float_max()` - Maximum value

### Mathematical Functions (Float & Double)
- `matrix_float_exp()` - Element-wise exponential
- `matrix_float_log()` - Element-wise logarithm
- `matrix_float_sqrt()` - Element-wise square root
- `matrix_float_tanh()` - Element-wise hyperbolic tangent
- `matrix_float_sigmoid()` - Element-wise sigmoid
- `matrix_float_relu()` - Element-wise ReLU

### Normalization (Float & Double)
- `matrix_float_standardize()` - Z-score normalization
- `matrix_float_normalize()` - Min-max scaling

### Slicing and Views (Float & Double)
- `matrix_float_get_row()` - Extract row as vector
- `matrix_float_get_col()` - Extract column as vector
- `matrix_float_get_diag()` - Extract diagonal as vector
- `matrix_float_submatrix()` - Extract submatrix block

### Random Generation (Float & Double)
- `matrix_float_random_uniform()` - Uniform distribution
- `matrix_float_random_normal()` - Normal distribution

### Utility (Float & Double)
- `matrix_float_copy()` - Deep copy
- `matrix_float_print()` - Print to stdout
- `matrix_float_data()` - Get raw data pointer (row-major, read-only)
- `matrix_float_reshape()` - Reshape to new dimensions

**Note**: All operations listed above are available for both `float` and `double` types.

---

## Advanced Linear Algebra

All decomposition and solver functions support both `float` and `double` types.

### Matrix Decompositions

#### LU Decomposition
- `matrix_float_lu()` - LU factorization with partial pivoting
  - Returns: Lower triangular L, upper triangular U, and pivot indices

#### QR Decomposition
- `matrix_float_qr()` - QR factorization using Householder reflections
  - Returns: Orthogonal matrix Q and upper triangular R

#### Cholesky Decomposition
- `matrix_float_cholesky()` - Cholesky factorization for positive definite matrices
  - Returns: Lower triangular L where A = L @ L^T

#### SVD (Singular Value Decomposition)
- `matrix_float_svd()` - Full SVD
  - Returns: U (left singular vectors), S (singular values), V^T (right singular vectors)

#### Eigenvalue Decomposition
- `matrix_float_eig()` - Eigenvalue and eigenvector computation
  - Returns: Eigenvalues vector and eigenvectors matrix

### Linear System Solvers

#### General Linear Systems
- `matrix_float_solve()` - Solve Ax = b using LU decomposition
  - Handles general square matrices

#### Triangular Systems
- `matrix_float_solve_triangular()` - Solve triangular system
  - Optimized for upper/lower triangular matrices

#### Least Squares
- `matrix_float_lstsq()` - Least squares solution (overdetermined systems)
  - Finds x that minimizes ||Ax - b||²

### Matrix Properties and Operations

#### Pseudo-Inverse
- `matrix_float_pinv()` - Moore-Penrose pseudo-inverse
  - Computed via SVD

#### Matrix Rank
- `matrix_float_rank()` - Numerical rank computation
  - Uses SVD with tolerance

#### Kronecker Product
- `matrix_float_kron()` - Kronecker tensor product
  - Returns: A ⊗ B

---

## Statistical Operations

### Summary Statistics (Vectors)
- Mean, variance, standard deviation
- Min, max, sum
- Median and quantiles

### Correlation and Covariance (Vectors)
- Pearson correlation coefficient
- Covariance between two vectors
- Spearman rank correlation

### Normalization and Standardization
- **Standardization**: Z-score normalization (mean=0, std=1)
  - Available for vectors and matrices
- **Normalization**: Min-max scaling to [0, 1]
  - Available for vectors and matrices

All statistical functions support both float and double precision.

---

## Mathematical Functions

Element-wise mathematical operations for vectors and matrices:

### Exponential and Logarithmic
- `exp()` - e^x
- `log()` - Natural logarithm
- `sqrt()` - Square root

### Trigonometric
- `sin()` - Sine
- `cos()` - Cosine
- `tan()` - Tangent

### Hyperbolic and Activation Functions
- `tanh()` - Hyperbolic tangent
- `sigmoid()` - Logistic sigmoid: 1/(1+e^-x)
- `relu()` - Rectified Linear Unit: max(0, x)

Available for both vectors and matrices in float and double precision.

---

## Neural Network Layers

All neural network layers support:
- **Automatic GPU acceleration** when available
- Forward and backward passes for training
- Both float and double precision

### Linear (Fully Connected) Layer
- `layer_linear_create_float()` - Create linear layer
- `layer_linear_forward_float()` - Forward pass: output = input @ weights^T + bias
- `layer_linear_backward_float()` - Backward pass for gradient computation
- `layer_linear_get_weights_float()` - Access weight matrix
- `layer_linear_get_bias_float()` - Access bias vector
- `layer_linear_destroy()` - Free layer memory

**Features**:
- Configurable input/output features
- Optional bias term
- Xavier/Glorot weight initialization
- GPU-accelerated matrix multiplications

### ReLU (Rectified Linear Unit)
- `layer_relu_create_float()` - Create ReLU layer
- `layer_relu_forward_float()` - Forward pass: max(0, x)
- `layer_relu_backward_float()` - Backward pass
- `layer_relu_destroy()` - Free layer memory

### Sigmoid Activation
- `layer_sigmoid_create_float()` - Create sigmoid layer
- `layer_sigmoid_forward_float()` - Forward pass: 1/(1+e^-x)
- `layer_sigmoid_backward_float()` - Backward pass
- `layer_sigmoid_destroy()` - Free layer memory

### Softmax Activation
- `layer_softmax_create_float()` - Create softmax layer
- `layer_softmax_forward_float()` - Forward pass: e^x_i / Σe^x_j
- `layer_softmax_backward_float()` - Backward pass
- `layer_softmax_destroy()` - Free layer memory

**Features**:
- Numerically stable implementation
- Row-wise softmax for batch processing

### Dropout Regularization
- `layer_dropout_create_float()` - Create dropout layer with dropout probability p
- `layer_dropout_forward_float()` - Forward pass (drops random neurons during training)
- `layer_dropout_backward_float()` - Backward pass
- `layer_dropout_train()` - Set training/inference mode
- `layer_dropout_destroy()` - Free layer memory

**Features**:
- Configurable dropout rate
- Training/inference mode switching
- Inverted dropout (scales during training)

### Batch Normalization (1D)
- `layer_batchnorm_create_float()` - Create batch normalization layer
- `layer_batchnorm_forward_float()` - Forward pass: normalizes along batch dimension
- `layer_batchnorm_backward_float()` - Backward pass
- `layer_batchnorm_train()` - Set training/inference mode
- `layer_batchnorm_destroy()` - Free layer memory

**Features**:
- Configurable epsilon and momentum
- Running mean/variance tracking
- Learnable scale (gamma) and shift (beta) parameters
- Training/inference mode for proper normalization

**Note**: All layer operations are available for both `float` and `double` types.

---

## Optimizers

Optimizers for training neural networks. Support both float and double precision.

### SGD (Stochastic Gradient Descent)
- `optimizer_sgd_create()` - Create SGD optimizer
  - Parameters: learning_rate, momentum
- `optimizer_sgd_step()` - Update parameters using computed gradients
- `optimizer_sgd_zero_grad()` - Clear parameter gradients
- `optimizer_sgd_destroy()` - Free optimizer memory

**Features**:
- Configurable learning rate
- Optional momentum for faster convergence
- Weight decay support

### Adam Optimizer
- `optimizer_adam_create()` - Create Adam optimizer
  - Parameters: learning_rate, beta1, beta2, epsilon
- `optimizer_adam_step()` - Update parameters using adaptive learning rates
- `optimizer_adam_zero_grad()` - Clear parameter gradients
- `optimizer_adam_destroy()` - Free optimizer memory

**Features**:
- Adaptive learning rates per parameter
- Momentum estimation (beta1)
- Variance estimation (beta2)
- Numerical stability (epsilon)
- Efficient for large-scale problems

**Note**: Parameters must be provided when creating the optimizer. The current design doesn't support dynamic parameter addition after creation.

---

## I/O Operations

Binary serialization for saving and loading tensors.

### Vector I/O (Float & Double)
- `vector_float_save()` - Save vector to binary file
- `vector_float_load()` - Load vector from binary file

### Matrix I/O (Float & Double)
- `matrix_float_save()` - Save matrix to binary file
- `matrix_float_load()` - Load matrix from binary file

**Features**:
- Binary format for efficiency
- Includes metadata (dimensions, data type)
- Cross-platform compatible
- Supports both float and double precision

---

## Utility Functions

### Information and Versioning
- `tensor_c_version()` - Get library version string
- `tensor_c_last_error()` - Get last error message (thread-local)

### Memory Management
- All `*_destroy()` functions properly free allocated memory
- Exception-safe implementations using RAII in C++ backend

### Debugging
- `vector_float_print()` - Print vector to stdout
- `matrix_float_print()` - Print matrix to stdout

### Data Access
- `vector_float_data()` - Get raw data pointer (read-only)
- `matrix_float_data()` - Get raw data pointer (row-major, read-only)

---

## Implementation Notes

### Thread Safety
- All operations are thread-safe
- Error messages are stored in thread-local storage
- Multiple tensors can be operated on concurrently

### Memory Model
- **Opaque handles**: All tensor objects are referenced through opaque pointers
- **Ownership**: Calling code owns returned tensors and must free them
- **Copy semantics**: Operations return new tensors (no in-place modifications by default)

### Performance Characteristics

#### Backend Performance Hierarchy
1. **GPU**: ~10-100x faster for large matrices (>1024×1024)
2. **BLAS**: ~5-20x faster than naive CPU for medium/large matrices
3. **CPU**: Optimized with modern C++ features, adequate for small matrices

#### Optimization Features
- **Parallel execution**: Multi-threaded operations via TBB when available
- **SIMD**: Vectorized operations on CPU
- **Cache-friendly**: Row-major layout, blocking for cache efficiency
- **GPU offloading**: Automatic for compatible operations

### C++ Backend Integration
The C API is a thin wrapper around the high-performance C++ `tensor4d` library:
- Templates instantiated for `float` and `double`
- Exception handling converted to error codes
- All memory management uses RAII internally for safety
- Zero-copy operations where possible

---

## Usage Example

```c
#include "tensor_c.h"
#include <stdio.h>

int main() {
    // Check if GPU is available
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
    
    // Get backend info
    TensorBackend backend;
    matrix_float_get_backend(C, &backend);
    printf("Using backend: %s\n", tensor_c_backend_name(backend));
    
    // Clean up
    matrix_float_destroy(A);
    matrix_float_destroy(B);
    matrix_float_destroy(C);
    
    return 0;
}
```

### Neural Network Example

```c
#include "tensor_c.h"

int main() {
    // Create a simple neural network
    LayerHandle linear1, relu, linear2, softmax;
    
    // Network: 784 -> 128 -> 10 (MNIST classifier)
    layer_linear_create_float(784, 128, true, &linear1);
    layer_relu_create_float(&relu);
    layer_linear_create_float(128, 10, true, &linear2);
    layer_softmax_create_float(&softmax);
    
    // Create input (batch_size=32, features=784)
    MatrixFloatHandle input;
    matrix_float_random_normal(32, 784, 0.0f, 1.0f, &input);
    
    // Forward pass (automatically uses GPU if available)
    MatrixFloatHandle h1, h2, h3, output;
    layer_linear_forward_float(linear1, input, &h1);
    layer_relu_forward_float(relu, h1, &h2);
    layer_linear_forward_float(linear2, h2, &h3);
    layer_softmax_forward_float(softmax, h3, &output);
    
    matrix_float_print(output);
    
    // Clean up
    matrix_float_destroy(input);
    matrix_float_destroy(h1);
    matrix_float_destroy(h2);
    matrix_float_destroy(h3);
    matrix_float_destroy(output);
    
    layer_linear_destroy(linear1);
    layer_relu_destroy(relu);
    layer_linear_destroy(linear2);
    layer_softmax_destroy(softmax);
    
    return 0;
}
```

---

## Summary

The Tensor4D C API provides:

- ✅ **256 functions** covering comprehensive tensor operations
- ✅ **Automatic GPU acceleration** - no manual configuration needed
- ✅ **Both float and double precision** for all operations
- ✅ **Complete linear algebra suite** - decompositions, solvers, matrix operations
- ✅ **Statistical operations** - correlations, normalizations, descriptive statistics
- ✅ **Neural network layers** - Linear, ReLU, Sigmoid, Softmax, Dropout, BatchNorm
- ✅ **Optimizers** - SGD and Adam for training
- ✅ **I/O operations** - binary serialization
- ✅ **Thread-safe** - safe for concurrent use
- ✅ **Production-ready** - comprehensive error handling and memory safety

The C API enables using the high-performance tensor4d library from any language with C FFI support, including Python (via ctypes/CFFI), Go, Rust, Julia, and more.
