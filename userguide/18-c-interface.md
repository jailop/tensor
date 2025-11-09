# C Interface

The tensor library provides a C interface for integration with C codebases. Since C doesn't support templates, classes, or operator overloading, the interface uses opaque handles and explicit function calls.

## Quick Reference

### Common Operations

| Operation | Function | Example |
|-----------|----------|---------|
| **Vector Creation** | `vector_float_create(size, data, &v)` | Create from array |
| | `vector_float_zeros(size, &v)` | All zeros |
| | `vector_float_ones(size, &v)` | All ones |
| **Matrix Creation** | `matrix_float_create(rows, cols, data, &m)` | Create from array |
| | `matrix_float_eye(n, &m)` | Identity matrix |
| **Arithmetic** | `vector_float_add(a, b, &result)` | Vector addition |
| | `matrix_float_matmul(a, b, &result)` | Matrix multiplication |
| | `vector_float_dot(a, b, &result)` | Dot product |
| **Linear Algebra** | `matrix_float_inverse(m, &inv)` | Matrix inverse |
| | `matrix_float_transpose(m, &trans)` | Transpose |
| | `matrix_float_determinant(m, &det)` | Determinant |
| | `matrix_float_solve(A, b, &x)` | Solve Ax=b |
| **Decompositions** | `matrix_float_lu(A, &L, &U, &p, &ps)` | LU decomposition |
| | `matrix_float_qr(A, &Q, &R)` | QR decomposition |
| | `matrix_float_svd(A, &U, &S, &Vt)` | SVD decomposition |
| **Statistics** | `vector_float_mean(v, &result)` | Mean value |
| | `vector_float_variance(v, &result)` | Variance |
| | `vector_float_correlation(a, b, &r)` | Correlation |
| **Cleanup** | `vector_float_destroy(v)` | Free vector |
| | `matrix_float_destroy(m)` | Free matrix |

### Error Handling

All functions return `TensorErrorCode`:
- `TENSOR_SUCCESS` (0): Operation succeeded
- `TENSOR_ERROR_ALLOCATION`: Memory allocation failed
- `TENSOR_ERROR_SHAPE`: Shape mismatch
- `TENSOR_ERROR_INDEX`: Invalid index
- `TENSOR_ERROR_COMPUTATION`: Numerical error
- `TENSOR_ERROR_NULL_POINTER`: Null pointer argument
- `TENSOR_ERROR_INVALID_OPERATION`: Unsupported operation

Always check return codes and destroy handles when done!

## Overview

The C interface is designed with the following principles:

1. **Opaque Handles**: C++ objects are represented as void pointers
2. **Explicit Type Functions**: Separate functions for each tensor type (float, double, int)
3. **Error Handling**: Return error codes instead of exceptions
4. **Memory Management**: Explicit create/destroy functions
5. **C Linkage**: Uses `extern "C"` to prevent name mangling

## Getting Started

### Include the Header

```c
#include "tensor_c.h"
```

### Error Codes

All functions return a `TensorErrorCode`:

```c
typedef enum {
    TENSOR_SUCCESS = 0,
    TENSOR_ERROR_ALLOCATION = 1,
    TENSOR_ERROR_SHAPE = 2,
    TENSOR_ERROR_INDEX = 3,
    TENSOR_ERROR_COMPUTATION = 4,
    TENSOR_ERROR_NULL_POINTER = 5,
    TENSOR_ERROR_INVALID_OPERATION = 6
} TensorErrorCode;
```

### Handle Types

```c
typedef void* TensorFloatHandle;
typedef void* TensorDoubleHandle;
typedef void* MatrixFloatHandle;
typedef void* VectorFloatHandle;
typedef void* OptimizerHandle;
typedef void* LossFunctionHandle;
```

## Basic Operations

### Creating Tensors

#### From Array Data

```c
TensorFloatHandle tensor;
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
size_t shape[] = {2, 2};

TensorErrorCode err = tensor_float_create_2d(shape, data, &tensor);
if (err != TENSOR_SUCCESS) {
    fprintf(stderr, "Error creating tensor: %d\n", err);
    return 1;
}

// Always destroy when done
tensor_float_destroy(tensor);
```

#### Zeros and Ones

```c
TensorFloatHandle zeros, ones;
size_t shape[] = {3, 4};

tensor_float_zeros(shape, 2, &zeros);
tensor_float_ones(shape, 2, &ones);

tensor_float_destroy(zeros);
tensor_float_destroy(ones);
```

#### Random Tensors

```c
TensorFloatHandle random;
size_t shape[] = {10, 10};

// Uniform distribution [0, 1)
tensor_float_random_uniform(shape, 2, &random);

// Normal distribution N(0, 1)
tensor_float_random_normal(shape, 2, &random);

tensor_float_destroy(random);
```

### Accessing Elements

```c
TensorFloatHandle tensor;
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
size_t shape[] = {2, 2};
tensor_float_create_2d(shape, data, &tensor);

// Get element at [1, 0]
float value;
size_t indices[] = {1, 0};
tensor_float_get(tensor, indices, 2, &value);
printf("Value at [1,0]: %f\n", value);

// Set element at [0, 1]
tensor_float_set(tensor, (size_t[]){0, 1}, 2, 5.0f);

tensor_float_destroy(tensor);
```

### Shape Information

```c
TensorFloatHandle tensor;
size_t shape[10];
size_t ndim;

// Get shape
tensor_float_shape(tensor, shape, &ndim);
printf("Shape: [");
for (size_t i = 0; i < ndim; i++) {
    printf("%zu%s", shape[i], (i < ndim-1) ? ", " : "");
}
printf("]\n");

// Get number of elements
size_t size;
tensor_float_size(tensor, &size);
printf("Total elements: %zu\n", size);
```

## Arithmetic Operations

### Element-wise Operations

```c
TensorFloatHandle A, B, C;
float dataA[] = {1.0f, 2.0f, 3.0f, 4.0f};
float dataB[] = {5.0f, 6.0f, 7.0f, 8.0f};
size_t shape[] = {2, 2};

tensor_float_create_2d(shape, dataA, &A);
tensor_float_create_2d(shape, dataB, &B);

// Addition: C = A + B
tensor_float_add(A, B, &C);

// Subtraction: C = A - B
tensor_float_subtract(A, B, &C);

// Multiplication (element-wise): C = A * B
tensor_float_multiply(A, B, &C);

// Division: C = A / B
tensor_float_divide(A, B, &C);

// Cleanup
tensor_float_destroy(A);
tensor_float_destroy(B);
tensor_float_destroy(C);
```

### In-place Operations

```c
TensorFloatHandle A, B;
// ... create A and B ...

// Modifies A in-place
tensor_float_add_inplace(A, B);  // A += B
tensor_float_sub_inplace(A, B);  // A -= B
tensor_float_mul_inplace(A, B);  // A *= B
tensor_float_div_inplace(A, B);  // A /= B

tensor_float_destroy(A);
tensor_float_destroy(B);
```

### Scalar Operations

```c
TensorFloatHandle A, result;
// ... create A ...

// Scalar addition
tensor_float_add_scalar(A, 5.0f, &result);

// Scalar multiplication
tensor_float_multiply_scalar(A, 2.0f, &result);

tensor_float_destroy(A);
tensor_float_destroy(result);
```

## Mathematical Functions

### Element-wise Math Functions

```c
TensorFloatHandle A, result;
// ... create A ...

tensor_float_exp(A, &result);      // e^x
tensor_float_log(A, &result);      // ln(x)
tensor_float_sqrt(A, &result);     // √x
tensor_float_abs(A, &result);      // |x|
tensor_float_pow(A, 2.0f, &result); // x^2

// Trigonometric
tensor_float_sin(A, &result);
tensor_float_cos(A, &result);
tensor_float_tan(A, &result);

// Activation functions
tensor_float_sigmoid(A, &result);
tensor_float_tanh(A, &result);
tensor_float_relu(A, &result);
tensor_float_leaky_relu(A, 0.01f, &result);

tensor_float_destroy(A);
tensor_float_destroy(result);
```

## Linear Algebra

### Matrix Operations

```c
MatrixFloatHandle A, B, C;
float dataA[] = {1.0f, 2.0f, 3.0f, 4.0f};
float dataB[] = {5.0f, 6.0f, 7.0f, 8.0f};
size_t shape[] = {2, 2};

matrix_float_create(shape[0], shape[1], dataA, &A);
matrix_float_create(shape[0], shape[1], dataB, &B);

// Matrix multiplication
matrix_float_matmul(A, B, &C);

// Transpose
MatrixFloatHandle At;
matrix_float_transpose(A, &At);

// Inverse
MatrixFloatHandle Ainv;
TensorErrorCode err = matrix_float_inverse(A, &Ainv);
if (err != TENSOR_SUCCESS) {
    fprintf(stderr, "Matrix is singular\n");
}

// Determinant
float det;
matrix_float_determinant(A, &det);
printf("Determinant: %f\n", det);

// Cleanup
matrix_float_destroy(A);
matrix_float_destroy(B);
matrix_float_destroy(C);
matrix_float_destroy(At);
matrix_float_destroy(Ainv);
```

### Vector Operations

```c
VectorFloatHandle v1, v2;
float data1[] = {1.0f, 2.0f, 3.0f};
float data2[] = {4.0f, 5.0f, 6.0f};

vector_float_create(3, data1, &v1);
vector_float_create(3, data2, &v2);

// Dot product
float dot;
vector_float_dot(v1, v2, &dot);
printf("Dot product: %f\n", dot);

// Cross product (3D only)
VectorFloatHandle cross;
vector_float_cross(v1, v2, &cross);

// L2 norm
float norm;
vector_float_norm_l2(v1, &norm);
printf("L2 norm: %f\n", norm);

// Cleanup
vector_float_destroy(v1);
vector_float_destroy(v2);
vector_float_destroy(cross);
```

### Decompositions

The library supports various matrix factorizations:

```c
MatrixFloatHandle A, U, S, Vt;
// ... create A ...

// SVD: A = U * S * Vt
TensorErrorCode err = matrix_float_svd(A, &U, &S, &Vt);
if (err == TENSOR_SUCCESS) {
    // U contains left singular vectors
    // S contains singular values (as vector)
    // Vt contains right singular vectors (transposed)
    
    matrix_float_destroy(U);
    vector_float_destroy(S);
    matrix_float_destroy(Vt);
}

// QR decomposition: A = Q * R
MatrixFloatHandle Q, R;
err = matrix_float_qr(A, &Q, &R);
if (err == TENSOR_SUCCESS) {
    // Q is orthogonal matrix
    // R is upper triangular
    
    matrix_float_destroy(Q);
    matrix_float_destroy(R);
}

// LU decomposition: A = P * L * U (with pivoting)
MatrixFloatHandle L, U_lu;
size_t* pivot;
size_t pivot_size;
err = matrix_float_lu(A, &L, &U_lu, &pivot, &pivot_size);
if (err == TENSOR_SUCCESS) {
    // L is lower triangular (with unit diagonal)
    // U_lu is upper triangular
    // pivot contains the permutation information
    
    matrix_float_destroy(L);
    matrix_float_destroy(U_lu);
    free(pivot);  // Remember to free the pivot array
}

// Cholesky decomposition (for SPD matrices): A = L * L^T
MatrixFloatHandle L_chol;
err = matrix_float_cholesky(A, &L_chol);
if (err == TENSOR_SUCCESS) {
    // L_chol is lower triangular
    // A must be symmetric positive definite
    
    matrix_float_destroy(L_chol);
}

// Eigenvalue decomposition (for symmetric matrices)
VectorFloatHandle eigenvalues;
MatrixFloatHandle eigenvectors;
err = matrix_float_eig(A, &eigenvalues, &eigenvectors);
if (err == TENSOR_SUCCESS) {
    // eigenvalues contains the eigenvalues
    // eigenvectors contains the corresponding eigenvectors as columns
    
    vector_float_destroy(eigenvalues);
    matrix_float_destroy(eigenvectors);
}

// Compute matrix rank
size_t rank;
err = matrix_float_rank(A, &rank);
if (err == TENSOR_SUCCESS) {
    printf("Matrix rank: %zu\n", rank);
}

// Cleanup
matrix_float_destroy(A);
```

**Note**: The availability of these decompositions depends on the backend:
- **CPU with BLAS/LAPACK**: All decompositions are supported
- **GPU**: Support varies based on cuBLAS/cuSOLVER availability
- **Pure CPU**: Limited support (some may not be available)

Always check the return code to handle cases where a decomposition is not available.

### Solvers

The library provides various methods for solving linear systems:

```c
MatrixFloatHandle A;
VectorFloatHandle b, x;
// ... create A and b ...

// General linear system solver (uses appropriate method based on matrix properties)
TensorErrorCode err = matrix_float_solve(A, b, &x);
if (err == TENSOR_SUCCESS) {
    // x now contains the solution to Ax = b
    vector_float_destroy(x);
}

// Solve using LU decomposition (direct method)
err = vector_float_solve_lu(A, b, &x);
if (err == TENSOR_SUCCESS) {
    vector_float_destroy(x);
}

// Solve using QR decomposition (more stable for rank-deficient systems)
err = vector_float_solve_qr(A, b, &x);
if (err == TENSOR_SUCCESS) {
    vector_float_destroy(x);
}

// Least squares solution (for overdetermined systems)
err = vector_float_lstsq(A, b, &x);
if (err == TENSOR_SUCCESS) {
    // x contains the least squares solution minimizing ||Ax - b||
    vector_float_destroy(x);
}

// Cleanup
matrix_float_destroy(A);
vector_float_destroy(b);
```

**Performance Tips**:
- Use `matrix_float_solve()` for general systems (it auto-selects the best method)
- Use `vector_float_solve_lu()` for well-conditioned square systems
- Use `vector_float_solve_qr()` for better numerical stability
- Use `vector_float_lstsq()` for overdetermined systems (more equations than unknowns)

## Statistical Operations

### Reductions

```c
TensorFloatHandle tensor;
// ... create tensor ...

// Compute statistics
float sum, mean, variance, std_dev, min_val, max_val;

tensor_float_sum(tensor, &sum);
tensor_float_mean(tensor, &mean);
tensor_float_variance(tensor, &variance);
tensor_float_std(tensor, &std_dev);
tensor_float_min(tensor, &min_val);
tensor_float_max(tensor, &max_val);

printf("Statistics:\n");
printf("  Sum: %f\n", sum);
printf("  Mean: %f\n", mean);
printf("  Variance: %f\n", variance);
printf("  Std Dev: %f\n", std_dev);
printf("  Min: %f\n", min_val);
printf("  Max: %f\n", max_val);

tensor_float_destroy(tensor);
```

### Axis-specific Reductions

```c
TensorFloatHandle tensor, result;
size_t shape[] = {100, 20};
// ... create tensor with shape [100, 20] ...

// Mean along axis 0 (column means)
tensor_float_mean_axis(tensor, 0, &result);

// Sum along axis 1 (row sums)
tensor_float_sum_axis(tensor, 1, &result);

tensor_float_destroy(tensor);
tensor_float_destroy(result);
```

## Autograd

### Basic Gradient Computation

```c
TensorFloatHandle x, y, z;
float data[] = {2.0f, 3.0f};

// Create input with gradient tracking
vector_float_create(2, data, &x);
tensor_float_requires_grad(x, true);

// Forward pass
tensor_float_multiply(x, x, &y);  // y = x²
tensor_float_sum(y, &z);           // z = sum(x²)

// Backward pass
tensor_float_backward(z);

// Get gradient
TensorFloatHandle grad;
tensor_float_grad(x, &grad);

// Print gradients
float grad_values[2];
tensor_float_get_data(grad, grad_values, 2);
printf("Gradients: [%f, %f]\n", grad_values[0], grad_values[1]);

// Cleanup
tensor_float_destroy(x);
tensor_float_destroy(y);
tensor_float_destroy(z);
tensor_float_destroy(grad);
```

### Zero Gradients

```c
TensorFloatHandle tensor;
// ... create tensor with requires_grad=true ...

// Clear gradients
tensor_float_zero_grad(tensor);

tensor_float_destroy(tensor);
```

## Machine Learning

### Loss Functions

```c
LossFunctionHandle loss;
TensorFloatHandle predictions, targets, loss_value;
// ... create predictions and targets ...

// Mean Squared Error
loss_function_mse_create(&loss);
loss_function_forward(loss, predictions, targets, &loss_value);

// Cross Entropy
loss_function_cross_entropy_create(&loss);
loss_function_forward(loss, predictions, targets, &loss_value);

// Binary Cross Entropy
loss_function_binary_cross_entropy_create(&loss);
loss_function_forward(loss, predictions, targets, &loss_value);

// Get loss value
float loss_scalar;
tensor_float_item(loss_value, &loss_scalar);
printf("Loss: %f\n", loss_scalar);

// Cleanup
loss_function_destroy(loss);
tensor_float_destroy(predictions);
tensor_float_destroy(targets);
tensor_float_destroy(loss_value);
```

### Optimizers

```c
OptimizerHandle optimizer;
TensorFloatHandle W, b;
// ... create parameters W and b with requires_grad=true ...

// Create optimizer
TensorFloatHandle params[] = {W, b};
optimizer_adam_create(0.001f, 0.9f, 0.999f, &optimizer);
optimizer_add_parameters(optimizer, params, 2);

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    // Forward pass
    // ... compute loss ...
    
    // Backward pass
    optimizer_zero_grad(optimizer);
    tensor_float_backward(loss);
    
    // Update parameters
    optimizer_step(optimizer);
    
    if (epoch % 10 == 0) {
        float loss_val;
        tensor_float_item(loss, &loss_val);
        printf("Epoch %d, Loss: %f\n", epoch, loss_val);
    }
}

// Cleanup
optimizer_destroy(optimizer);
tensor_float_destroy(W);
tensor_float_destroy(b);
```

### Optimizer Types

```c
OptimizerHandle sgd, adam, adamw, rmsprop;
TensorFloatHandle params[] = {W, b};

// SGD with momentum
optimizer_sgd_create(0.01f, 0.9f, &sgd);

// Adam
optimizer_adam_create(0.001f, 0.9f, 0.999f, &adam);

// AdamW (with weight decay)
optimizer_adamw_create(0.001f, 0.9f, 0.999f, 0.01f, &adamw);

// RMSprop
optimizer_rmsprop_create(0.001f, 0.99f, 1e-8f, &rmsprop);

// Add parameters to optimizer
optimizer_add_parameters(sgd, params, 2);

optimizer_destroy(sgd);
```

## I/O Operations

### Saving and Loading

```c
TensorFloatHandle tensor, loaded;
// ... create tensor ...

// Save to binary file
TensorErrorCode err = tensor_float_save(tensor, "tensor.bin");
if (err != TENSOR_SUCCESS) {
    fprintf(stderr, "Error saving tensor\n");
}

// Load from file
err = tensor_float_load("tensor.bin", &loaded);
if (err != TENSOR_SUCCESS) {
    fprintf(stderr, "Error loading tensor\n");
}

tensor_float_destroy(tensor);
tensor_float_destroy(loaded);
```

### NumPy Format

```c
TensorFloatHandle tensor;
// ... create tensor ...

// Save to .npy format (compatible with NumPy)
tensor_float_save_npy(tensor, "tensor.npy");

// Load from .npy
TensorFloatHandle from_npy;
tensor_float_load_npy("tensor.npy", &from_npy);

tensor_float_destroy(tensor);
tensor_float_destroy(from_npy);
```

### CSV Format

```c
TensorFloatHandle matrix;
// ... create 2D tensor ...

// Save as CSV
matrix_float_save_csv(matrix, "data.csv");

// Load from CSV
MatrixFloatHandle from_csv;
matrix_float_load_csv("data.csv", &from_csv);

matrix_float_destroy(matrix);
matrix_float_destroy(from_csv);
```

## Shape Manipulation

### Reshape and Transpose

```c
TensorFloatHandle tensor, reshaped, transposed;
// ... create tensor with shape [24] ...

// Reshape to [4, 6]
size_t new_shape[] = {4, 6};
tensor_float_reshape(tensor, new_shape, 2, &reshaped);

// Transpose
tensor_float_transpose(reshaped, &transposed);

tensor_float_destroy(tensor);
tensor_float_destroy(reshaped);
tensor_float_destroy(transposed);
```

### Squeeze and Unsqueeze

```c
TensorFloatHandle tensor, squeezed, unsqueezed;
// ... create tensor ...

// Remove dimensions of size 1
tensor_float_squeeze(tensor, &squeezed);

// Add dimension at position 0
tensor_float_unsqueeze(tensor, 0, &unsqueezed);

tensor_float_destroy(tensor);
tensor_float_destroy(squeezed);
tensor_float_destroy(unsqueezed);
```

## Memory Management Best Practices

### Always Check Return Codes

```c
TensorErrorCode err;
TensorFloatHandle tensor;

err = tensor_float_zeros((size_t[]){10, 10}, 2, &tensor);
if (err != TENSOR_SUCCESS) {
    fprintf(stderr, "Failed to create tensor: %d\n", err);
    return 1;
}

// ... use tensor ...

tensor_float_destroy(tensor);
```

### Destroy All Handles

```c
TensorFloatHandle A, B, C;

// Create tensors
tensor_float_create_1d(10, data, &A);
tensor_float_create_1d(10, data2, &B);

// Operation
tensor_float_add(A, B, &C);

// IMPORTANT: Destroy all three handles
tensor_float_destroy(A);
tensor_float_destroy(B);
tensor_float_destroy(C);  // Don't forget the result!
```

### Use RAII Pattern (if using C++)

```cpp
// If calling from C++, wrap in smart pointers
struct TensorDeleter {
    void operator()(void* handle) {
        if (handle) tensor_float_destroy(handle);
    }
};

using TensorPtr = std::unique_ptr<void, TensorDeleter>;

TensorPtr tensor(/* ... */);
// Automatically destroyed when out of scope
```

## Complete Example

### Linear Regression Training

```c
#include "tensor_c.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    TensorErrorCode err;
    
    // Create synthetic data
    MatrixFloatHandle X;
    VectorFloatHandle y;
    float X_data[100 * 10];  // 100 samples, 10 features
    float y_data[100];
    
    // Initialize with random data (simplified)
    for (int i = 0; i < 1000; i++) X_data[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < 100; i++) y_data[i] = (float)rand() / RAND_MAX;
    
    matrix_float_create(100, 10, X_data, &X);
    vector_float_create(100, y_data, &y);
    
    // Model parameters
    VectorFloatHandle W, b;
    float W_init[10] = {0};
    float b_init[1] = {0};
    vector_float_create(10, W_init, &W);
    vector_float_create(1, b_init, &b);
    tensor_float_requires_grad(W, true);
    tensor_float_requires_grad(b, true);
    
    // Create optimizer
    OptimizerHandle optimizer;
    TensorFloatHandle params[] = {W, b};
    optimizer_adam_create(0.001f, 0.9f, 0.999f, &optimizer);
    optimizer_add_parameters(optimizer, params, 2);
    
    // Loss function
    LossFunctionHandle loss_fn;
    loss_function_mse_create(&loss_fn);
    
    // Training loop
    printf("Training...\n");
    for (int epoch = 0; epoch < 100; epoch++) {
        // Forward pass
        VectorFloatHandle y_pred, loss;
        matrix_vector_multiply(X, W, &y_pred);
        vector_float_add(y_pred, b, &y_pred);
        
        // Compute loss
        loss_function_forward(loss_fn, y_pred, y, &loss);
        
        // Backward pass
        optimizer_zero_grad(optimizer);
        tensor_float_backward(loss);
        optimizer_step(optimizer);
        
        // Print progress
        if (epoch % 10 == 0) {
            float loss_val;
            tensor_float_item(loss, &loss_val);
            printf("Epoch %d, Loss: %f\n", epoch, loss_val);
        }
        
        // Cleanup iteration
        tensor_float_destroy(y_pred);
        tensor_float_destroy(loss);
    }
    
    // Cleanup
    matrix_float_destroy(X);
    vector_float_destroy(y);
    vector_float_destroy(W);
    vector_float_destroy(b);
    optimizer_destroy(optimizer);
    loss_function_destroy(loss_fn);
    
    printf("Training complete!\n");
    return 0;
}
```

### Compilation

```bash
# Compile the C program
gcc -o train train.c -L./build -ltensor_c -I./include -lm

# Run with library path
export LD_LIBRARY_PATH=./build
./train
```

## Device Management

### GPU Support

```c
TensorFloatHandle cpu_tensor, gpu_tensor;

// Create on CPU
tensor_float_create_1d(1000, data, &cpu_tensor);

// Move to GPU
TensorErrorCode err = tensor_float_to_device(cpu_tensor, TENSOR_DEVICE_GPU, &gpu_tensor);
if (err != TENSOR_SUCCESS) {
    fprintf(stderr, "GPU not available\n");
}

// Perform GPU operations
TensorFloatHandle gpu_result;
tensor_float_multiply(gpu_tensor, gpu_tensor, &gpu_result);

// Move back to CPU
TensorFloatHandle cpu_result;
tensor_float_to_device(gpu_result, TENSOR_DEVICE_CPU, &cpu_result);

// Cleanup
tensor_float_destroy(cpu_tensor);
tensor_float_destroy(gpu_tensor);
tensor_float_destroy(gpu_result);
tensor_float_destroy(cpu_result);
```

### Check Device

```c
TensorFloatHandle tensor;
TensorDevice device;

tensor_float_get_device(tensor, &device);
if (device == TENSOR_DEVICE_GPU) {
    printf("Tensor is on GPU\n");
} else {
    printf("Tensor is on CPU\n");
}
```

## Error Handling Patterns

### Robust Error Handling

```c
TensorErrorCode safe_tensor_operation(TensorFloatHandle A, TensorFloatHandle B,
                                     TensorFloatHandle* result) {
    if (!A || !B || !result) {
        return TENSOR_ERROR_NULL_POINTER;
    }
    
    TensorFloatHandle temp;
    TensorErrorCode err;
    
    // First operation
    err = tensor_float_add(A, B, &temp);
    if (err != TENSOR_SUCCESS) {
        return err;
    }
    
    // Second operation
    err = tensor_float_multiply(temp, temp, result);
    tensor_float_destroy(temp);  // Clean up temporary
    
    return err;
}

// Usage
TensorFloatHandle A, B, result;
// ... create A and B ...

TensorErrorCode err = safe_tensor_operation(A, B, &result);
if (err == TENSOR_SUCCESS) {
    printf("Operation succeeded\n");
    tensor_float_destroy(result);
} else {
    fprintf(stderr, "Operation failed with error: %d\n", err);
}
```

## Thread Safety

The C interface is thread-safe for read operations on different tensors. However, concurrent modifications to the same tensor require external synchronization.

```c
#include <pthread.h>

pthread_mutex_t tensor_mutex = PTHREAD_MUTEX_INITIALIZER;
TensorFloatHandle shared_tensor;

void* thread_func(void* arg) {
    pthread_mutex_lock(&tensor_mutex);
    
    // Safe to modify tensor here
    tensor_float_multiply_scalar(shared_tensor, 2.0f, &shared_tensor);
    
    pthread_mutex_unlock(&tensor_mutex);
    return NULL;
}
```

## Integration with Existing C Code

The C interface allows seamless integration with existing C projects:

1. **Include the header**: `#include "tensor_c.h"`
2. **Link the library**: `-ltensor_c`
3. **Use opaque handles**: No need to understand C++ internals
4. **Manage memory explicitly**: Clear ownership model

This makes it easy to add tensor operations to legacy C codebases without rewriting existing code.

## Summary

The C interface provides comprehensive access to the tensor library's functionality:

### Core Features
- ✅ **Vectors and Matrices**: Type-safe handles for 1D and 2D data
- ✅ **Arithmetic Operations**: Element-wise and matrix operations
- ✅ **Mathematical Functions**: Trigonometric, exponential, activation functions
- ✅ **Statistical Operations**: Mean, variance, std, correlation, covariance

### Linear Algebra
- ✅ **Matrix Operations**: Multiplication, transpose, inverse, determinant
- ✅ **Decompositions**: LU (with pivoting), QR, Cholesky, SVD, Eigenvalues
- ✅ **Solvers**: Linear systems, least squares
- ✅ **Matrix Properties**: Rank computation, norms

### Machine Learning
- ✅ **Autograd**: Automatic differentiation with gradient tracking
- ✅ **Loss Functions**: MSE, Cross Entropy, Binary Cross Entropy
- ✅ **Optimizers**: SGD, Adam, AdamW, RMSprop

### I/O and Utilities
- ✅ **Save/Load**: Binary, NumPy (.npy), CSV formats
- ✅ **Shape Manipulation**: Reshape, transpose, squeeze, unsqueeze
- ✅ **Device Management**: CPU/GPU support
- ✅ **Error Handling**: Comprehensive error codes and messages

### Backend Support
The C interface automatically uses the best available backend:
- **GPU** (CUDA): If available, for maximum performance
- **BLAS/LAPACK**: On CPU for optimized linear algebra
- **Pure CPU**: Fallback implementation

### Example Usage
```c
// Quick start example
#include "tensor_c.h"

int main() {
    // Create matrix and vector
    MatrixFloatHandle A;
    VectorFloatHandle x, b;
    float A_data[] = {2.0f, 1.0f, 1.0f, 3.0f};
    float x_data[] = {1.0f, 1.0f};
    
    matrix_float_create(2, 2, A_data, &A);
    vector_float_create(2, x_data, &x);
    
    // Compute b = A * x
    matrix_float_matvec(A, x, &b);
    
    // Get result
    float b0, b1;
    vector_float_get(b, 0, &b0);
    vector_float_get(b, 1, &b1);
    printf("Result: [%.1f, %.1f]\n", b0, b1);
    
    // Cleanup
    matrix_float_destroy(A);
    vector_float_destroy(x);
    vector_float_destroy(b);
    
    return 0;
}
```

For complete examples, see:
- **c_example.c**: Comprehensive demonstration of all features
- **tests/tensor_c_test.c**: Unit tests showing proper usage patterns

---

**Previous**: [← Python Integration](17-python-integration.md) | **Up**: [Index ↑](00-index.md)
