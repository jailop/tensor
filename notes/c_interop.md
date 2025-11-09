# C Interoperability Guide for Tensor Library

## Overview

This document describes how to create a C interface for the C++ Tensor library. Since C doesn't support templates, classes, or operator overloading, we need to create opaque handle types and explicit function APIs.

## Design Principles

1. **Opaque Handles**: Use void pointers to represent C++ objects
2. **Explicit Type Functions**: Create separate functions for each tensor type (float, double, int)
3. **Error Handling**: Return error codes instead of throwing exceptions
4. **Memory Management**: Explicit create/destroy functions for all objects
5. **C Linkage**: Use `extern "C"` to prevent name mangling

## Architecture

### 1. Header Structure

Create a C header file `tensor_c.h`:

```c
#ifndef TENSOR_C_H
#define TENSOR_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

/* Opaque handle types */
typedef void* TensorFloatHandle;
typedef void* TensorDoubleHandle;
typedef void* MatrixFloatHandle;
typedef void* VectorFloatHandle;
typedef void* OptimizerHandle;

/* Error codes */
typedef enum {
    TENSOR_SUCCESS = 0,
    TENSOR_ERROR_ALLOCATION = 1,
    TENSOR_ERROR_SHAPE = 2,
    TENSOR_ERROR_INDEX = 3,
    TENSOR_ERROR_COMPUTATION = 4,
    TENSOR_ERROR_NULL_POINTER = 5,
    TENSOR_ERROR_INVALID_OPERATION = 6
} TensorErrorCode;

/* Device types */
typedef enum {
    TENSOR_DEVICE_CPU = 0,
    TENSOR_DEVICE_GPU = 1
} TensorDevice;

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_C_H */
```

### 2. Implementation Structure

Create `tensor_c.cpp` implementing the C interface:

```cpp
#include "tensor_c.h"
#include "tensor.h"
#include "tensor_types.h"
#include "optimizers.h"
#include <new>
#include <exception>

extern "C" {

// Helper macro for exception handling
#define TENSOR_TRY_BEGIN try {
#define TENSOR_TRY_END(handle) \
    return TENSOR_SUCCESS; \
    } catch (const std::bad_alloc&) { \
        return TENSOR_ERROR_ALLOCATION; \
    } catch (const std::out_of_range&) { \
        return TENSOR_ERROR_INDEX; \
    } catch (const std::exception&) { \
        return TENSOR_ERROR_COMPUTATION; \
    }

// Tensor Creation and Destruction
TensorErrorCode tensor_float_create_1d(size_t size, float* data, 
                                       TensorFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    std::vector<float> vec(data, data + size);
    auto* tensor = new Tensor<float, 1>({size}, vec);
    *out_handle = tensor;
    TENSOR_TRY_END(out_handle)
}

TensorErrorCode tensor_float_destroy(TensorFloatHandle handle) {
    if (!handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    delete static_cast<Tensor<float, 1>*>(handle);
    TENSOR_TRY_END(handle)
}

// Matrix operations
TensorErrorCode matrix_float_create(size_t rows, size_t cols, 
                                    float* data, MatrixFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    std::vector<float> vec(data, data + rows * cols);
    auto* matrix = new Matrix<float>({rows, cols}, vec);
    *out_handle = matrix;
    TENSOR_TRY_END(out_handle)
}

// Vector operations
TensorErrorCode vector_float_create(size_t size, float* data, 
                                    VectorFloatHandle* out_handle) {
    if (!out_handle) return TENSOR_ERROR_NULL_POINTER;
    
    TENSOR_TRY_BEGIN
    std::vector<float> vec(data, data + size);
    auto* vector = new Vector<float>({size}, vec);
    *out_handle = vector;
    TENSOR_TRY_END(out_handle)
}

} // extern "C"
```

## API Design Patterns

### 3. Basic Operations

For each operation, follow these patterns:

#### 3.1 Creation Functions
```c
TensorErrorCode tensor_float_zeros(size_t* shape, size_t ndim, 
                                   TensorFloatHandle* out_handle);
TensorErrorCode tensor_float_ones(size_t* shape, size_t ndim, 
                                  TensorFloatHandle* out_handle);
TensorErrorCode tensor_float_random(size_t* shape, size_t ndim, 
                                    float min, float max, 
                                    TensorFloatHandle* out_handle);
```

#### 3.2 Element Access
```c
TensorErrorCode tensor_float_get(TensorFloatHandle handle, 
                                size_t* indices, size_t ndim, 
                                float* out_value);
TensorErrorCode tensor_float_set(TensorFloatHandle handle, 
                                size_t* indices, size_t ndim, 
                                float value);
```

#### 3.3 Shape Operations
```c
TensorErrorCode tensor_float_shape(TensorFloatHandle handle, 
                                  size_t* out_shape, size_t* out_ndim);
TensorErrorCode tensor_float_reshape(TensorFloatHandle handle, 
                                    size_t* new_shape, size_t ndim,
                                    TensorFloatHandle* out_handle);
TensorErrorCode tensor_float_transpose(TensorFloatHandle handle,
                                      TensorFloatHandle* out_handle);
```

#### 3.4 Arithmetic Operations
```c
TensorErrorCode tensor_float_add(TensorFloatHandle lhs, TensorFloatHandle rhs,
                                TensorFloatHandle* out_handle);
TensorErrorCode tensor_float_subtract(TensorFloatHandle lhs, TensorFloatHandle rhs,
                                     TensorFloatHandle* out_handle);
TensorErrorCode tensor_float_multiply(TensorFloatHandle lhs, TensorFloatHandle rhs,
                                     TensorFloatHandle* out_handle);
TensorErrorCode tensor_float_divide(TensorFloatHandle lhs, TensorFloatHandle rhs,
                                   TensorFloatHandle* out_handle);
```

#### 3.5 Linear Algebra Operations
```c
TensorErrorCode matrix_float_matmul(MatrixFloatHandle lhs, MatrixFloatHandle rhs,
                                   MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_dot(VectorFloatHandle lhs, VectorFloatHandle rhs,
                                float* out_value);
TensorErrorCode matrix_float_inverse(MatrixFloatHandle handle,
                                    MatrixFloatHandle* out_handle);
TensorErrorCode matrix_float_determinant(MatrixFloatHandle handle,
                                        float* out_value);
```

#### 3.6 Statistical Operations
```c
TensorErrorCode tensor_float_mean(TensorFloatHandle handle, float* out_value);
TensorErrorCode tensor_float_variance(TensorFloatHandle handle, float* out_value);
TensorErrorCode tensor_float_std(TensorFloatHandle handle, float* out_value);
TensorErrorCode tensor_float_sum(TensorFloatHandle handle, float* out_value);
```

### 4. Advanced Features

#### 4.1 Autograd Interface
```c
TensorErrorCode tensor_float_requires_grad(TensorFloatHandle handle, bool requires);
TensorErrorCode tensor_float_backward(TensorFloatHandle handle);
TensorErrorCode tensor_float_zero_grad(TensorFloatHandle handle);
TensorErrorCode tensor_float_grad(TensorFloatHandle handle, TensorFloatHandle* out_grad);
```

#### 4.2 Optimizer Interface
```c
TensorErrorCode optimizer_sgd_create(float learning_rate, 
                                    OptimizerHandle* out_handle);
TensorErrorCode optimizer_adam_create(float learning_rate, float beta1, float beta2,
                                     OptimizerHandle* out_handle);
TensorErrorCode optimizer_add_parameter(OptimizerHandle handle, 
                                       TensorFloatHandle tensor);
TensorErrorCode optimizer_step(OptimizerHandle handle);
TensorErrorCode optimizer_zero_grad(OptimizerHandle handle);
TensorErrorCode optimizer_destroy(OptimizerHandle handle);
```

#### 4.3 Loss Functions
```c
TensorErrorCode loss_mse(TensorFloatHandle predictions, TensorFloatHandle targets,
                        float* out_loss);
TensorErrorCode loss_cross_entropy(TensorFloatHandle predictions, 
                                   TensorFloatHandle targets,
                                   float* out_loss);
```

#### 4.4 Device Management
```c
TensorErrorCode tensor_float_to_device(TensorFloatHandle handle, TensorDevice device,
                                      TensorFloatHandle* out_handle);
TensorErrorCode tensor_float_get_device(TensorFloatHandle handle, 
                                       TensorDevice* out_device);
```

### 5. I/O Operations
```c
TensorErrorCode tensor_float_save(TensorFloatHandle handle, const char* filename);
TensorErrorCode tensor_float_load(const char* filename, TensorFloatHandle* out_handle);
TensorErrorCode tensor_float_to_csv(TensorFloatHandle handle, const char* filename);
TensorErrorCode tensor_float_from_csv(const char* filename, TensorFloatHandle* out_handle);
```

## Implementation Guidelines

### Error Handling Pattern

Always follow this pattern in implementation:

```cpp
TensorErrorCode tensor_operation(/* params */) {
    // 1. Validate input pointers
    if (!handle || !out_handle) {
        return TENSOR_ERROR_NULL_POINTER;
    }
    
    // 2. Try-catch block for C++ exceptions
    TENSOR_TRY_BEGIN
    
    // 3. Cast handle to appropriate type
    auto* tensor = static_cast<Tensor<float, N>*>(handle);
    
    // 4. Perform operation
    auto result = tensor->operation();
    
    // 5. Store result
    *out_handle = new Tensor<float, N>(result);
    
    TENSOR_TRY_END(handle)
}
```

### Memory Management Rules

1. **Every create function must have a corresponding destroy function**
2. **Caller is responsible for destroying returned handles**
3. **Functions that return new tensors allocate new memory**
4. **In-place operations modify existing tensors**

Example usage in C:

```c
TensorFloatHandle tensor1, tensor2, result;
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};

// Create tensors
tensor_float_create_1d(4, data, &tensor1);
tensor_float_zeros(&(size_t){4}, 1, &tensor2);

// Perform operation
tensor_float_add(tensor1, tensor2, &result);

// Clean up
tensor_float_destroy(tensor1);
tensor_float_destroy(tensor2);
tensor_float_destroy(result);
```

## Build Configuration

### CMakeLists.txt Addition

```cmake
# C Interface library
add_library(tensor_c SHARED
    src/tensor_c.cpp
)

target_include_directories(tensor_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tensor_c PRIVATE
    tensor_static
)

# Set C linkage
set_target_properties(tensor_c PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Install C library and header
install(TARGETS tensor_c
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/tensor_c.h
    DESTINATION include
)
```

## Usage Example

### Complete C Program Example

```c
#include "tensor_c.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    TensorFloatHandle A, B, C;
    TensorErrorCode err;
    
    // Create matrix A (2x2)
    float dataA[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t shape[] = {2, 2};
    err = tensor_float_create_2d(shape, dataA, &A);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating tensor A: %d\n", err);
        return 1;
    }
    
    // Create matrix B (2x2)
    float dataB[] = {5.0f, 6.0f, 7.0f, 8.0f};
    err = tensor_float_create_2d(shape, dataB, &B);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating tensor B: %d\n", err);
        tensor_float_destroy(A);
        return 1;
    }
    
    // Matrix multiplication
    err = tensor_float_matmul(A, B, &C);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error in matmul: %d\n", err);
        tensor_float_destroy(A);
        tensor_float_destroy(B);
        return 1;
    }
    
    // Get result values
    float value;
    size_t indices[2];
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            indices[0] = i;
            indices[1] = j;
            tensor_float_get(C, indices, 2, &value);
            printf("C[%zu,%zu] = %f\n", i, j, value);
        }
    }
    
    // Cleanup
    tensor_float_destroy(A);
    tensor_float_destroy(B);
    tensor_float_destroy(C);
    
    return 0;
}
```

### Compilation

```bash
# Compile the C program
gcc -o example example.c -L./build -ltensor_c -I./include

# Run with library path
LD_LIBRARY_PATH=./build ./example
```

## Advanced Topics

### Thread Safety

For thread-safe operations, add mutex protection:

```c
typedef struct {
    void* tensor_ptr;
    pthread_mutex_t* mutex;
} ThreadSafeTensorHandle;
```

### Callback Functions

For custom operations:

```c
typedef float (*TensorElementFunc)(float value, void* user_data);

TensorErrorCode tensor_float_apply(TensorFloatHandle handle,
                                  TensorElementFunc func,
                                  void* user_data,
                                  TensorFloatHandle* out_handle);
```

### Batch Operations

For efficiency with multiple tensors:

```c
TensorErrorCode tensor_float_batch_add(TensorFloatHandle* handles,
                                      size_t count,
                                      TensorFloatHandle* out_handle);
```

## Testing

Create C tests in `tests/tensor_c_test.c`:

```c
#include "tensor_c.h"
#include <assert.h>
#include <math.h>

void test_tensor_creation() {
    TensorFloatHandle handle;
    float data[] = {1.0f, 2.0f, 3.0f};
    size_t shape[] = {3};
    
    TensorErrorCode err = tensor_float_create_1d(3, data, &handle);
    assert(err == TENSOR_SUCCESS);
    assert(handle != NULL);
    
    tensor_float_destroy(handle);
}

void test_tensor_operations() {
    TensorFloatHandle A, B, C;
    float dataA[] = {1.0f, 2.0f};
    float dataB[] = {3.0f, 4.0f};
    
    tensor_float_create_1d(2, dataA, &A);
    tensor_float_create_1d(2, dataB, &B);
    
    TensorErrorCode err = tensor_float_add(A, B, &C);
    assert(err == TENSOR_SUCCESS);
    
    float value;
    size_t idx[] = {0};
    tensor_float_get(C, idx, 1, &value);
    assert(fabs(value - 4.0f) < 1e-6);
    
    tensor_float_destroy(A);
    tensor_float_destroy(B);
    tensor_float_destroy(C);
}

int main() {
    test_tensor_creation();
    test_tensor_operations();
    printf("All C tests passed!\n");
    return 0;
}
```

## Best Practices

1. **Always check return codes** from every function call
2. **Destroy all created handles** to prevent memory leaks
3. **Use const char*** for string parameters (filenames, etc.)
4. **Document ownership** of returned handles clearly
5. **Provide version information** for ABI compatibility
6. **Use size_t for array sizes** to match C conventions
7. **Add prefix** to all functions (e.g., `tensor_`) to avoid naming conflicts
8. **Keep the API flat** - avoid nested structures where possible

## Conclusion

This C interface provides:
- Full access to tensor operations from C code
- Safe memory management with explicit create/destroy functions
- Clear error handling through return codes
- Type-specific functions for different tensor types
- Integration with existing C codebases

The interface can be extended with additional functions as needed while maintaining backward compatibility.
