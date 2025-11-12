# Tensor Base Architecture Summary

## Key Findings from tensor_base.h

### 1. Backend System
```cpp
enum class Backend { CPU, BLAS, GPU };
```
- **Priority Order**: GPU > BLAS > CPU
- **Runtime Detection**: `get_active_backend()` checks availability
- **Compile-time Flags**: `USE_GPU`, `USE_BLAS`, `USE_CUBLAS`

### 2. Tensor Class Structure
```cpp
template <typename T, size_t N>
class Tensor {
    std::shared_ptr<T> data_;        // Smart pointer to data
    TensorIndices<N> dims_;          // Dimensions
    bool use_gpu_;                   // GPU flag
    // ... other members
}
```

### 3. Memory Management
- **CPU Memory**: Standard `new[]` / `delete[]`
- **GPU Memory**: `cudaMalloc` / `cudaFree` via custom deleter
- **Access**: `data_ptr()` returns raw pointer (CPU or GPU address)
- **GPU Check**: `uses_gpu()` indicates data location

### 4. Data Location Strategy
From analysis of existing operations:
```cpp
// GPU operations expect data already on GPU
if (tensor.uses_gpu()) {
    T* gpu_ptr = tensor.data_ptr();  // Already on GPU
    // Call GPU kernel directly on gpu_ptr
}
```

### 5. GPU Operation Pattern
Looking at existing GPU implementations in tensor_gpu.cu:

**Pattern A: Copy-based (used for dot products)**
```cpp
void operation_gpu(const T* cpu_a, const T* cpu_b, T* cpu_result, size_t n) {
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, cpu_a, size, cudaMemcpyHostToDevice);
    kernel<<<...>>>(d_a, d_b, d_result);
    cudaMemcpy(cpu_result, d_result, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
```

**Pattern B: Direct (for operations on GPU tensors)**
```cpp
void operation_gpu_direct(T* d_a, T* d_result, size_t n) {
    // d_a and d_result already on GPU
    kernel<<<...>>>(d_a, d_result);
}
```

### 6. Existing GPU Operations
From tensor_gpu.cuh:
- Element-wise: add, sub, mul, div (with scalar variants)
- Math functions: exp, log, sqrt, pow, sin, cos, tanh, sigmoid, relu
- **Both variants exist**: copy-based and direct (`_gpu_direct`)

### 7. BLAS Integration
From tensor_blas.h:
```cpp
#ifdef USE_BLAS
extern "C" {
    // Level 1: Vector operations
    double cblas_dnrm2(int N, const double *X, int incX);  // L2 norm
    double cblas_dasum(int N, const double *X, int incX);  // L1 norm (absolute sum)
    
    // For variance: use ddot for sum of squares
    double cblas_ddot(int N, const double *X, int incX, 
                      const double *Y, int incY);
}
#endif
```

### 8. Parallel CPU Pattern
From existing code:
```cpp
#include <execution>
#include <algorithm>

// Parallel reduce
auto sum = std::reduce(std::execution::par, 
                       data, data + n, T(0));

// Parallel transform
std::transform(std::execution::par,
               src, src + n, dst,
               [](T x) { return f(x); });
```

## Optimization Approach for normalize.h

### Strategy
1. **Detect backend at runtime**
2. **Keep data in place** - no unnecessary transfers
3. **Use direct GPU operations** when tensor is on GPU
4. **Fall back** through BLAS → Parallel CPU → Sequential CPU

### Code Structure
```cpp
template <typename T, size_t N>
Tensor<T, N> normalize_l1(const Tensor<T, N>& tensor, int axis = -1) {
    Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
    
#ifdef USE_GPU
    if (tensor.uses_gpu()) {
        normalize_l1_gpu_direct(tensor.data_ptr(), result.data_ptr(), 
                               /* params */, axis);
        return result;
    }
#endif

#ifdef USE_BLAS
    if (is_blas_available()) {
        normalize_l1_blas(tensor.data_ptr(), result.data_ptr(),
                         /* params */, axis);
        return result;
    }
#endif

    // Parallel CPU with std::execution::par
    normalize_l1_parallel_cpu(tensor.data_ptr(), result.data_ptr(),
                             /* params */, axis);
    return result;
}
```

## Next Steps

1. **Create GPU kernels** in `tensor_normalize_gpu.cu`
2. **Create GPU header** `tensor_normalize_gpu.cuh`
3. **Update normalize.h** with multi-backend logic
4. **Add BLAS implementations** where applicable
5. **Add parallel CPU** using `std::execution::par`
6. **Update CMakeLists.txt** to compile new CUDA file
7. **Run benchmarks** to verify improvements

Ready to proceed with implementation?
