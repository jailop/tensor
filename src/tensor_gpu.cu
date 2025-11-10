#include "tensor_gpu.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace TensorGPU {

// Device-compatible limit helpers - specializations for different types
template<typename T>
__device__ __forceinline__ T get_lowest_value();

template<>
__device__ __forceinline__ int get_lowest_value<int>() { return -2147483648; }

template<>
__device__ __forceinline__ float get_lowest_value<float>() { return -1e30f; }

template<>
__device__ __forceinline__ double get_lowest_value<double>() { return -1e30; }

template<typename T>
__device__ __forceinline__ T get_max_value();

template<>
__device__ __forceinline__ int get_max_value<int>() { return 2147483647; }

template<>
__device__ __forceinline__ float get_max_value<float>() { return 1e30f; }

template<>
__device__ __forceinline__ double get_max_value<double>() { return 1e30; }

bool is_gpu_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

template<typename T>
__global__ void dot_1d_kernel(const T* a, const T* b, T* partial_sums, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    T sum = T();
    if (i < n) {
        sum = a[i] * b[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

template<typename T>
void dot_1d_gpu(const T* a, const T* b, T* result, size_t n) {
    T *d_a, *d_b, *d_partial;
    
    size_t size = n * sizeof(T);
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_partial, blocks * sizeof(T));
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    dot_1d_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(T)>>>(
        d_a, d_b, d_partial, n);
    
    T* h_partial = new T[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost);
    
    *result = T();
    for (int i = 0; i < blocks; i++) {
        *result += h_partial[i];
    }
    
    delete[] h_partial;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial);
}

template<typename T>
__global__ void dot_2d_kernel(const T* a, const T* b, T* c, 
                               size_t m, size_t n, size_t p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < p) {
        T sum = T();
        for (size_t k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * p + col];
        }
        c[row * p + col] = sum;
    }
}

template<typename T>
void dot_2d_gpu(const T* a, const T* b, T* result, size_t m, size_t n, size_t p) {
    T *d_a, *d_b, *d_c;
    
    size_t size_a = m * n * sizeof(T);
    size_t size_b = n * p * sizeof(T);
    size_t size_c = m * p * sizeof(T);
    
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((p + 15) / 16, (m + 15) / 16);
    
    dot_2d_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, m, n, p);
    
    cudaMemcpy(result, d_c, size_c, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

template<typename T>
__global__ void dot_nd_kernel(const T* a, const T* b, T* c,
                               size_t outer_size, size_t contract_dim, size_t inner_size) {
    int outer = blockIdx.y * blockDim.y + threadIdx.y;
    int inner = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outer < outer_size && inner < inner_size) {
        T sum = T();
        for (size_t k = 0; k < contract_dim; k++) {
            size_t idx_a = outer * contract_dim + k;
            size_t idx_b = k * inner_size + inner;
            sum += a[idx_a] * b[idx_b];
        }
        c[outer * inner_size + inner] = sum;
    }
}

template<typename T>
void dot_nd_gpu(const T* a, const T* b, T* result,
                size_t outer_size, size_t contract_dim, size_t inner_size) {
    T *d_a, *d_b, *d_c;
    
    size_t size_a = outer_size * contract_dim * sizeof(T);
    size_t size_b = contract_dim * inner_size * sizeof(T);
    size_t size_c = outer_size * inner_size * sizeof(T);
    
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((inner_size + 15) / 16, (outer_size + 15) / 16);
    
    dot_nd_kernel<<<num_blocks, threads_per_block>>>(
        d_a, d_b, d_c, outer_size, contract_dim, inner_size);
    
    cudaMemcpy(result, d_c, size_c, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

template void dot_1d_gpu<int>(const int*, const int*, int*, size_t);
template void dot_1d_gpu<float>(const float*, const float*, float*, size_t);
template void dot_1d_gpu<double>(const double*, const double*, double*, size_t);

template void dot_2d_gpu<int>(const int*, const int*, int*, size_t, size_t, size_t);
template void dot_2d_gpu<float>(const float*, const float*, float*, size_t, size_t, size_t);
template void dot_2d_gpu<double>(const double*, const double*, double*, size_t, size_t, size_t);

template void dot_nd_gpu<int>(const int*, const int*, int*, size_t, size_t, size_t);
template void dot_nd_gpu<float>(const float*, const float*, float*, size_t, size_t, size_t);
template void dot_nd_gpu<double>(const double*, const double*, double*, size_t, size_t, size_t);

// Cross product kernel for 3D vectors
template<typename T>
__global__ void cross_3d_kernel(const T* a, const T* b, T* result) {
    // Cross product: a × b = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
    if (threadIdx.x == 0) {
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
    }
}

template<typename T>
void cross_3d_gpu(const T* a, const T* b, T* result) {
    T *d_a, *d_b, *d_c;
    
    size_t size = 3 * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    cross_3d_kernel<<<1, 1>>>(d_a, d_b, d_c);
    
    cudaMemcpy(result, d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

template void cross_3d_gpu<int>(const int*, const int*, int*);
template void cross_3d_gpu<float>(const float*, const float*, float*);
template void cross_3d_gpu<double>(const double*, const double*, double*);

// Element-wise operation kernels
template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void sub_kernel(const T* a, const T* b, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

template<typename T>
__global__ void mul_kernel(const T* a, const T* b, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

template<typename T>
__global__ void div_kernel(const T* a, const T* b, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] / b[idx];
    }
}

template<typename T>
void add_gpu(const T* a, const T* b, T* result, size_t n) {
    T *d_a, *d_b, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

template<typename T>
void sub_gpu(const T* a, const T* b, T* result, size_t n) {
    T *d_a, *d_b, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    sub_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

template<typename T>
void mul_gpu(const T* a, const T* b, T* result, size_t n) {
    T *d_a, *d_b, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    mul_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

template<typename T>
void div_gpu(const T* a, const T* b, T* result, size_t n) {
    T *d_a, *d_b, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    div_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

// Scalar operation kernels
template<typename T>
__global__ void add_scalar_kernel(const T* a, T scalar, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + scalar;
    }
}

template<typename T>
__global__ void sub_scalar_kernel(const T* a, T scalar, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - scalar;
    }
}

template<typename T>
__global__ void mul_scalar_kernel(const T* a, T scalar, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * scalar;
    }
}

template<typename T>
__global__ void div_scalar_kernel(const T* a, T scalar, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] / scalar;
    }
}

template<typename T>
void add_scalar_gpu(const T* a, T scalar, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    add_scalar_kernel<<<blocks, threads_per_block>>>(d_a, scalar, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void sub_scalar_gpu(const T* a, T scalar, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    sub_scalar_kernel<<<blocks, threads_per_block>>>(d_a, scalar, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void mul_scalar_gpu(const T* a, T scalar, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    mul_scalar_kernel<<<blocks, threads_per_block>>>(d_a, scalar, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void div_scalar_gpu(const T* a, T scalar, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    div_scalar_kernel<<<blocks, threads_per_block>>>(d_a, scalar, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

// Math function kernels
template<typename T>
__global__ void exp_kernel(const T* a, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = exp(a[idx]);
    }
}

template<typename T>
__global__ void log_kernel(const T* a, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = log(a[idx]);
    }
}

template<typename T>
__global__ void sqrt_kernel(const T* a, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = sqrt(a[idx]);
    }
}

template<typename T>
__global__ void pow_kernel(const T* a, T exponent, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = pow(a[idx], exponent);
    }
}

template<typename T>
__global__ void sin_kernel(const T* a, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = sin(a[idx]);
    }
}

template<typename T>
__global__ void cos_kernel(const T* a, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = cos(a[idx]);
    }
}

template<typename T>
__global__ void tanh_kernel(const T* a, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = tanh(a[idx]);
    }
}

template<typename T>
__global__ void sigmoid_kernel(const T* a, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = T(1) / (T(1) + exp(-a[idx]));
    }
}

template<typename T>
__global__ void relu_kernel(const T* a, T* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] > T(0) ? a[idx] : T(0);
    }
}

template<typename T>
void exp_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    exp_kernel<<<blocks, threads_per_block>>>(d_a, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void log_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    log_kernel<<<blocks, threads_per_block>>>(d_a, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void sqrt_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    sqrt_kernel<<<blocks, threads_per_block>>>(d_a, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void pow_gpu(const T* a, T exponent, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    pow_kernel<<<blocks, threads_per_block>>>(d_a, exponent, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void sin_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    sin_kernel<<<blocks, threads_per_block>>>(d_a, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void cos_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    cos_kernel<<<blocks, threads_per_block>>>(d_a, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void tanh_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    tanh_kernel<<<blocks, threads_per_block>>>(d_a, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void sigmoid_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    sigmoid_kernel<<<blocks, threads_per_block>>>(d_a, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

template<typename T>
void relu_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_result;
    size_t size = n * sizeof(T);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    relu_kernel<<<blocks, threads_per_block>>>(d_a, d_result, n);
    
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_result);
}

// Reduction operation kernels
template<typename T>
__global__ void sum_kernel(const T* a, T* partial_sums, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    T sum = (i < n) ? a[i] : T(0);
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

template<typename T>
void sum_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_partial;
    size_t size = n * sizeof(T);
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_partial, blocks * sizeof(T));
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    sum_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(T)>>>(
        d_a, d_partial, n);
    
    T* h_partial = new T[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost);
    
    *result = T(0);
    for (int i = 0; i < blocks; i++) {
        *result += h_partial[i];
    }
    
    delete[] h_partial;
    cudaFree(d_a);
    cudaFree(d_partial);
}

template<typename T>
void mean_gpu(const T* a, T* result, size_t n) {
    sum_gpu(a, result, n);
    *result /= static_cast<T>(n);
}

template<typename T>
__global__ void max_kernel(const T* a, T* partial_max, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use device-compatible lowest value
    T val = (i < n) ? a[i] : (i == 0 ? a[0] : get_lowest_value<T>());
    sdata[tid] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

template<typename T>
void max_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_partial;
    size_t size = n * sizeof(T);
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_partial, blocks * sizeof(T));
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    max_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(T)>>>(
        d_a, d_partial, n);
    
    T* h_partial = new T[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost);
    
    *result = h_partial[0];
    for (int i = 1; i < blocks; i++) {
        if (h_partial[i] > *result) {
            *result = h_partial[i];
        }
    }
    
    delete[] h_partial;
    cudaFree(d_a);
    cudaFree(d_partial);
}

template<typename T>
__global__ void min_kernel(const T* a, T* partial_min, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use device-compatible max value
    T val = (i < n) ? a[i] : (i == 0 ? a[0] : get_max_value<T>());
    sdata[tid] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] < sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_min[blockIdx.x] = sdata[0];
    }
}

template<typename T>
void min_gpu(const T* a, T* result, size_t n) {
    T *d_a, *d_partial;
    size_t size = n * sizeof(T);
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_partial, blocks * sizeof(T));
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    min_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(T)>>>(
        d_a, d_partial, n);
    
    T* h_partial = new T[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost);
    
    *result = h_partial[0];
    for (int i = 1; i < blocks; i++) {
        if (h_partial[i] < *result) {
            *result = h_partial[i];
        }
    }
    
    delete[] h_partial;
    cudaFree(d_a);
    cudaFree(d_partial);
}

// Template instantiations for element-wise operations
template void add_gpu<int>(const int*, const int*, int*, size_t);
template void add_gpu<float>(const float*, const float*, float*, size_t);
template void add_gpu<double>(const double*, const double*, double*, size_t);

template void sub_gpu<int>(const int*, const int*, int*, size_t);
template void sub_gpu<float>(const float*, const float*, float*, size_t);
template void sub_gpu<double>(const double*, const double*, double*, size_t);

// ============================================================================
// Axis Reduction Operations
// ============================================================================

/**
 * @brief GPU kernel for summing along an axis
 * Computes sum reduction along one dimension with outer x inner parallelism
 */
template<typename T>
__global__ void reduce_sum_axis_kernel(const T* input, T* output,
                                       size_t outer, size_t axis_size, size_t inner) {
    size_t o = blockIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (o < outer && i < inner) {
        T sum = T(0);
        for (size_t a = 0; a < axis_size; ++a) {
            size_t src_idx = o * axis_size * inner + a * inner + i;
            sum += input[src_idx];
        }
        size_t dst_idx = o * inner + i;
        output[dst_idx] = sum;
    }
}

template<typename T>
void reduce_sum_axis_gpu(const T* input, T* output,
                         size_t outer, size_t axis_size, size_t inner) {
    int threads = 256;
    dim3 blocks((inner + threads - 1) / threads, outer);
    
    reduce_sum_axis_kernel<<<blocks, threads>>>(input, output, outer, axis_size, inner);
    cudaDeviceSynchronize();
}

/**
 * @brief GPU kernel for broadcasting gradient during backward pass
 * Broadcasts reduced gradient back to original shape
 */
template<typename T>
__global__ void broadcast_add_axis_kernel(const T* grad, T* output,
                                          size_t outer, size_t axis_size, size_t inner) {
    size_t o = blockIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (o < outer && i < inner) {
        size_t grad_idx = o * inner + i;
        T grad_val = grad[grad_idx];
        
        for (size_t a = 0; a < axis_size; ++a) {
            size_t dst_idx = o * axis_size * inner + a * inner + i;
            atomicAdd(&output[dst_idx], grad_val);
        }
    }
}

template<typename T>
void broadcast_add_axis_gpu(const T* grad, T* output,
                            size_t outer, size_t axis_size, size_t inner) {
    int threads = 256;
    dim3 blocks((inner + threads - 1) / threads, outer);
    
    broadcast_add_axis_kernel<<<blocks, threads>>>(grad, output, outer, axis_size, inner);
    cudaDeviceSynchronize();
}

// Template instantiations for axis reduction operations
template void reduce_sum_axis_gpu<float>(const float*, float*, size_t, size_t, size_t);
template void reduce_sum_axis_gpu<double>(const double*, double*, size_t, size_t, size_t);

template void broadcast_add_axis_gpu<float>(const float*, float*, size_t, size_t, size_t);
template void broadcast_add_axis_gpu<double>(const double*, double*, size_t, size_t, size_t);

// ============================================================================

template void mul_gpu<int>(const int*, const int*, int*, size_t);
template void mul_gpu<float>(const float*, const float*, float*, size_t);
template void mul_gpu<double>(const double*, const double*, double*, size_t);

template void div_gpu<int>(const int*, const int*, int*, size_t);
template void div_gpu<float>(const float*, const float*, float*, size_t);
template void div_gpu<double>(const double*, const double*, double*, size_t);

// Template instantiations for scalar operations
template void add_scalar_gpu<int>(const int*, int, int*, size_t);
template void add_scalar_gpu<float>(const float*, float, float*, size_t);
template void add_scalar_gpu<double>(const double*, double, double*, size_t);

template void sub_scalar_gpu<int>(const int*, int, int*, size_t);
template void sub_scalar_gpu<float>(const float*, float, float*, size_t);
template void sub_scalar_gpu<double>(const double*, double, double*, size_t);

template void mul_scalar_gpu<int>(const int*, int, int*, size_t);
template void mul_scalar_gpu<float>(const float*, float, float*, size_t);
template void mul_scalar_gpu<double>(const double*, double, double*, size_t);

template void div_scalar_gpu<int>(const int*, int, int*, size_t);
template void div_scalar_gpu<float>(const float*, float, float*, size_t);
template void div_scalar_gpu<double>(const double*, double, double*, size_t);

// Template instantiations for math functions
template void exp_gpu<float>(const float*, float*, size_t);
template void exp_gpu<double>(const double*, double*, size_t);

template void log_gpu<float>(const float*, float*, size_t);
template void log_gpu<double>(const double*, double*, size_t);

template void sqrt_gpu<float>(const float*, float*, size_t);
template void sqrt_gpu<double>(const double*, double*, size_t);

template void pow_gpu<float>(const float*, float, float*, size_t);
template void pow_gpu<double>(const double*, double, double*, size_t);

template void sin_gpu<float>(const float*, float*, size_t);
template void sin_gpu<double>(const double*, double*, size_t);

template void cos_gpu<float>(const float*, float*, size_t);
template void cos_gpu<double>(const double*, double*, size_t);

template void tanh_gpu<float>(const float*, float*, size_t);
template void tanh_gpu<double>(const double*, double*, size_t);

template void sigmoid_gpu<float>(const float*, float*, size_t);
template void sigmoid_gpu<double>(const double*, double*, size_t);

template void relu_gpu<float>(const float*, float*, size_t);
template void relu_gpu<double>(const double*, double*, size_t);

// Template instantiations for reduction operations
template void sum_gpu<int>(const int*, int*, size_t);
template void sum_gpu<float>(const float*, float*, size_t);
template void sum_gpu<double>(const double*, double*, size_t);

template void mean_gpu<int>(const int*, int*, size_t);
template void mean_gpu<float>(const float*, float*, size_t);
template void mean_gpu<double>(const double*, double*, size_t);

template void max_gpu<int>(const int*, int*, size_t);
template void max_gpu<float>(const float*, float*, size_t);
template void max_gpu<double>(const double*, double*, size_t);

template void min_gpu<int>(const int*, int*, size_t);
template void min_gpu<float>(const float*, float*, size_t);
template void min_gpu<double>(const double*, double*, size_t);

} // namespace TensorGPU

// C wrapper functions for CUDA API (for use in non-CUDA files)
extern "C" {

void* cuda_malloc_wrapper(size_t bytes) {
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void cuda_free_wrapper(void* ptr) {
    cudaFree(ptr);
}

void cuda_memcpy_h2d_wrapper(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void cuda_memcpy_d2h_wrapper(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void cuda_memcpy_d2d_wrapper(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}

} // extern "C"
