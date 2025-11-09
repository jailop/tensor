#ifndef _TENSOR_GPU_CUH
#define _TENSOR_GPU_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <cstddef>

namespace TensorGPU {

bool is_gpu_available();

// Dot product operations
template<typename T>
void dot_1d_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void dot_2d_gpu(const T* a, const T* b, T* result, size_t m, size_t n, size_t p);

template<typename T>
void dot_nd_gpu(const T* a, const T* b, T* result, 
                size_t outer_size, size_t contract_dim, size_t inner_size);

// Cross product for 3D vectors
template<typename T>
void cross_3d_gpu(const T* a, const T* b, T* result);

// Element-wise operations
template<typename T>
void add_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void sub_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void mul_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void div_gpu(const T* a, const T* b, T* result, size_t n);

// Scalar operations
template<typename T>
void add_scalar_gpu(const T* a, T scalar, T* result, size_t n);

template<typename T>
void sub_scalar_gpu(const T* a, T scalar, T* result, size_t n);

template<typename T>
void mul_scalar_gpu(const T* a, T scalar, T* result, size_t n);

template<typename T>
void div_scalar_gpu(const T* a, T scalar, T* result, size_t n);

// Math functions
template<typename T>
void exp_gpu(const T* a, T* result, size_t n);

template<typename T>
void log_gpu(const T* a, T* result, size_t n);

template<typename T>
void sqrt_gpu(const T* a, T* result, size_t n);

template<typename T>
void pow_gpu(const T* a, T exponent, T* result, size_t n);

template<typename T>
void sin_gpu(const T* a, T* result, size_t n);

template<typename T>
void cos_gpu(const T* a, T* result, size_t n);

template<typename T>
void tanh_gpu(const T* a, T* result, size_t n);

template<typename T>
void sigmoid_gpu(const T* a, T* result, size_t n);

template<typename T>
void relu_gpu(const T* a, T* result, size_t n);

// Reduction operations
template<typename T>
void sum_gpu(const T* a, T* result, size_t n);

template<typename T>
void mean_gpu(const T* a, T* result, size_t n);

template<typename T>
void max_gpu(const T* a, T* result, size_t n);

template<typename T>
void min_gpu(const T* a, T* result, size_t n);

} // namespace TensorGPU

#endif // _TENSOR_GPU_CUH
