#ifndef _TENSOR_GPU_CUH
#define _TENSOR_GPU_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

#include <cstddef>
#include <stdexcept>

namespace tensor {

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

template<typename T>
void exp_gpu_direct(T* d_a, T* d_result, size_t n);

template<typename T>
void log_gpu_direct(T* d_a, T* d_result, size_t n);

template<typename T>
void sqrt_gpu_direct(T* d_a, T* d_result, size_t n);

template<typename T>
void pow_gpu_direct(T* d_a, T exponent, T* d_result, size_t n);

template<typename T>
void sin_gpu_direct(T* d_a, T* d_result, size_t n);

template<typename T>
void cos_gpu_direct(T* d_a, T* d_result, size_t n);

template<typename T>
void tanh_gpu_direct(T* d_a, T* d_result, size_t n);


// Reduction operations
template<typename T>
void sum_gpu(const T* a, T* result, size_t n);

template<typename T>
void mean_gpu(const T* a, T* result, size_t n);

template<typename T>
void max_gpu(const T* a, T* result, size_t n);

template<typename T>
void min_gpu(const T* a, T* result, size_t n);

// Axis reduction operations
template<typename T>
void reduce_sum_axis_gpu(const T* input, T* output,
                         size_t outer, size_t axis_size, size_t inner);

template<typename T>
void broadcast_add_axis_gpu(const T* grad, T* output,
                            size_t outer, size_t axis_size, size_t inner);

// Binary element-wise operations
template<typename T>
void add_gpu_direct(T* d_a, T* d_b, T* d_result, size_t n);

template<typename T>
void sub_gpu_direct(T* d_a, T* d_b, T* d_result, size_t n);

template<typename T>
void mul_gpu_direct(T* d_a, T* d_b, T* d_result, size_t n);

template<typename T>
void div_gpu_direct(T* d_a, T* d_b, T* d_result, size_t n);

// Activation functions
template<typename T>
void sigmoid_gpu_direct(T* d_a, T* d_result, size_t n);

template<typename T>
void relu_gpu_direct(T* d_a, T* d_result, size_t n);

// Fill operation
template<typename T>
void fill_gpu_direct(T* d_data, T value, size_t n);

} // namespace tensor 

#endif // _TENSOR_GPU_CUH
