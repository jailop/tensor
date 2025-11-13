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

/**
 * Check if a GPU is available for computations.
 * @return true if a GPU is available, false otherwise.
 */
bool is_gpu_available();

template<typename T>
void dot_1d_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void dot_2d_gpu(const T* a, const T* b, T* result, size_t m, size_t n, size_t p);

template<typename T>
void dot_nd_gpu(const T* a, const T* b, T* result, 
                size_t outer_size, size_t contract_dim, size_t inner_size);

template<typename T>
void cross_3d_gpu(const T* a, const T* b, T* result);

template<typename T>
void add_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void sub_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void mul_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void div_gpu(const T* a, const T* b, T* result, size_t n);

template<typename T>
void add_scalar_gpu(const T* a, T scalar, T* result, size_t n);

template<typename T>
void sub_scalar_gpu(const T* a, T scalar, T* result, size_t n);

template<typename T>
void mul_scalar_gpu(const T* a, T scalar, T* result, size_t n);

template<typename T>
void div_scalar_gpu(const T* a, T scalar, T* result, size_t n);

template<typename T>
void div_scalar_gpu_direct(T* d_a, T scalar, T* d_result, size_t n);

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

// Direct GPU reduction operations (data already on GPU)
template<typename T>
void sum_gpu_direct(const T* d_a, T* d_result, size_t n);

template<typename T>
void min_gpu_direct(const T* d_a, T* d_result, size_t n);

template<typename T>
void max_gpu_direct(const T* d_a, T* d_result, size_t n);

// Element-wise abs operation
template<typename T>
void abs_gpu_direct(const T* d_src, T* d_dst, size_t n);

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

// L1 Normalization helpers
template<typename T>
void abs_sum_gpu_direct(const T* d_src, T* d_result, size_t n);

template<typename T>
void abs_sum_axis_gpu_direct(const T* d_src, T* d_sums, 
                              size_t outer, size_t axis_size, size_t inner);

template<typename T>
void normalize_by_sums_gpu_direct(const T* d_src, const T* d_sums, T* d_dst,
                                   size_t outer, size_t axis_size, size_t inner);

// L2 Normalization helpers
template<typename T>
void l2_norm_gpu_direct(const T* d_src, T* d_result, size_t n);

template<typename T>
void l2_norm_axis_gpu_direct(const T* d_src, T* d_norms,
                              size_t outer, size_t axis_size, size_t inner);

// Z-score Normalization helpers
template<typename T>
void zscore_normalize_gpu_direct(const T* d_src, T* d_dst, size_t n, T eps);

template<typename T>
void zscore_normalize_axis_gpu_direct(const T* d_src, T* d_dst,
                                       size_t outer, size_t axis_size,
                                       size_t inner, T eps);

// Min-Max Normalization helpers
template<typename T>
void minmax_normalize_gpu_direct(const T* d_src, T* d_dst, size_t n,
                                 T min_val, T max_val, T eps);

template<typename T>
void minmax_normalize_axis_gpu_direct(const T* d_src, T* d_dst,
                                      size_t outer, size_t axis_size,
                                      size_t inner, T min_val, T max_val,
                                      T eps);

} // namespace tensor 

#endif // _TENSOR_GPU_CUH
