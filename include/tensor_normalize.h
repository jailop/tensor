#ifndef TENSOR_NORMALIZE_H
#define TENSOR_NORMALIZE_H

#include "tensor_base.h"
#include <numeric>

#ifdef USE_BLAS
#include "tensor_blas.h"
#endif

namespace tensor {

/**
 * @brief Normalize tensor using L1 norm (sum of absolute values)
 * @tparam T Data type
 * @tparam N Number of dimensions
 * @param tensor Input tensor
 * @param axis Axis along which to normalize (-1 for all elements)
 * @return Normalized tensor where L1 norm equals 1
 * 
 * L1 normalization divides each element by the sum of absolute values.
 * Useful for probability distributions and sparse data.
 */
template <typename T, size_t N>
Tensor<T, N> normalize_l1(const Tensor<T, N>& tensor, int axis = -1) {
    if (axis == -1) {
        T sum = tensor.abs().sum();
        if (sum > T(0)) {
            return tensor / sum;
        } else {
            return tensor;
        }
    } else if constexpr (N >= 2) {
        // Normalize along specific axis with three-tier backend
        Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
        const T* src = tensor.data_ptr();
        T* dst = result.data_ptr();
        
        auto dims = tensor.dims();
        size_t axis_size = dims[axis];
        size_t outer_size = 1;
        size_t inner_size = 1;
        
        for (size_t i = 0; i < axis; ++i) {
            outer_size *= dims[i];
        }
        for (size_t i = axis + 1; i < N; ++i) {
            inner_size *= dims[i];
        }
        
#ifdef USE_GPU
        if (tensor.uses_gpu()) {
            size_t sums_size = outer_size * inner_size;
            Tensor<T, 1> d_sums_tensor({sums_size}, true);
            T* d_sums = d_sums_tensor.data_ptr();
            
            abs_sum_axis_gpu_direct(src, d_sums, outer_size, axis_size, inner_size);
            normalize_by_sums_gpu_direct(src, d_sums, dst, outer_size, axis_size, inner_size);
            return result;
        }
#endif

#ifdef USE_BLAS
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t inner = 0; inner < inner_size; ++inner) {
                    size_t first_idx = outer * axis_size * inner_size + inner;
                    
                    // Compute L1 norm using BLAS
                    T sum;
                    if constexpr (std::is_same_v<T, float>) {
                        sum = cblas_sasum(static_cast<int>(axis_size), 
                                         src + first_idx, static_cast<int>(inner_size));
                    } else {
                        sum = cblas_dasum(static_cast<int>(axis_size), 
                                         src + first_idx, static_cast<int>(inner_size));
                    }
                    
                    // Normalize
                    if (sum > T(0)) {
                        for (size_t ax = 0; ax < axis_size; ++ax) {
                            size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                            dst[idx] = src[idx] / sum;
                        }
                    } else {
                        for (size_t ax = 0; ax < axis_size; ++ax) {
                            size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                            dst[idx] = src[idx];
                        }
                    }
                }
            }
            return result;
        }
#endif

        // Parallel CPU fallback
        std::for_each(std::execution::par_unseq,
                     std::views::iota(size_t(0), outer_size).begin(),
                     std::views::iota(size_t(0), outer_size).end(),
                     [&](size_t outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                T sum = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    sum += std::abs(src[idx]);
                }
                
                if (sum > T(0)) {
                    for (size_t ax = 0; ax < axis_size; ++ax) {
                        size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                        dst[idx] = src[idx] / sum;
                    }
                } else {
                    for (size_t ax = 0; ax < axis_size; ++ax) {
                        size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                        dst[idx] = src[idx];
                    }
                }
            }
        });
        return result;
    } else {
        return tensor;
    }
}

/**
 * @brief Normalize tensor using L2 norm (Euclidean norm)
 * @tparam T Data type
 * @tparam N Number of dimensions
 * @param tensor Input tensor
 * @param axis Axis along which to normalize (-1 for all elements)
 * @return Normalized tensor where L2 norm equals 1
 * 
 * L2 normalization divides each element by the square root of the sum of squares.
 * Common in machine learning for feature normalization.
 * 
 * Optimized for GPU, BLAS, and parallel CPU execution.
 */
template <typename T, size_t N>
Tensor<T, N> normalize_l2(const Tensor<T, N>& tensor, int axis = -1) {
    if (axis == -1) {
        auto squared = tensor * tensor;
        T sum_sq = std::get<Tensor<T, N>>(squared).sum();
        T norm = std::sqrt(sum_sq);
        
        if (norm > T(0)) {
            return tensor / norm;
        } else {
            return tensor;
        }
    } else if constexpr (N >= 2) {
        // Normalize along specific axis with three-tier backend
        Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
        const T* src = tensor.data_ptr();
        T* dst = result.data_ptr();
        
        auto dims = tensor.dims();
        size_t axis_size = dims[axis];
        size_t outer_size = 1;
        size_t inner_size = 1;
        
        for (size_t i = 0; i < axis; ++i) {
            outer_size *= dims[i];
        }
        for (size_t i = axis + 1; i < N; ++i) {
            inner_size *= dims[i];
        }
        
#ifdef USE_GPU
        if (tensor.uses_gpu()) {
            size_t norms_size = outer_size * inner_size;
            Tensor<T, 1> d_norms_tensor({norms_size}, true);
            T* d_norms = d_norms_tensor.data_ptr();
            
            l2_norm_axis_gpu_direct(src, d_norms, outer_size, axis_size, inner_size);
            normalize_by_sums_gpu_direct(src, d_norms, dst, outer_size, axis_size, inner_size);
            return result;
        }
#endif

#ifdef USE_BLAS
        // BLAS path for axis-specific operations
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t inner = 0; inner < inner_size; ++inner) {
                    size_t first_idx = outer * axis_size * inner_size + inner;
                    
                    // Compute L2 norm using BLAS
                    T norm;
                    if constexpr (std::is_same_v<T, float>) {
                        norm = cblas_snrm2(static_cast<int>(axis_size), 
                                          src + first_idx, static_cast<int>(inner_size));
                    } else {
                        norm = cblas_dnrm2(static_cast<int>(axis_size), 
                                          src + first_idx, static_cast<int>(inner_size));
                    }
                    
                    // Normalize
                    if (norm > T(0)) {
                        for (size_t ax = 0; ax < axis_size; ++ax) {
                            size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                            dst[idx] = src[idx] / norm;
                        }
                    } else {
                        for (size_t ax = 0; ax < axis_size; ++ax) {
                            size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                            dst[idx] = src[idx];
                        }
                    }
                }
            }
            return result;
        }
#endif

        // Parallel CPU fallback
        std::for_each(std::execution::par_unseq,
                     std::views::iota(size_t(0), outer_size).begin(),
                     std::views::iota(size_t(0), outer_size).end(),
                     [&](size_t outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                T sum_sq = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    sum_sq += src[idx] * src[idx];
                }
                
                T norm = std::sqrt(sum_sq);
                
                if (norm > T(0)) {
                    for (size_t ax = 0; ax < axis_size; ++ax) {
                        size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                        dst[idx] = src[idx] / norm;
                    }
                } else {
                    for (size_t ax = 0; ax < axis_size; ++ax) {
                        size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                        dst[idx] = src[idx];
                    }
                }
            }
        });
        return result;
    } else {
        return tensor;
    }
}

/**
 * @brief Z-score normalization (standardization): (x - mean) / std
 * @tparam T Data type
 * @tparam N Number of dimensions
 * @param tensor Input tensor
 * @param axis Axis along which to normalize (-1 for all elements)
 * @param eps Small constant to avoid division by zero
 * @return Standardized tensor with mean ~0 and std ~1
 * 
 * Z-score normalization is common in statistical analysis and machine learning.
 * 
 * Optimized for GPU, BLAS, and parallel CPU execution.
 */
template <typename T, size_t N>
Tensor<T, N> normalize_zscore(const Tensor<T, N>& tensor, int axis = -1, T eps = T(1e-8)) {
    if (axis == -1) {
        T mean_val = tensor.mean();
        T std_val = tensor.std();
        std_val = std::max(std_val, eps);
        
        return (tensor - mean_val) / std_val;
    } else if constexpr (N >= 2) {
        // Normalize along specific axis with three-tier backend
        Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
        const T* src = tensor.data_ptr();
        T* dst = result.data_ptr();
        
        auto dims = tensor.dims();
        size_t axis_size = dims[axis];
        size_t outer_size = 1;
        size_t inner_size = 1;
        
        for (size_t i = 0; i < axis; ++i) {
            outer_size *= dims[i];
        }
        for (size_t i = axis + 1; i < N; ++i) {
            inner_size *= dims[i];
        }
        
#ifdef USE_GPU
        if (tensor.uses_gpu()) {
            zscore_normalize_axis_gpu_direct(src, dst, outer_size, axis_size, inner_size, eps);
            return result;
        }
#endif

        // CPU fallback: BLAS or parallel CPU
        std::for_each(std::execution::par_unseq,
                     std::views::iota(size_t(0), outer_size).begin(),
                     std::views::iota(size_t(0), outer_size).end(),
                     [&](size_t outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                T mean = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    mean += src[idx];
                }
                mean /= axis_size;
                
                T var = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    T diff = src[idx] - mean;
                    var += diff * diff;
                }
                var /= axis_size;
                T std_dev = std::sqrt(var + eps);
                
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    dst[idx] = (src[idx] - mean) / std_dev;
                }
            }
        });
        return result;
    } else {
        return tensor;
    }
}

/**
 * @brief Min-Max normalization: scales values to [min_val, max_val] range
 * @tparam T Data type
 * @tparam N Number of dimensions
 * @param tensor Input tensor
 * @param axis Axis along which to normalize (-1 for all elements)
 * @param min_val Minimum value of output range (default 0)
 * @param max_val Maximum value of output range (default 1)
 * @param eps Small constant to avoid division by zero
 * @return Scaled tensor in the range [min_val, max_val]
 * 
 * Min-max scaling is useful when you need values in a specific range.
 * 
 * Optimized for GPU, BLAS, and parallel CPU execution.
 */
template <typename T, size_t N>
Tensor<T, N> normalize_minmax(const Tensor<T, N>& tensor, int axis = -1, 
                               T min_val = T(0), T max_val = T(1), T eps = T(1e-8)) {
    if (axis == -1) {
        T min_elem = tensor.min();
        T max_elem = tensor.max();
        
        T range = max_elem - min_elem;
        if (range > eps) {
            T scale = (max_val - min_val) / range;
            return min_val + (tensor - min_elem) * scale;
        } else {
            T mid = (min_val + max_val) / T(2);
            Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
            result.fill(mid);
            return result;
        }
    } else if constexpr (N >= 2) {
        // Normalize along specific axis with three-tier backend
        Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
        const T* src = tensor.data_ptr();
        T* dst = result.data_ptr();
        
        auto dims = tensor.dims();
        size_t axis_size = dims[axis];
        size_t outer_size = 1;
        size_t inner_size = 1;
        
        for (size_t i = 0; i < axis; ++i) {
            outer_size *= dims[i];
        }
        for (size_t i = axis + 1; i < N; ++i) {
            inner_size *= dims[i];
        }
        
#ifdef USE_GPU
        if (tensor.uses_gpu()) {
            minmax_normalize_axis_gpu_direct(src, dst, outer_size, axis_size, inner_size,
                                             min_val, max_val, eps);
            return result;
        }
#endif

        // CPU fallback: BLAS or parallel CPU
        std::for_each(std::execution::par_unseq,
                     std::views::iota(size_t(0), outer_size).begin(),
                     std::views::iota(size_t(0), outer_size).end(),
                     [&](size_t outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                size_t first_idx = outer * axis_size * inner_size + inner;
                T min_elem = src[first_idx];
                T max_elem = src[first_idx];
                
                for (size_t ax = 1; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    if (src[idx] < min_elem) min_elem = src[idx];
                    if (src[idx] > max_elem) max_elem = src[idx];
                }
                
                T range = max_elem - min_elem;
                
                if (range > eps) {
                    T scale = (max_val - min_val) / range;
                    for (size_t ax = 0; ax < axis_size; ++ax) {
                        size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                        dst[idx] = min_val + (src[idx] - min_elem) * scale;
                    }
                } else {
                    T mid = (min_val + max_val) / T(2);
                    for (size_t ax = 0; ax < axis_size; ++ax) {
                        size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                        dst[idx] = mid;
                    }
                }
            }
        });
        return result;
    } else {
        return tensor;
    }
}

} // namespace tensor

#endif // TENSOR_NORMALIZE_H
