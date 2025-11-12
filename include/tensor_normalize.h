#ifndef TENSOR_NORMALIZE_H
#define TENSOR_NORMALIZE_H

#include "tensor_base.h"

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
    Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
    const T* src = tensor.data_ptr();
    T* dst = result.data_ptr();
    size_t total = tensor.total_size();
    
    if (axis == -1) {
        // Normalize over all elements
        T sum = T(0);
        for (size_t i = 0; i < total; ++i) {
            sum += std::abs(src[i]);
        }
        if (sum > T(0)) {
            for (size_t i = 0; i < total; ++i) {
                dst[i] = src[i] / sum;
            }
        } else {
            std::copy_n(src, total, dst);
        }
    } else if constexpr (N >= 2) {
        // Normalize along specific axis
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
        
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                // Compute sum for this slice
                T sum = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    sum += std::abs(src[idx]);
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
    } else {
        std::copy_n(src, total, dst);
    }
    
    return result;
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
 */
template <typename T, size_t N>
Tensor<T, N> normalize_l2(const Tensor<T, N>& tensor, int axis = -1) {
    Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
    const T* src = tensor.data_ptr();
    T* dst = result.data_ptr();
    size_t total = tensor.total_size();
    
    if (axis == -1) {
        // Normalize over all elements
        T sum_sq = T(0);
        for (size_t i = 0; i < total; ++i) {
            sum_sq += src[i] * src[i];
        }
        T norm = std::sqrt(sum_sq);
        if (norm > T(0)) {
            for (size_t i = 0; i < total; ++i) {
                dst[i] = src[i] / norm;
            }
        } else {
            std::copy_n(src, total, dst);
        }
    } else if constexpr (N >= 2) {
        // Normalize along specific axis
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
        
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                // Compute sum of squares for this slice
                T sum_sq = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    sum_sq += src[idx] * src[idx];
                }
                
                T norm = std::sqrt(sum_sq);
                
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
    } else {
        std::copy_n(src, total, dst);
    }
    
    return result;
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
 */
template <typename T, size_t N>
Tensor<T, N> normalize_zscore(const Tensor<T, N>& tensor, int axis = -1, T eps = T(1e-8)) {
    Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
    const T* src = tensor.data_ptr();
    T* dst = result.data_ptr();
    size_t total = tensor.total_size();
    
    if (axis == -1) {
        // Normalize over all elements
        T mean = T(0);
        for (size_t i = 0; i < total; ++i) {
            mean += src[i];
        }
        mean /= total;
        
        T var = T(0);
        for (size_t i = 0; i < total; ++i) {
            T diff = src[i] - mean;
            var += diff * diff;
        }
        var /= total;
        T std_dev = std::sqrt(var + eps);
        
        for (size_t i = 0; i < total; ++i) {
            dst[i] = (src[i] - mean) / std_dev;
        }
    } else if constexpr (N >= 2) {
        // Normalize along specific axis
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
        
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                // Compute mean for this slice
                T mean = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    mean += src[idx];
                }
                mean /= axis_size;
                
                // Compute variance
                T var = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    T diff = src[idx] - mean;
                    var += diff * diff;
                }
                var /= axis_size;
                T std_dev = std::sqrt(var + eps);
                
                // Normalize
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    dst[idx] = (src[idx] - mean) / std_dev;
                }
            }
        }
    } else {
        std::copy_n(src, total, dst);
    }
    
    return result;
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
 */
template <typename T, size_t N>
Tensor<T, N> normalize_minmax(const Tensor<T, N>& tensor, int axis = -1, 
                               T min_val = T(0), T max_val = T(1), T eps = T(1e-8)) {
    Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
    const T* src = tensor.data_ptr();
    T* dst = result.data_ptr();
    size_t total = tensor.total_size();
    
    if (axis == -1) {
        // Normalize over all elements
        T min_elem = src[0];
        T max_elem = src[0];
        for (size_t i = 1; i < total; ++i) {
            if (src[i] < min_elem) min_elem = src[i];
            if (src[i] > max_elem) max_elem = src[i];
        }
        
        T range = max_elem - min_elem;
        if (range > eps) {
            T scale = (max_val - min_val) / range;
            for (size_t i = 0; i < total; ++i) {
                dst[i] = min_val + (src[i] - min_elem) * scale;
            }
        } else {
            // All values are the same
            T mid = (min_val + max_val) / T(2);
            for (size_t i = 0; i < total; ++i) {
                dst[i] = mid;
            }
        }
    } else if constexpr (N >= 2) {
        // Normalize along specific axis
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
        
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                // Find min and max for this slice
                size_t first_idx = outer * axis_size * inner_size + inner;
                T min_elem = src[first_idx];
                T max_elem = src[first_idx];
                
                for (size_t ax = 1; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    if (src[idx] < min_elem) min_elem = src[idx];
                    if (src[idx] > max_elem) max_elem = src[idx];
                }
                
                T range = max_elem - min_elem;
                
                // Normalize
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
        }
    } else {
        std::copy_n(src, total, dst);
    }
    
    return result;
}

} // namespace tensor

#endif // TENSOR_NORMALIZE_H
