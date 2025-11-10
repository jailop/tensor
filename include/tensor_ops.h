#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"
#include <numeric>

/**
 * @file tensor_ops.h
 * @brief Core neural network operations with broadcasting and autograd
 * 
 * This header provides essential operations for building neural networks:
 * - Matrix multiplication (matmul) with GPU/BLAS acceleration
 * - Broadcasting for element-wise operations
 * - Reduction operations (sum, mean, max, min, etc.)
 * - Softmax and log-softmax
 * - All operations support automatic differentiation
 * 
 * @author Tensor Library Team
 * @version 1.0
 * @date 2024
 * 
 * @section broadcasting Broadcasting Rules
 * Broadcasting allows operations on tensors of different shapes:
 * - Dimensions are aligned from the right
 * - Dimensions of size 1 are stretched to match
 * - Missing dimensions are treated as 1
 * 
 * Example: (3, 1, 5) + (4, 5) broadcasts to (3, 4, 5)
 * 
 * @section usage_ops Usage Example
 * @code
 * Matrix<float> A({3, 4});
 * Matrix<float> B({4, 5});
 * 
 * // Matrix multiplication with autograd
 * auto C = tensor_ops::matmul(A, B);
 * 
 * // Softmax for classification
 * auto probs = tensor_ops::softmax(C, 1);
 * @endcode
 */

/// @brief Forward declarations
template <typename T, size_t N>
class Tensor;

/**
 * @namespace tensor_ops
 * @brief Core tensor operations for neural networks
 * 
 * Provides high-level operations with:
 * - Automatic broadcasting
 * - Gradient computation (autograd)
 * - GPU and BLAS acceleration
 * - Numerical stability (e.g., for softmax)
 */
namespace tensor_ops {

/**
 * @brief Check if two tensor shapes are broadcastable
 * @tparam N1 Number of dimensions in first shape
 * @tparam N2 Number of dimensions in second shape
 * @param shape1 First tensor shape
 * @param shape2 Second tensor shape
 * @return True if shapes can be broadcast together
 * 
 * Broadcasting rules:
 * - Dimensions are compared from right to left
 * - Two dimensions are compatible if they are equal or one is 1
 */
template<size_t N1, size_t N2>
bool are_broadcastable(const TensorIndices<N1>& shape1, const TensorIndices<N2>& shape2) {
    size_t max_dims = std::max(N1, N2);
    for (size_t i = 0; i < max_dims; ++i) {
        size_t idx1 = (i < N1) ? (N1 - 1 - i) : 0;
        size_t idx2 = (i < N2) ? (N2 - 1 - i) : 0;
        
        size_t dim1 = (i < N1) ? shape1[idx1] : 1;
        size_t dim2 = (i < N2) ? shape2[idx2] : 1;
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Compute broadcasted shape for two tensors
 * @tparam N1 Number of dimensions in first shape
 * @tparam N2 Number of dimensions in second shape
 * @param shape1 First tensor shape
 * @param shape2 Second tensor shape
 * @return Broadcasted shape
 */
template<size_t N1, size_t N2>
auto broadcast_shape(const TensorIndices<N1>& shape1, const TensorIndices<N2>& shape2) {
    constexpr size_t max_dims = std::max(N1, N2);
    TensorIndices<max_dims> result;
    
    for (size_t i = 0; i < max_dims; ++i) {
        // Align from the right: rightmost dimensions are aligned first
        size_t idx1 = (i < N1) ? (N1 - 1 - i) : SIZE_MAX;
        size_t idx2 = (i < N2) ? (N2 - 1 - i) : SIZE_MAX;
        
        size_t dim1 = (idx1 != SIZE_MAX) ? shape1[idx1] : 1;
        size_t dim2 = (idx2 != SIZE_MAX) ? shape2[idx2] : 1;
        
        result[max_dims - 1 - i] = std::max(dim1, dim2);
    }
    
    return result;
}

} // namespace tensor_ops

// Add these methods to Tensor class (will be inserted into tensor.h)

// Helper template to compute result dimensions for matmul
template<size_t NA, size_t NB>
struct MatmulResultDims {
    static constexpr size_t value = NA + NB - 2;
};

#endif // TENSOR_OPS_H
