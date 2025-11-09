/**
 * @file tensor_types.h
 * @brief Type aliases for common Tensor specializations
 * 
 * This header provides convenient type aliases for commonly used
 * Vector and Matrix types. These are pre-instantiated in the library
 * for faster compilation and smaller binary size.
 * 
 * @section example_types Usage Example
 * @code
 * #include "tensor_types.h"
 * using namespace tensor4d;
 * 
 * // Create a float vector
 * Vectorf v({10});
 * v.fill(1.0f);
 * 
 * // Create a double matrix
 * Matrixd M({3, 3});
 * M.fill(0.0);
 * 
 * // Create a 3D tensor
 * Tensor3f data({4, 5, 6});  // 4 channels, 5 rows, 6 cols
 * 
 * // All aliases support the full tensor API:
 * auto result = v.exp().sum();
 * auto product = M.matmul(M);
 * @endcode
 */

#ifndef TENSOR_TYPES_H
#define TENSOR_TYPES_H

#include "tensor.h"

namespace tensor4d {

/**
 * @defgroup TypeAliases Type Aliases
 * @brief Convenient type aliases for common tensor types
 * @{
 */

// ============================================================================
// Vector Type Aliases (Rank 1 Tensors)
// ============================================================================

/** @brief Float vector (1D tensor) */
using Vectorf = Tensor<float, 1>;

/** @brief Double precision vector (1D tensor) */
using Vectord = Tensor<double, 1>;

/** @brief Integer vector (1D tensor) */
using Vectori = Tensor<int, 1>;

/** @brief Long integer vector (1D tensor) */
using Vectorl = Tensor<long, 1>;

// ============================================================================
// Matrix Type Aliases (Rank 2 Tensors)
// ============================================================================

/** @brief Float matrix (2D tensor) */
using Matrixf = Tensor<float, 2>;

/** @brief Double precision matrix (2D tensor) */
using Matrixd = Tensor<double, 2>;

/** @brief Integer matrix (2D tensor) */
using Matrixi = Tensor<int, 2>;

/** @brief Long integer matrix (2D tensor) */
using Matrixl = Tensor<long, 2>;

// ============================================================================
// 3D Tensor Type Aliases
// ============================================================================

/** @brief Float 3D tensor */
using Tensor3f = Tensor<float, 3>;

/** @brief Double precision 3D tensor */
using Tensor3d = Tensor<double, 3>;

/** @brief Integer 3D tensor */
using Tensor3i = Tensor<int, 3>;

// ============================================================================
// 4D Tensor Type Aliases
// ============================================================================

/** @brief Float 4D tensor */
using Tensor4f = Tensor<float, 4>;

/** @brief Double precision 4D tensor */
using Tensor4d = Tensor<double, 4>;

/** @brief Integer 4D tensor */
using Tensor4i = Tensor<int, 4>;

/** @} */ // end of TypeAliases group

}  // namespace tensor4d

#endif  // TENSOR_TYPES_H
