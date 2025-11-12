/**
 * This header provides convenient type aliases for commonly used
 * Vector and Matrix types. These are pre-instantiated in the library
 * for faster compilation and smaller binary size.
  */

#ifndef TENSOR_TYPES_H
#define TENSOR_TYPES_H

#include "tensor.h"

namespace tensor {

// Vector Type Aliases (Rank 1 Tensors)

/** @brief Float vector (1D tensor) */
using Vectorf = Tensor<float, 1>;

/** @brief Double precision vector (1D tensor) */
using Vectord = Tensor<double, 1>;

/** @brief Integer vector (1D tensor) */
using Vectori = Tensor<int, 1>;

/** @brief Long integer vector (1D tensor) */
using Vectorl = Tensor<long, 1>;

// Matrix Type Aliases (Rank 2 Tensors)

/** @brief Float matrix (2D tensor) */
using Matrixf = Tensor<float, 2>;

/** @brief Double precision matrix (2D tensor) */
using Matrixd = Tensor<double, 2>;

/** @brief Integer matrix (2D tensor) */
using Matrixi = Tensor<int, 2>;

/** @brief Long integer matrix (2D tensor) */
using Matrixl = Tensor<long, 2>;

// 3D Tensor Type Aliases

/** @brief Float 3D tensor */
using Tensor3f = Tensor<float, 3>;

/** @brief Double precision 3D tensor */
using Tensor3d = Tensor<double, 3>;

/** @brief Integer 3D tensor */
using Tensor3i = Tensor<int, 3>;

// 4D Tensor Type Aliases

/** @brief Float 4D tensor */
using Tensor4f = Tensor<float, 4>;

/** @brief Double precision 4D tensor */
using Tensor4d = Tensor<double, 4>;

/** @brief Integer 4D tensor */
using Tensor4i = Tensor<int, 4>;

}  // namespace tensor

#endif  // TENSOR_TYPES_H
