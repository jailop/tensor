/**
 * @brief Explicit template instantiations for Matrix and Tensor types
 */

#include "tensor.h"
#include "linalg.h"

namespace tensor {

/**
 * @name Matrix Instantiations
 * @brief Explicit instantiations for 2D tensors (matrices)
 * @{
 */

/** Single precision float matrix */
template class Tensor<float, 2>;

/** Double precision float matrix */
template class Tensor<double, 2>;

/** @} */

/**
 * @name 3D Tensor Instantiations
 * @brief Explicit instantiations for 3D tensors
 * @{
 */

/** Single precision 3D tensor */
template class Tensor<float, 3>;

/** Double precision 3D tensor */
template class Tensor<double, 3>;

/** @} */

/**
 * @name 4D Tensor Instantiations
 * @brief Explicit instantiations for 4D tensors
 * @{
 */

/** Single precision 4D tensor */
template class Tensor<float, 4>;

/** Double precision 4D tensor */
template class Tensor<double, 4>;

} // namespace tensor
