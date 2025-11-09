/**
 * @file tensor_instantiations.cc
 * @brief Explicit template instantiations for Matrix and Tensor types
 * @ingroup TypeAliases
 * 
 * This file provides explicit instantiations of the Tensor template for
 * common Matrix (rank 2) and higher-dimensional types. This allows the library
 * to be compiled into static and shared libraries.
 * 
 * @section inst_why Why Explicit Instantiations?
 * 
 * Explicit template instantiations allow us to:
 * - Compile template code once into the library (static/shared)
 * - Reduce compilation time for client code
 * - Reduce executable size by avoiding template bloat
 * - Distribute binary libraries without exposing all implementation details
 * 
 * @section inst_what What is Instantiated?
 * 
 * The following types are explicitly instantiated and available in the library:
 * 
 * **Matrix Types (Rank 2):**
 * - Tensor<float, 2> - Single precision matrix (Matrixf)
 * - Tensor<double, 2> - Double precision matrix (Matrixd)
 * 
 * **3D Tensors (Rank 3):**
 * - Tensor<float, 3> - Single precision 3D tensor (Tensor3f)
 * - Tensor<double, 3> - Double precision 3D tensor (Tensor3d)
 * 
 * **4D Tensors (Rank 4):**
 * - Tensor<float, 4> - Single precision 4D tensor (Tensor4f)
 * - Tensor<double, 4> - Double precision 4D tensor (Tensor4d)
 * 
 * @section inst_vectors Why No Vector Instantiations?
 * 
 * We do not instantiate 1D tensors (Vectors) because many operations
 * (transpose, vstack, hstack) require N >= 2 and would trigger static assertions.
 * Vectors remain available as header-only templates and work seamlessly with
 * the type aliases (Vectorf, Vectord, etc.).
 * 
 * @see @ref instantiations for detailed documentation
 * @see tensor_types.h for type aliases
 */

#include "tensor.h"
#include "linalg.h"

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

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

/** @} */
