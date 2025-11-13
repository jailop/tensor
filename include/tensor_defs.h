#ifndef TENSOR_DEFS_H
#define TENSOR_DEFS_H

#include <string>
#include <variant>
#include <array>
#include "tensor_error.h"

namespace tensor {

/**
 * Indicates which backend is being used for tensor operations.
 * Priority order: GPU > BLAS > CPU
 */
enum class Backend {
    CPU,   ///< Standard CPU implementation
    BLAS,  ///< Optimized BLAS for CPU operations
    GPU    ///< CUDA GPU acceleration
};

/**
 * Get the name of a backend as a string
 * @param backend The backend enum value
 * @return Human-readable name of the backend
 */
std::string toString(Backend backend);

/**
 * Operations that can fail return a variant containing either the result
 * or a TensorError. Use std::holds_alternative and std::get to access.
 */
template <typename T>
using TensorResult = std::variant<T, TensorError>;

/**
 * Fixed-size array representing indices or coordinates in N-dimensional space.
 */
template <size_t N>
using TensorIndices = std::array<size_t, N>;

} // namespace tensor

#endif // TENSOR_DEFS_H
