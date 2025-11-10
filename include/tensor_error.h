#ifndef TENSOR_ERROR_H
#define TENSOR_ERROR_H

#include <string>

/**
 * @enum TensorError
 * @brief Error codes for tensor operations
 * 
 * Provides enumeration of possible errors that can occur during tensor operations.
 * These errors are returned as part of TensorResult variant type.
 */
enum class TensorError {
    DimensionMismatch,    ///< Tensor dimensions do not match for the operation
    ContractionMismatch,  ///< Contraction dimensions are incompatible
    InvalidArgument,      ///< Invalid argument provided to a function
    SingularMatrix,       ///< Matrix is singular (non-invertible)
    NotPositiveDefinite,  ///< Matrix is not positive definite
    NotSquare,            ///< Operation requires square matrix
    EmptyMatrix,          ///< Matrix is empty
    LapackError,          ///< LAPACK routine error
    NotImplemented        ///< Feature not yet implemented
};

inline std::string to_string(TensorError error);

#endif // TENSOR_ERROR_H
