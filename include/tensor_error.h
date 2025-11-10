#ifndef TENSOR_ERROR_H
#define TENSOR_ERROR_H

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

/**
 * @brief Convert TensorError to human-readable string
 * @param error The error code to convert
 * @return A string description of the error
 */
inline std::string to_string(TensorError error) {
    switch (error) {
        case TensorError::DimensionMismatch:
            return "Tensor dimensions must match";
        case TensorError::ContractionMismatch:
            return "Contraction dimension must match";
        case TensorError::InvalidArgument:
            return "Invalid argument provided";
        case TensorError::SingularMatrix:
            return "Matrix is singular (non-invertible)";
        case TensorError::NotPositiveDefinite:
            return "Matrix is not positive definite";
        case TensorError::NotSquare:
            return "Operation requires square matrix";
        case TensorError::EmptyMatrix:
            return "Matrix is empty";
        case TensorError::LapackError:
            return "LAPACK routine error";
        case TensorError::NotImplemented:
            return "Feature not yet implemented";
        default:
            return "Unknown error";
    }
}

#endif // TENSOR_ERROR_H
