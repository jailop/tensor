#include "tensor_error.h"

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


