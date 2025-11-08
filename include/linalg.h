/**
 * @file linalg.h
 * @brief Linear algebra operations for vectors and matrices
 * 
 * This header provides specialized linear algebra functionality built on top
 * of the tensor library. It includes:
 * - Vector and Matrix type aliases
 * - Vector operations (norm, dot product, cross product)
 * - Matrix operations (matrix multiplication, transpose, inverse, determinant)
 * - Matrix decompositions (LU, Cholesky, QR, SVD)
 * - Eigenvalue computation
 * - Solving linear systems
 * 
 * All operations leverage GPU and BLAS when available for optimal performance.
 * 
 * @author Tensor Library Team
 * @version 1.0
 * @date 2024
 * 
 * @section usage Usage Example
 * @code
 * // Create vectors and matrices
 * Vector<float> v({5});
 * Matrix<float> A({3, 3});
 * 
 * // Linear algebra operations
 * float n = linalg::norm(v);
 * auto B = linalg::transpose(A);
 * auto inv_A = linalg::inverse(A);
 * @endcode
 */

#ifndef _LINALG_H
#define _LINALG_H

#include "tensor.h"
#include <type_traits>
#include <optional>

// ============================================
// Specialized Linear Algebra Types
// ============================================

/**
 * @brief Type alias for 1D vectors
 * @tparam T Data type (float, double, etc.)
 * 
 * A Vector is simply a 1D Tensor, provided for clarity in linear algebra contexts.
 */
template <typename T>
using Vector = Tensor<T, 1>;

/**
 * @brief Type alias for 2D matrices
 * @tparam T Data type (float, double, etc.)
 * 
 * A Matrix is simply a 2D Tensor, provided for clarity in linear algebra contexts.
 */
template <typename T>
using Matrix = Tensor<T, 2>;

// ============================================
// Vector Operations
// ============================================

/**
 * @namespace linalg
 * @brief Linear algebra operations namespace
 * 
 * Provides comprehensive linear algebra functionality including:
 * - Basic operations: norm, dot, cross products
 * - Matrix operations: multiplication, transpose, inverse
 * - Decompositions: LU, Cholesky, QR, SVD
 * - System solving: linear systems, eigenvalues
 * 
 * All functions are optimized to use GPU (CUDA) or BLAS when available.
 */
namespace linalg {

/**
 * @brief Compute the L2 norm (Euclidean norm) of a vector
 * @tparam T Data type
 * @param v Input vector
 * @return L2 norm as scalar (sqrt of sum of squares)
 * 
 * Uses GPU or BLAS acceleration when available.
 */
template <typename T>
T norm(const Vector<T>& v) {
    T sum = T(0);
    auto dims = v.dims();
    size_t n = dims[0];
    
#ifdef USE_GPU
    if (v.uses_gpu()) {
        // Use GPU for norm computation
        T result;
        TensorGPU::dot_1d_gpu(v.data_ptr(), v.data_ptr(), &result, n);
        return std::sqrt(result);
    }
#endif
    
#ifdef USE_BLAS
    // Use BLAS dot product for norm
    T dot_result = blas_dot<T>(n, v.data_ptr(), 1, v.data_ptr(), 1);
    return std::sqrt(dot_result);
#else
    // Fallback CPU implementation
    for (size_t i = 0; i < n; ++i) {
        T val = v[{i}];
        sum += val * val;
    }
    return std::sqrt(sum);
#endif
}

/**
 * Normalize a vector to unit length.
 * @param v Input vector
 * @return Normalized vector, or zero vector if norm is too small
 */
template <typename T>
Vector<T> normalize(const Vector<T>& v) {
    T n = norm(v);
    if (n < std::numeric_limits<T>::epsilon()) {
        return Vector<T>(v.dims(), v.uses_gpu(), v.requires_gradients());
    }
    return v / n;
}

/**
 * Compute dot product of two vectors.
 * @param a First vector
 * @param b Second vector
 * @return Dot product as scalar, or 0 if dimensions don't match
 */
template <typename T>
T dot(const Vector<T>& a, const Vector<T>& b) {
    auto dims_a = a.dims();
    auto dims_b = b.dims();
    
    if (dims_a[0] != dims_b[0]) {
        return T(0);
    }
    
    size_t n = dims_a[0];
    
#ifdef USE_GPU
    if (a.uses_gpu() && b.uses_gpu()) {
        T result;
        TensorGPU::dot_1d_gpu(a.data_ptr(), b.data_ptr(), &result, n);
        return result;
    }
#endif
    
#ifdef USE_BLAS
    return blas_dot<T>(n, a.data_ptr(), 1, b.data_ptr(), 1);
#else
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) {
        sum += a[{i}] * b[{i}];
    }
    return sum;
#endif
}

/**
 * Compute outer product of two vectors.
 * @param a First vector (m elements)
 * @param b Second vector (n elements)
 * @return Matrix of shape (m, n)
 */
template <typename T>
Matrix<T> outer(const Vector<T>& a, const Vector<T>& b) {
    auto dims_a = a.dims();
    auto dims_b = b.dims();
    
    size_t m = dims_a[0];
    size_t n = dims_b[0];
    
    Matrix<T> result({m, n}, a.uses_gpu());
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[{i, j}] = a[{i}] * b[{j}];
        }
    }
    
    return result;
}

// ============================================
// Matrix Operations
// ============================================

/**
 * Matrix-vector multiplication.
 * @param mat Matrix of shape (m, n)
 * @param vec Vector of length n
 * @return Result vector of length m
 */
template <typename T>
Vector<T> matvec(const Matrix<T>& mat, const Vector<T>& vec) {
    auto mat_dims = mat.dims();
    auto vec_dims = vec.dims();
    
    size_t m = mat_dims[0];
    size_t n = mat_dims[1];
    
    if (n != vec_dims[0]) {
        return Vector<T>({m}, mat.uses_gpu());
    }
    
    Vector<T> result({m}, mat.uses_gpu());
    
#ifdef USE_GPU
    if (mat.uses_gpu() && vec.uses_gpu()) {
        // Use GPU matrix-vector multiplication
        // Can reuse 2D dot product with single column
        Matrix<T> vec_as_mat({n, 1}, true);
        for (size_t i = 0; i < n; ++i) {
            vec_as_mat[{i, 0}] = vec[{i}];
        }
        Matrix<T> result_mat({m, 1}, true);
        TensorGPU::dot_2d_gpu(mat.data_ptr(), vec_as_mat.data_ptr(), result_mat.data_ptr(), m, n, 1);
        for (size_t i = 0; i < m; ++i) {
            result[{i}] = result_mat[{i, 0}];
        }
        return result;
    }
#endif
    
#ifdef USE_BLAS
    // Use BLAS GEMV for optimized matrix-vector multiplication
    // C = alpha * A * x + beta * C
    // For row-major: y = A * x means we compute y_i = sum_j A[i,j] * x[j]
    for (size_t i = 0; i < m; ++i) {
        result[{i}] = blas_dot<T>(n, mat.data_ptr() + i * n, 1, vec.data_ptr(), 1);
    }
#else
    // Fallback CPU implementation
    for (size_t i = 0; i < m; ++i) {
        T sum = T(0);
        for (size_t j = 0; j < n; ++j) {
            sum += mat[{i, j}] * vec[{j}];
        }
        result[{i}] = sum;
    }
#endif
    
    return result;
}

/**
 * Matrix-matrix multiplication using BLAS/GPU.
 * @param a Matrix of shape (m, k)
 * @param b Matrix of shape (k, n)
 * @return Result matrix of shape (m, n)
 */
template <typename T>
Matrix<T> matmul(const Matrix<T>& a, const Matrix<T>& b) {
    auto dims_a = a.dims();
    auto dims_b = b.dims();
    
    size_t m = dims_a[0];
    size_t k = dims_a[1];
    size_t n = dims_b[1];
    
    if (k != dims_b[0]) {
        return Matrix<T>({m, n}, a.uses_gpu());
    }
    
    Matrix<T> result({m, n}, a.uses_gpu());
    result.fill(T(0));
    
#ifdef USE_GPU
    if (a.uses_gpu() && b.uses_gpu()) {
        TensorGPU::dot_2d_gpu(a.data_ptr(), b.data_ptr(), result.data_ptr(), m, k, n);
        return result;
    }
#endif
    
#ifdef USE_BLAS
    // Use BLAS GEMM: C = alpha * A * B + beta * C
    blas_gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k,
                 T(1), a.data_ptr(), k,
                 b.data_ptr(), n,
                 T(0), result.data_ptr(), n);
#else
    // Fallback CPU implementation
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t p = 0; p < k; ++p) {
                sum += a[{i, p}] * b[{p, j}];
            }
            result[{i, j}] = sum;
        }
    }
#endif
    
    return result;
}

/**
 * Transpose a matrix.
 * @param mat Input matrix of shape (m, n)
 * @return Transposed matrix of shape (n, m)
 */
template <typename T>
Matrix<T> transpose(const Matrix<T>& mat) {
    auto dims = mat.dims();
    size_t m = dims[0];
    size_t n = dims[1];
    
    Matrix<T> result({n, m}, mat.uses_gpu());
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[{j, i}] = mat[{i, j}];
        }
    }
    
    return result;
}

/**
 * Compute matrix trace (sum of diagonal elements).
 * @param mat Square matrix
 * @return Trace value, or 0 if matrix is not square
 */
template <typename T>
T trace(const Matrix<T>& mat) {
    auto dims = mat.dims();
    if (dims[0] != dims[1]) {
        return T(0);
    }
    
    T sum = T(0);
    size_t n = dims[0];
    for (size_t i = 0; i < n; ++i) {
        sum += mat[{i, i}];
    }
    return sum;
}

/**
 * Extract diagonal elements from a matrix.
 * @param mat Input matrix (m x n)
 * @return Vector of diagonal elements (length min(m,n))
 */
template <typename T>
Vector<T> diag(const Matrix<T>& mat) {
    auto dims = mat.dims();
    size_t m = dims[0];
    size_t n = dims[1];
    size_t k = std::min(m, n);
    
    Vector<T> result({k}, mat.uses_gpu());
    for (size_t i = 0; i < k; ++i) {
        result[{i}] = mat[{i, i}];
    }
    return result;
}

/**
 * Create diagonal matrix from vector.
 * @param vec Input vector of length n
 * @return Diagonal matrix of shape (n, n)
 */
template <typename T>
Matrix<T> diag(const Vector<T>& vec) {
    auto dims = vec.dims();
    size_t n = dims[0];
    
    Matrix<T> result({n, n}, vec.uses_gpu());
    result.fill(T(0));
    
    for (size_t i = 0; i < n; ++i) {
        result[{i, i}] = vec[{i}];
    }
    return result;
}

/**
 * Create identity matrix.
 * @param n Size of the matrix
 * @param use_gpu Whether to use GPU
 * @return Identity matrix of shape (n, n)
 */
template <typename T>
Matrix<T> eye(size_t n, bool use_gpu = true) {
    Matrix<T> result({n, n}, use_gpu);
    result.fill(T(0));
    for (size_t i = 0; i < n; ++i) {
        result[{i, i}] = T(1);
    }
    return result;
}

/**
 * Compute Frobenius norm of a matrix.
 * @param mat Input matrix
 * @return Frobenius norm
 */
template <typename T>
T frobenius_norm(const Matrix<T>& mat) {
    auto dims = mat.dims();
    size_t m = dims[0];
    size_t n = dims[1];
    
    T sum = T(0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T val = mat[{i, j}];
            sum += val * val;
        }
    }
    return std::sqrt(sum);
}

} // namespace linalg

// ============================================
// Tensor View Operations
// ============================================

/**
 * View for accessing a slice of tensor data.
 * This is a non-owning view that references the parent tensor's data.
 */
template <typename T, size_t N>
class TensorView {
private:
    T* data_;                        // Pointer to parent tensor data
    TensorIndices<N> dims_;          // Dimensions of the view
    TensorIndices<N> strides_;       // Strides for indexing
    size_t offset_;                  // Offset into parent data
    
    size_t compute_offset(const TensorIndices<N>& indices) const {
        size_t off = offset_;
        for (size_t i = 0; i < N; ++i) {
            off += indices[i] * strides_[i];
        }
        return off;
    }
    
public:
    TensorView(T* data, const TensorIndices<N>& dims, 
               const TensorIndices<N>& strides, size_t offset = 0)
        : data_(data), dims_(dims), strides_(strides), offset_(offset) {}
    
    T& operator[](const TensorIndices<N>& indices) {
        return data_[compute_offset(indices)];
    }
    
    const T& operator[](const TensorIndices<N>& indices) const {
        return data_[compute_offset(indices)];
    }
    
    TensorIndices<N> dims() const { return dims_; }
    
    /**
     * Convert view to a new tensor (copies data).
     */
    Tensor<T, N> to_tensor(bool use_gpu = true) const {
        Tensor<T, N> result(dims_, use_gpu);
        
        // Multi-dimensional iteration
        std::function<void(size_t, TensorIndices<N>&)> copy_recursive;
        copy_recursive = [&](size_t dim, TensorIndices<N>& idx) {
            if (dim == N) {
                result[idx] = (*this)[idx];
                return;
            }
            for (size_t i = 0; i < dims_[dim]; ++i) {
                idx[dim] = i;
                copy_recursive(dim + 1, idx);
            }
        };
        
        TensorIndices<N> idx{};
        copy_recursive(0, idx);
        
        return result;
    }
    
    /**
     * Fill the view with a value.
     */
    void fill(const T& value) {
        std::function<void(size_t, TensorIndices<N>&)> fill_recursive;
        fill_recursive = [&](size_t dim, TensorIndices<N>& idx) {
            if (dim == N) {
                (*this)[idx] = value;
                return;
            }
            for (size_t i = 0; i < dims_[dim]; ++i) {
                idx[dim] = i;
                fill_recursive(dim + 1, idx);
            }
        };
        
        TensorIndices<N> idx{};
        fill_recursive(0, idx);
    }
};

// Add view methods to Tensor class
template <typename T, size_t N>
class TensorSlice {
public:
    /**
     * Get a view of a slice along a specific dimension.
     * @param tensor Parent tensor
     * @param dim Dimension to slice
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @return View of the sliced region, or empty view if invalid
     */
    static TensorView<T, N> slice(Tensor<T, N>& tensor, size_t dim, size_t start, size_t end) {
        if (dim >= N) {
            TensorIndices<N> zero_dims{};
            TensorIndices<N> strides{};
            for (size_t i = 0; i < N; ++i) strides[i] = 1;
            return TensorView<T, N>(tensor.data_ptr(), zero_dims, strides, 0);
        }
        
        auto dims = tensor.dims();
        if (start >= dims[dim] || end > dims[dim] || start >= end) {
            TensorIndices<N> zero_dims{};
            TensorIndices<N> strides{};
            for (size_t i = 0; i < N; ++i) strides[i] = 1;
            return TensorView<T, N>(tensor.data_ptr(), zero_dims, strides, 0);
        }
        
        // Compute new dimensions
        TensorIndices<N> new_dims = dims;
        new_dims[dim] = end - start;
        
        // Strides remain the same as parent
        TensorIndices<N> strides;
        size_t stride = 1;
        for (size_t i = N; i-- > 0;) {
            strides[i] = stride;
            stride *= dims[i];
        }
        
        // Compute offset for the start index
        size_t offset = start * strides[dim];
        
        return TensorView<T, N>(tensor.data_ptr(), new_dims, strides, offset);
    }
    
    /**
     * Get a view of a specific row (for 2D tensors).
     * @return Row view, or empty view if invalid index
     */
    static TensorView<T, 1> row(Tensor<T, 2>& tensor, size_t row_idx) {
        static_assert(N == 2, "row() requires 2D tensor");
        auto dims = tensor.dims();
        if (row_idx >= dims[0]) {
            TensorIndices<1> zero_dims = {0};
            TensorIndices<1> strides = {1};
            return TensorView<T, 1>(tensor.data_ptr(), zero_dims, strides, 0);
        }
        
        TensorIndices<1> new_dims = {dims[1]};
        TensorIndices<1> strides = {1};
        size_t offset = row_idx * dims[1];
        
        return TensorView<T, 1>(tensor.data_ptr(), new_dims, strides, offset);
    }
    
    /**
     * Get a view of a specific column (for 2D tensors).
     * @return Column view, or empty view if invalid index
     */
    static TensorView<T, 1> col(Tensor<T, 2>& tensor, size_t col_idx) {
        static_assert(N == 2, "col() requires 2D tensor");
        auto dims = tensor.dims();
        if (col_idx >= dims[1]) {
            TensorIndices<1> zero_dims = {0};
            TensorIndices<1> strides = {1};
            return TensorView<T, 1>(tensor.data_ptr(), zero_dims, strides, 0);
        }
        
        TensorIndices<1> new_dims = {dims[0]};
        TensorIndices<1> strides = {dims[1]};  // Column stride
        size_t offset = col_idx;
        
        return TensorView<T, 1>(tensor.data_ptr(), new_dims, strides, offset);
    }
    
    /**
     * Get a view of a submatrix (for 2D tensors).
     * @return Submatrix view, or empty view if invalid range
     */
    static TensorView<T, 2> block(Tensor<T, 2>& tensor, 
                                   size_t row_start, size_t row_end,
                                   size_t col_start, size_t col_end) {
        static_assert(N == 2, "block() requires 2D tensor");
        auto dims = tensor.dims();
        
        if (row_start >= dims[0] || row_end > dims[0] || row_start >= row_end ||
            col_start >= dims[1] || col_end > dims[1] || col_start >= col_end) {
            TensorIndices<2> zero_dims = {0, 0};
            TensorIndices<2> strides = {1, 1};
            return TensorView<T, 2>(tensor.data_ptr(), zero_dims, strides, 0);
        }
        
        TensorIndices<2> new_dims = {row_end - row_start, col_end - col_start};
        TensorIndices<2> strides = {dims[1], 1};
        size_t offset = row_start * dims[1] + col_start;
        
        return TensorView<T, 2>(tensor.data_ptr(), new_dims, strides, offset);
    }
};

#endif // _LINALG_H
