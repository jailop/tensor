/**
 * @file tensor_matrix.h
 * @brief Matrix class derived from Tensor<T, 2> with matrix-specific operations
 * 
 * This file defines the Matrix<T> class, which is a specialized 2D Tensor
 * with additional matrix-specific operations and a more convenient interface
 * for linear algebra operations.
 */

#ifndef TENSOR_MATRIX_H
#define TENSOR_MATRIX_H

#include "tensor.h"
#include <initializer_list>

namespace tensor {

/**
 * @brief Matrix class - a specialized 2D Tensor for linear algebra
 * @tparam T Data type (float, double, etc.)
 * 
 * Matrix<T> inherits from Tensor<T, 2> and provides additional matrix-specific
 * operations. All Tensor operations are available, plus matrix-specific methods.
 */
template <typename T>
class Matrix : public Tensor<T, 2> {
public:
    // Constructors
    
    /**
     * @brief Construct matrix with dimensions
     * @param rows Number of rows
     * @param cols Number of columns
     * @param use_gpu Whether to allocate on GPU
     */
    Matrix(size_t rows, size_t cols, bool use_gpu = false)
        : Tensor<T, 2>({rows, cols}, use_gpu) {}
    
    /**
     * @brief Construct matrix from TensorIndices
     * @param shape Shape as {rows, cols}
     * @param use_gpu Whether to allocate on GPU
     */
    explicit Matrix(const TensorIndices<2>& shape, bool use_gpu = false)
        : Tensor<T, 2>(shape, use_gpu) {}
    
    /**
     * @brief Construct matrix from initializer list
     * @param rows Initializer list of rows
     * @param use_gpu Whether to allocate on GPU
     */
    Matrix(std::initializer_list<std::initializer_list<T>> rows, bool use_gpu = false)
        : Tensor<T, 2>({rows.size(), rows.begin()->size()}, use_gpu) {
        size_t i = 0;
        for (const auto& row : rows) {
            size_t j = 0;
            for (const auto& val : row) {
                (*this)[{i, j}] = val;
                ++j;
            }
            ++i;
        }
    }
    
    /**
     * @brief Copy constructor
     */
    Matrix(const Matrix<T>& other) : Tensor<T, 2>(other) {}
    
    /**
     * @brief Move constructor
     */
    Matrix(Matrix<T>&& other) noexcept : Tensor<T, 2>(std::move(other)) {}
    
    /**
     * @brief Construct from Tensor<T, 2>
     */
    explicit Matrix(const Tensor<T, 2>& tensor) : Tensor<T, 2>(tensor) {}
    
    /**
     * @brief Construct from Tensor<T, 2> (move)
     */
    explicit Matrix(Tensor<T, 2>&& tensor) noexcept : Tensor<T, 2>(std::move(tensor)) {}
    
    // Assignment operators
    
    Matrix<T>& operator=(const Matrix<T>& other) {
        Tensor<T, 2>::operator=(other);
        return *this;
    }
    
    Matrix<T>& operator=(Matrix<T>&& other) noexcept {
        Tensor<T, 2>::operator=(std::move(other));
        return *this;
    }
    
    // Matrix dimensions accessors
    
    /**
     * @brief Get number of rows
     */
    size_t rows() const { return this->dims()[0]; }
    
    /**
     * @brief Get number of columns
     */
    size_t cols() const { return this->dims()[1]; }
    
    /**
     * @brief Check if matrix is square
     */
    bool is_square() const { return rows() == cols(); }
    
    // Matrix-specific operations
    
    /**
     * @brief Transpose the matrix
     * @return Transposed matrix
     */
    Matrix<T> transpose() const;
    
    /**
     * @brief Matrix-matrix multiplication
     * @param other Right-hand side matrix
     * @return Result of matrix multiplication
     */
    Matrix<T> matmul(const Matrix<T>& other) const;
    
    /**
     * @brief Matrix-vector multiplication
     * @param vec Column vector
     * @return Result vector
     */
    Tensor<T, 1> matvec(const Tensor<T, 1>& vec) const;
    
    /**
     * @brief Extract diagonal elements
     * @return Vector containing diagonal elements
     */
    Tensor<T, 1> diag() const;
    
    /**
     * @brief Extract a rectangular block
     * @param start_row Starting row index
     * @param start_col Starting column index
     * @param num_rows Number of rows to extract
     * @param num_cols Number of columns to extract
     * @return Submatrix
     */
    Matrix<T> block(size_t start_row, size_t start_col, 
                    size_t num_rows, size_t num_cols) const;
    
    /**
     * @brief Extract a row as a vector
     * @param row_idx Row index
     * @return Row vector
     */
    Tensor<T, 1> row(size_t row_idx) const;
    
    /**
     * @brief Extract a column as a vector
     * @param col_idx Column index
     * @return Column vector
     */
    Tensor<T, 1> col(size_t col_idx) const;
    
    /**
     * @brief Compute trace (sum of diagonal elements)
     * @return Trace value
     */
    T trace() const;
    
    /**
     * @brief Compute Frobenius norm
     * @return Frobenius norm value
     */
    T frobenius_norm() const;
    
    /**
     * @brief Compute L1 norm (max column sum)
     * @return L1 norm value
     */
    T norm_l1() const;
    
    /**
     * @brief Compute infinity norm (max row sum)
     * @return Infinity norm value
     */
    T norm_inf() const;
    
    /**
     * @brief Compute matrix rank
     * @param tol Tolerance for rank determination
     * @return Rank of the matrix
     */
    size_t rank(T tol = T(-1)) const;
    
    /**
     * @brief Compute condition number
     * @return Condition number
     */
    T condition_number() const;
    
    /**
     * @brief Check if matrix is symmetric
     * @param tol Tolerance for symmetry check
     * @return true if symmetric
     */
    bool is_symmetric(T tol = T(1e-10)) const;
    
    /**
     * @brief Check if matrix is positive definite
     * @return true if positive definite
     */
    bool is_positive_definite() const;
    
    // Static factory methods
    
    /**
     * @brief Create identity matrix
     * @param n Size of the square matrix
     * @param use_gpu Whether to use GPU
     * @return Identity matrix
     */
    static Matrix<T> eye(size_t n, bool use_gpu = false);
    
    /**
     * @brief Create matrix from diagonal vector
     * @param diagonal Diagonal elements
     * @return Diagonal matrix
     */
    static Matrix<T> from_diag(const Tensor<T, 1>& diagonal);
    
    /**
     * @brief Create zero matrix
     * @param rows Number of rows
     * @param cols Number of columns
     * @param use_gpu Whether to use GPU
     * @return Zero matrix
     */
    static Matrix<T> zeros(size_t rows, size_t cols, bool use_gpu = false);
    
    /**
     * @brief Create matrix filled with ones
     * @param rows Number of rows
     * @param cols Number of columns
     * @param use_gpu Whether to use GPU
     * @return Matrix of ones
     */
    static Matrix<T> ones(size_t rows, size_t cols, bool use_gpu = false);
    
    /**
     * @brief Create matrix with random values
     * @param rows Number of rows
     * @param cols Number of columns
     * @param use_gpu Whether to use GPU
     * @return Random matrix
     */
    static Matrix<T> randn(size_t rows, size_t cols, bool use_gpu = false);
};

// Implementation of member functions

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    auto dims = this->dims();
    size_t m = dims[0];
    size_t n = dims[1];
    
    Matrix<T> result(n, m, this->uses_gpu());
    
    const T* src = this->begin();
    T* dst = result.begin();
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            dst[j * m + i] = src[i * n + j];
        }
    }
    
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::matmul(const Matrix<T>& other) const {
    size_t m = this->rows();
    size_t k = this->cols();
    size_t n = other.cols();
    
    if (k != other.rows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix<T> result(m, n, this->uses_gpu());
    
    const T* a_data = this->begin();
    const T* b_data = other.begin();
    T* c_data = result.begin();
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t p = 0; p < k; ++p) {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
    
    return result;
}

template <typename T>
Tensor<T, 1> Matrix<T>::matvec(const Tensor<T, 1>& vec) const {
    size_t m = this->rows();
    size_t n = this->cols();
    
    if (n != vec.dims()[0]) {
        throw std::invalid_argument("Matrix-vector dimensions incompatible");
    }
    
    Tensor<T, 1> result({m}, this->uses_gpu());
    
    const T* mat_data = this->begin();
    const T* vec_data = vec.begin();
    T* res_data = result.begin();
    
    for (size_t i = 0; i < m; ++i) {
        T sum = T(0);
        for (size_t j = 0; j < n; ++j) {
            sum += mat_data[i * n + j] * vec_data[j];
        }
        res_data[i] = sum;
    }
    
    return result;
}

template <typename T>
Tensor<T, 1> Matrix<T>::diag() const {
    size_t m = this->rows();
    size_t n = this->cols();
    size_t diag_size = std::min(m, n);
    
    Tensor<T, 1> result({diag_size}, this->uses_gpu());
    
    const T* src = this->begin();
    T* dst = result.begin();
    
    for (size_t i = 0; i < diag_size; ++i) {
        dst[i] = src[i * n + i];
    }
    
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::block(size_t start_row, size_t start_col,
                           size_t num_rows, size_t num_cols) const {
    if (start_row + num_rows > this->rows() || start_col + num_cols > this->cols()) {
        throw std::out_of_range("Block exceeds matrix bounds");
    }
    
    Matrix<T> result(num_rows, num_cols, this->uses_gpu());
    
    const T* src = this->begin();
    T* dst = result.begin();
    size_t src_cols = this->cols();
    
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            dst[i * num_cols + j] = src[(start_row + i) * src_cols + (start_col + j)];
        }
    }
    
    return result;
}

template <typename T>
Tensor<T, 1> Matrix<T>::row(size_t row_idx) const {
    if (row_idx >= this->rows()) {
        throw std::out_of_range("Row index out of bounds");
    }
    
    size_t n = this->cols();
    Tensor<T, 1> result({n}, this->uses_gpu());
    
    const T* src = this->begin();
    T* dst = result.begin();
    
    for (size_t j = 0; j < n; ++j) {
        dst[j] = src[row_idx * n + j];
    }
    
    return result;
}

template <typename T>
Tensor<T, 1> Matrix<T>::col(size_t col_idx) const {
    if (col_idx >= this->cols()) {
        throw std::out_of_range("Column index out of bounds");
    }
    
    size_t m = this->rows();
    size_t n = this->cols();
    Tensor<T, 1> result({m}, this->uses_gpu());
    
    const T* src = this->begin();
    T* dst = result.begin();
    
    for (size_t i = 0; i < m; ++i) {
        dst[i] = src[i * n + col_idx];
    }
    
    return result;
}

template <typename T>
T Matrix<T>::trace() const {
    if (!is_square()) {
        throw std::invalid_argument("Trace requires square matrix");
    }
    
    size_t n = this->rows();
    const T* data = this->begin();
    
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) {
        sum += data[i * n + i];
    }
    
    return sum;
}

template <typename T>
T Matrix<T>::frobenius_norm() const {
    size_t m = this->rows();
    size_t n = this->cols();
    const T* data = this->begin();
    
    T sum = T(0);
    for (size_t i = 0; i < m * n; ++i) {
        sum += data[i] * data[i];
    }
    
    return std::sqrt(sum);
}

template <typename T>
T Matrix<T>::norm_l1() const {
    size_t m = this->rows();
    size_t n = this->cols();
    const T* data = this->begin();
    
    T max_sum = T(0);
    for (size_t j = 0; j < n; ++j) {
        T col_sum = T(0);
        for (size_t i = 0; i < m; ++i) {
            col_sum += std::abs(data[i * n + j]);
        }
        max_sum = std::max(max_sum, col_sum);
    }
    
    return max_sum;
}

template <typename T>
T Matrix<T>::norm_inf() const {
    size_t m = this->rows();
    size_t n = this->cols();
    const T* data = this->begin();
    
    T max_sum = T(0);
    for (size_t i = 0; i < m; ++i) {
        T row_sum = T(0);
        for (size_t j = 0; j < n; ++j) {
            row_sum += std::abs(data[i * n + j]);
        }
        max_sum = std::max(max_sum, row_sum);
    }
    
    return max_sum;
}

template <typename T>
size_t Matrix<T>::rank(T tol) const {
    // Simple rank computation using Gaussian elimination
    // For production, use SVD from linalg_advanced.h
    Matrix<T> A = *this;
    size_t m = A.rows();
    size_t n = A.cols();
    
    if (tol < 0) {
        tol = std::max(m, n) * std::numeric_limits<T>::epsilon();
    }
    
    size_t rank = 0;
    T* data = A.begin();
    
    for (size_t col = 0; col < std::min(m, n); ++col) {
        // Find pivot
        size_t pivot_row = col;
        T max_val = std::abs(data[col * n + col]);
        
        for (size_t row = col + 1; row < m; ++row) {
            T val = std::abs(data[row * n + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = row;
            }
        }
        
        if (max_val < tol) continue;
        
        rank++;
        
        // Swap rows if needed
        if (pivot_row != col) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(data[col * n + j], data[pivot_row * n + j]);
            }
        }
        
        // Eliminate below pivot
        T pivot = data[col * n + col];
        for (size_t row = col + 1; row < m; ++row) {
            T factor = data[row * n + col] / pivot;
            for (size_t j = col; j < n; ++j) {
                data[row * n + j] -= factor * data[col * n + j];
            }
        }
    }
    
    return rank;
}

template <typename T>
T Matrix<T>::condition_number() const {
    // Simplified condition number estimate
    // For production, use SVD from linalg_advanced.h
    T norm = this->frobenius_norm();
    if (norm < std::numeric_limits<T>::epsilon()) {
        return std::numeric_limits<T>::infinity();
    }
    
    // This is a rough estimate
    return norm * norm;
}

template <typename T>
bool Matrix<T>::is_symmetric(T tol) const {
    if (!is_square()) return false;
    
    size_t n = this->rows();
    const T* data = this->begin();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(data[i * n + j] - data[j * n + i]) > tol) {
                return false;
            }
        }
    }
    
    return true;
}

template <typename T>
bool Matrix<T>::is_positive_definite() const {
    if (!is_square()) return false;
    if (!is_symmetric()) return false;
    
    size_t n = this->rows();
    const T* data = this->begin();
    
    // Check diagonal elements are positive
    for (size_t i = 0; i < n; ++i) {
        if (data[i * n + i] <= T(0)) return false;
    }
    
    return true;
}

template <typename T>
Matrix<T> Matrix<T>::eye(size_t n, bool use_gpu) {
    Matrix<T> result(n, n, use_gpu);
    result.fill(T(0));
    
    T* data = result.begin();
    for (size_t i = 0; i < n; ++i) {
        data[i * n + i] = T(1);
    }
    
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::from_diag(const Tensor<T, 1>& diagonal) {
    size_t n = diagonal.dims()[0];
    Matrix<T> result(n, n, diagonal.uses_gpu());
    result.fill(T(0));
    
    const T* src = diagonal.begin();
    T* dst = result.begin();
    
    for (size_t i = 0; i < n; ++i) {
        dst[i * n + i] = src[i];
    }
    
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::zeros(size_t rows, size_t cols, bool use_gpu) {
    Matrix<T> result(rows, cols, use_gpu);
    result.fill(T(0));
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::ones(size_t rows, size_t cols, bool use_gpu) {
    Matrix<T> result(rows, cols, use_gpu);
    result.fill(T(1));
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::randn(size_t rows, size_t cols, bool use_gpu) {
    Matrix<T> result(rows, cols, use_gpu);
    
    // Fill with random normal distribution
    T* data = result.begin();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(T(0), T(1));
    
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] = dist(gen);
    }
    
    return result;
}

} // namespace tensor

#endif // TENSOR_MATRIX_H
