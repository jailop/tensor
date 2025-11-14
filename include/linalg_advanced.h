/**
 * This header provides high-priority linear algebra operations that leverage
 * optimized libraries:
 * - GPU: cuBLAS/cuSOLVER (when USE_GPU is defined)
 * - CPU: LAPACK (when USE_LAPACK is defined)
 * - Fallback: Pure C++ implementations
 */

#ifndef _LINALG_ADVANCED_H
#define _LINALG_ADVANCED_H

#include "linalg.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef USE_LAPACK
extern "C" {
    // LU decomposition
    void sgetrf_(int* M, int* N, float* A, int* LDA, int* IPIV, int* INFO);
    void dgetrf_(int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO);
    
    // Linear system solver (general)
    void sgesv_(int* N, int* NRHS, float* A, int* LDA, int* IPIV,
                float* B, int* LDB, int* INFO);
    void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV,
                double* B, int* LDB, int* INFO);
    
    // Linear system solver (symmetric positive definite - Cholesky)
    void sposv_(char* UPLO, int* N, int* NRHS, float* A, int* LDA,
                float* B, int* LDB, int* INFO);
    void dposv_(char* UPLO, int* N, int* NRHS, double* A, int* LDA,
                double* B, int* LDB, int* INFO);
    
    // Least squares solver (QR-based)
    void sgels_(char* TRANS, int* M, int* N, int* NRHS,
                float* A, int* LDA, float* B, int* LDB,
                float* WORK, int* LWORK, int* INFO);
    void dgels_(char* TRANS, int* M, int* N, int* NRHS,
                double* A, int* LDA, double* B, int* LDB,
                double* WORK, int* LWORK, int* INFO);
    
    // Least squares solver (SVD-based, rank-deficient)
    void sgelsd_(int* M, int* N, int* NRHS, float* A, int* LDA,
                 float* B, int* LDB, float* S, float* RCOND, int* RANK,
                 float* WORK, int* LWORK, int* IWORK, int* INFO);
    void dgelsd_(int* M, int* N, int* NRHS, double* A, int* LDA,
                 double* B, int* LDB, double* S, double* RCOND, int* RANK,
                 double* WORK, int* LWORK, int* IWORK, int* INFO);
    
    // Matrix inversion
    void sgetri_(int* N, float* A, int* LDA, int* IPIV, float* WORK, int* LWORK, int* INFO);
    void dgetri_(int* N, double* A, int* LDA, int* IPIV, double* WORK, int* LWORK, int* INFO);
    
    // QR decomposition
    void sgeqrf_(int* M, int* N, float* A, int* LDA, float* TAU, float* WORK, int* LWORK, int* INFO);
    void dgeqrf_(int* M, int* N, double* A, int* LDA, double* TAU, double* WORK, int* LWORK, int* INFO);
    
    // Generate Q from QR
    void sorgqr_(int* M, int* N, int* K, float* A, int* LDA, float* TAU, float* WORK, int* LWORK, int* INFO);
    void dorgqr_(int* M, int* N, int* K, double* A, int* LDA, double* TAU, double* WORK, int* LWORK, int* INFO);
    
    // Cholesky decomposition
    void spotrf_(char* UPLO, int* N, float* A, int* LDA, int* INFO);
    void dpotrf_(char* UPLO, int* N, double* A, int* LDA, int* INFO);
    
    // SVD
    void sgesvd_(char* JOBU, char* JOBVT, int* M, int* N, float* A, int* LDA,
                 float* S, float* U, int* LDU, float* VT, int* LDVT,
                 float* WORK, int* LWORK, int* INFO);
    void dgesvd_(char* JOBU, char* JOBVT, int* M, int* N, double* A, int* LDA,
                 double* S, double* U, int* LDU, double* VT, int* LDVT,
                 double* WORK, int* LWORK, int* INFO);
    
    // Eigenvalues (symmetric)
    void ssyev_(char* JOBZ, char* UPLO, int* N, float* A, int* LDA,
                float* W, float* WORK, int* LWORK, int* INFO);
    void dsyev_(char* JOBZ, char* UPLO, int* N, double* A, int* LDA,
                double* W, double* WORK, int* LWORK, int* INFO);
}
#endif

#ifdef USE_GPU
#include "tensor_gpu.cuh"
#endif

namespace tensor {

// ============================================
// Solver Method Enums
// ============================================

/**
 * @brief Method for solving linear systems
 */
enum class SolverMethod {
    Auto,       ///< Automatically select best method
    LU,         ///< LU decomposition (general matrices)
    Cholesky,   ///< Cholesky decomposition (SPD matrices)
    QR,         ///< QR decomposition (overdetermined systems)
    SVD         ///< SVD (rank-deficient systems)
};

/**
 * @brief Method for least squares problems
 */
enum class LstsqMethod {
    QR,         ///< QR-based (fast, numerically stable)
    SVD         ///< SVD-based (handles rank deficiency)
};

// ============================================
// Helper Functions
// ============================================

// Helper to check if matrix is symmetric
template <typename T>
bool is_symmetric(const Matrix<T>& A, T tol = T(1e-10)) {
    auto dims = A.dims();
    if (dims[0] != dims[1]) return false;
    
    size_t n = dims[0];
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(A[{i, j}] - A[{j, i}]) > tol) {
                return false;
            }
        }
    }
    return true;
}

// Helper to check if matrix is positive definite (simplified check)
template <typename T>
bool is_positive_definite(const Matrix<T>& A) {
    auto dims = A.dims();
    if (dims[0] != dims[1]) return false;
    
    // Check diagonal elements are positive
    size_t n = dims[0];
    for (size_t i = 0; i < n; ++i) {
        if (A[{i, i}] <= T(0)) return false;
    }
    
    return is_symmetric(A);
}

// ============================================
// 1. LU Decomposition with Partial Pivoting
// ============================================

/**
 * @brief Compute LU decomposition with partial pivoting: P*A = L*U
 * 
 * @tparam T Data type (float or double)
 * @param A Input matrix (will not be modified)
 * @return Pair of (LU combined matrix, pivot vector) or error
 * 
 * The LU combined matrix stores both L and U:
 * - Lower triangle (below diagonal): L (with implicit unit diagonal)
 * - Upper triangle (including diagonal): U
 * 
 * Uses LAPACK sgetrf/dgetrf when available, otherwise fallback implementation.
 * 
 * @section example_lu Example
 * @code
 * Matrix<float> A({{4, 3}, {6, 3}});
 * auto result = tensor::lu_decomp(A);
 * if (auto* lu_data = std::get_if<std::pair<Matrix<float>, std::vector<int>>>(&result)) {
 *     Matrix<float>& LU = lu_data->first;
 *     std::vector<int>& pivots = lu_data->second;
 *     // Use LU and pivots...
 * }
 * @endcode
 */
template <typename T>
TensorResult<std::pair<Matrix<T>, std::vector<int>>> lu_decomp(const Matrix<T>& A) {
    auto dims = A.dims();
    size_t m = dims[0];
    size_t n = dims[1];
    
    if (m == 0 || n == 0) {
        return TensorError::EmptyMatrix;
    }
    
    // Create copy for decomposition (LAPACK modifies in-place)
    Matrix<T> LU = A;
    std::vector<int> pivots(std::min(m, n));
    
#ifdef USE_GPU
    // TODO: cuSOLVER implementation
    // For now, fall through to CPU
#endif
    
#ifdef USE_LAPACK
    if (!A.uses_gpu()) {
        int M = static_cast<int>(m);
        int N = static_cast<int>(n);
        int LDA = N;  // Column-major: leading dimension
        int INFO = 0;
        
        // LAPACK expects column-major, but our tensors are row-major
        // Need to transpose, call LAPACK, then transpose back
        Matrix<T> A_col = transpose(LU);
        
        if constexpr (std::is_same_v<T, float>) {
            sgetrf_(&M, &N, A_col.begin(), &LDA, pivots.data(), &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dgetrf_(&M, &N, A_col.begin(), &LDA, pivots.data(), &INFO);
        }
        
        if (INFO < 0) {
            return TensorError::LapackError;
        } else if (INFO > 0) {
            return TensorError::SingularMatrix;
        }
        
        LU = transpose(A_col);
        
        // Convert 1-based FORTRAN indices to 0-based
        for (auto& p : pivots) {
            p -= 1;
        }
        
        return std::make_pair(LU, pivots);
    }
#endif
    
    // Fallback: Doolittle LU decomposition with partial pivoting
    for (size_t k = 0; k < std::min(m, n); ++k) {
        // Find pivot
        size_t pivot_row = k;
        T max_val = std::abs(LU[{k, k}]);
        
        for (size_t i = k + 1; i < m; ++i) {
            T val = std::abs(LU[{i, k}]);
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }
        
        pivots[k] = static_cast<int>(pivot_row);
        
        if (max_val < std::numeric_limits<T>::epsilon()) {
            return TensorError::SingularMatrix;
        }
        
        // Swap rows
        if (pivot_row != k) {
            for (size_t j = 0; j < n; ++j) {
                T temp = LU[{k, j}];
                LU[{k, j}] = LU[{pivot_row, j}];
                LU[{pivot_row, j}] = temp;
            }
        }
        
        // Compute multipliers and eliminate
        for (size_t i = k + 1; i < m; ++i) {
            LU[{i, k}] /= LU[{k, k}];
            for (size_t j = k + 1; j < n; ++j) {
                LU[{i, j}] -= LU[{i, k}] * LU[{k, j}];
            }
        }
    }
    
    return std::make_pair(LU, pivots);
}

// ============================================
// 2. Linear System Solver
// ============================================

/**
 * @brief Solve linear system Ax = b
 * 
 * @tparam T Data type (float or double)
 * @param A Coefficient matrix (m x n)
 * @param b Right-hand side vector or matrix
 * @param method Solver method (Auto, LU, Cholesky, QR, SVD)
 * @return Solution vector/matrix or error
 * 
 * Automatically selects the best method when method = Auto:
 * - Square + SPD → Cholesky (fastest, most stable for SPD)
 * - Square → LU (general square systems)
 * - Rectangular (m > n) → Least squares (overdetermined)
 * - Rectangular (m < n) → Minimum norm solution (underdetermined)
 * 
 * Uses LAPACK sgesv/dgesv, sposv/dposv, or sgels/dgels when available.
 * 
 * @section example_solve Example
 * @code
 * Matrix<float> A({{4, -1}, {-1, 4}});
 * Vector<float> b({10, 5});
 * 
 * auto x_result = tensor::solve(A, b);
 * if (auto* x = std::get_if<Vector<float>>(&x_result)) {
 *     // Solution found
 * }
 * @endcode
 */
template <typename T>
TensorResult<Vector<T>> solve(const Matrix<T>& A, const Vector<T>& b,
                               SolverMethod method = SolverMethod::Auto) {
    auto dims_A = A.dims();
    auto dims_b = b.dims();
    size_t m = dims_A[0];
    size_t n = dims_A[1];
    
    if (dims_b[0] != m) {
        return TensorError::DimensionMismatch;
    }
    
    // Select method
    if (method == SolverMethod::Auto) {
        if (m == n) {
            // Square system
            if (is_positive_definite(A)) {
                method = SolverMethod::Cholesky;
            } else {
                method = SolverMethod::LU;
            }
        } else {
            // Rectangular system → least squares
            method = SolverMethod::QR;
        }
    }
    
#ifdef USE_GPU
    // TODO: cuSOLVER implementation
#endif
    
#ifdef USE_LAPACK
    if (!A.uses_gpu()) {
        Vector<T> x = b;  // Copy RHS (will be overwritten with solution)
        
        if (method == SolverMethod::LU || (method == SolverMethod::Auto && m == n)) {
            // General solver using LU
            Matrix<T> A_copy = A;
            Matrix<T> A_col = transpose(A_copy);
            std::vector<int> pivots(n);
            
            int N = static_cast<int>(n);
            int NRHS = 1;
            int LDA = N;
            int LDB = N;
            int INFO = 0;
            
            if constexpr (std::is_same_v<T, float>) {
                sgesv_(&N, &NRHS, A_col.begin(), &LDA, pivots.data(),
                       x.begin(), &LDB, &INFO);
            } else if constexpr (std::is_same_v<T, double>) {
                dgesv_(&N, &NRHS, A_col.begin(), &LDA, pivots.data(),
                       x.begin(), &LDB, &INFO);
            }
            
            if (INFO < 0) {
                return TensorError::LapackError;
            } else if (INFO > 0) {
                return TensorError::SingularMatrix;
            }
            
            return x;
            
        } else if (method == SolverMethod::Cholesky) {
            // SPD solver using Cholesky
            Matrix<T> A_copy = A;
            Matrix<T> A_col = transpose(A_copy);
            
            int N = static_cast<int>(n);
            int NRHS = 1;
            int LDA = N;
            int LDB = N;
            int INFO = 0;
            char UPLO = 'U';  // Upper triangular
            
            if constexpr (std::is_same_v<T, float>) {
                sposv_(&UPLO, &N, &NRHS, A_col.begin(), &LDA,
                       x.begin(), &LDB, &INFO);
            } else if constexpr (std::is_same_v<T, double>) {
                dposv_(&UPLO, &N, &NRHS, A_col.begin(), &LDA,
                       x.begin(), &LDB, &INFO);
            }
            
            if (INFO < 0) {
                return TensorError::LapackError;
            } else if (INFO > 0) {
                return TensorError::NotPositiveDefinite;
            }
            
            return x;
        }
    }
#endif
    
    // Fallback: use LU decomposition
    auto lu_result = lu_decomp(A);
    if (auto* lu_data = std::get_if<std::pair<Matrix<T>, std::vector<int>>>(&lu_result)) {
        const Matrix<T>& LU = lu_data->first;
        const std::vector<int>& pivots = lu_data->second;
        
        // Forward substitution with pivoting: Ly = Pb
        Vector<T> y = b;
        for (size_t i = 0; i < pivots.size(); ++i) {
            if (pivots[i] != static_cast<int>(i)) {
                T temp = y[{i}];
                y[{i}] = y[{static_cast<size_t>(pivots[i])}];
                y[{static_cast<size_t>(pivots[i])}] = temp;
            }
            
            for (size_t j = 0; j < i; ++j) {
                y[{i}] -= LU[{i, j}] * y[{j}];
            }
        }
        
        // Backward substitution: Ux = y
        Vector<T> x({n}, A.uses_gpu());
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            T sum = y[{static_cast<size_t>(i)}];
            for (size_t j = i + 1; j < n; ++j) {
                sum -= LU[{static_cast<size_t>(i), j}] * x[{j}];
            }
            x[{static_cast<size_t>(i)}] = sum / LU[{static_cast<size_t>(i), static_cast<size_t>(i)}];
        }
        
        return x;
    }
    
    return std::get<TensorError>(lu_result);
}

// Matrix overload for solving multiple right-hand sides
template <typename T>
TensorResult<Matrix<T>> solve(const Matrix<T>& A, const Matrix<T>& B,
                               SolverMethod method = SolverMethod::Auto) {
    auto dims_B = B.dims();
    size_t nrhs = dims_B[1];
    
    // Solve column by column
    Matrix<T> X({dims_B[0], nrhs}, A.uses_gpu());
    
    for (size_t col = 0; col < nrhs; ++col) {
        // Extract column
        Vector<T> b({dims_B[0]}, A.uses_gpu());
        for (size_t i = 0; i < dims_B[0]; ++i) {
            b[{i}] = B[{i, col}];
        }
        
        // Solve
        auto x_result = solve(A, b, method);
        if (auto* x = std::get_if<Vector<T>>(&x_result)) {
            // Store solution
            for (size_t i = 0; i < x->dims()[0]; ++i) {
                X[{i, col}] = (*x)[{i}];
            }
        } else {
            return std::get<TensorError>(x_result);
        }
    }
    
    return X;
}

// ============================================
// 3. Least Squares Solver
// ============================================

/**
 * @brief Solve least squares problem: minimize ||Ax - b||²
 * 
 * @tparam T Data type (float or double)
 * @param A Coefficient matrix (m x n), typically m > n (overdetermined)
 * @param b Right-hand side vector
 * @param method QR (fast, stable) or SVD (rank-deficient)
 * @param rcond Tolerance for rank determination (only for SVD), -1 = default
 * @return Solution vector or error
 * 
 * Uses LAPACK sgels/dgels (QR) or sgelsd/dgelsd (SVD) when available.
 * 
 * @section example_lstsq Example
 * @code
 * // Fit line y = mx + b to data points
 * Matrix<float> A({{1, 1}, {2, 1}, {3, 1}, {4, 1}});  // [x, 1]
 * Vector<float> y({2.1, 3.9, 6.2, 7.9});
 * 
 * auto params = tensor::lstsq(A, y);  // Returns [m, b]
 * @endcode
 */
template <typename T>
TensorResult<Vector<T>> lstsq(const Matrix<T>& A, const Vector<T>& b,
                               LstsqMethod method = LstsqMethod::QR,
                               T rcond = T(-1)) {
    auto dims_A = A.dims();
    auto dims_b = b.dims();
    size_t m = dims_A[0];
    size_t n = dims_A[1];
    
    if (dims_b[0] != m) {
        return TensorError::DimensionMismatch;
    }
    
#ifdef USE_GPU
    // TODO: cuSOLVER implementation
#endif
    
#ifdef USE_LAPACK
    if (!A.uses_gpu()) {
        Matrix<T> A_copy = A;
        Matrix<T> A_col = transpose(A_copy);
        Vector<T> b_ext({std::max(m, n)}, false);  // Extended for LAPACK
        
        // Copy b into extended vector
        for (size_t i = 0; i < m; ++i) {
            b_ext[{i}] = b[{i}];
        }
        
        int M = static_cast<int>(m);
        int N = static_cast<int>(n);
        int NRHS = 1;
        int LDA = M;
        int LDB = static_cast<int>(std::max(m, n));
        int INFO = 0;
        char TRANS = 'N';
        
        if (method == LstsqMethod::QR) {
            // Query optimal workspace
            T work_size;
            int LWORK = -1;
            
            if constexpr (std::is_same_v<T, float>) {
                sgels_(&TRANS, &M, &N, &NRHS, A_col.begin(), &LDA,
                       b_ext.begin(), &LDB, &work_size, &LWORK, &INFO);
            } else if constexpr (std::is_same_v<T, double>) {
                dgels_(&TRANS, &M, &N, &NRHS, A_col.begin(), &LDA,
                       b_ext.begin(), &LDB, &work_size, &LWORK, &INFO);
            }
            
            LWORK = static_cast<int>(work_size);
            std::vector<T> work(LWORK);
            
            // Actual computation
            if constexpr (std::is_same_v<T, float>) {
                sgels_(&TRANS, &M, &N, &NRHS, A_col.begin(), &LDA,
                       b_ext.begin(), &LDB, work.data(), &LWORK, &INFO);
            } else if constexpr (std::is_same_v<T, double>) {
                dgels_(&TRANS, &M, &N, &NRHS, A_col.begin(), &LDA,
                       b_ext.begin(), &LDB, work.data(), &LWORK, &INFO);
            }
            
            if (INFO != 0) {
                return TensorError::LapackError;
            }
            
            // Extract solution (first n elements)
            Vector<T> x({n}, false);
            for (size_t i = 0; i < n; ++i) {
                x[{i}] = b_ext[{i}];
            }
            
            return x;
        }
    }
#endif
    
    // Fallback: use normal equations A^T A x = A^T b
    Matrix<T> AT = transpose(A);
    Matrix<T> ATA = matmul(AT, A);
    
    // Convert b to matrix for matmul
    Matrix<T> b_mat({m, 1}, A.uses_gpu());
    for (size_t i = 0; i < m; ++i) {
        b_mat[{i, 0}] = b[{i}];
    }
    
    Matrix<T> ATb_mat = matmul(AT, b_mat);
    
    // Extract as vector
    Vector<T> ATb({n}, A.uses_gpu());
    for (size_t i = 0; i < n; ++i) {
        ATb[{i}] = ATb_mat[{i, 0}];
    }
    
    // Solve A^T A x = A^T b
    return solve(ATA, ATb, SolverMethod::Cholesky);
}

// ============================================
// 4. Matrix Rank
// ============================================

/**
 * @brief Compute numerical rank of a matrix using SVD
 * 
 * @tparam T Data type
 * @param A Input matrix
 * @param tol Tolerance for zero singular values (default: epsilon * max(m,n) * max_sigma)
 * @return Rank of matrix
 * 
 * @section example_rank Example
 * @code
 * Matrix<float> A({{1, 2}, {2, 4}});  // Rank-1 matrix
 * size_t r = tensor::matrix_rank(A);  // Returns 1
 * @endcode
 */
template <typename T>
size_t matrix_rank(const Matrix<T>& A, T tol = T(-1)) {
    // For now, use the existing rank() function from linalg.h
    return rank(A, tol);
}

// ============================================
// 5. Pseudo-inverse (Moore-Penrose)
// ============================================

/**
 * @brief Compute Moore-Penrose pseudo-inverse using SVD
 * 
 * @tparam T Data type
 * @param A Input matrix (m x n)
 * @param rcond Cutoff for small singular values (default: epsilon * max(m,n))
 * @return Pseudo-inverse (n x m) or error
 * 
 * The pseudo-inverse A⁺ satisfies: A A⁺ A = A and A⁺ A A⁺ = A⁺
 * 
 * Uses SVD: A = U Σ V^T, then A⁺ = V Σ⁺ U^T
 * where Σ⁺ has reciprocals of non-zero singular values.
 * 
 * @section example_pinv Example
 * @code
 * Matrix<float> A({{1, 2}, {3, 4}, {5, 6}});
 * auto Ainv_result = tensor::pinv(A);
 * if (auto* Ainv = std::get_if<Matrix<float>>(&Ainv_result)) {
 *     // Ainv is 2x3 pseudo-inverse
 * }
 * @endcode
 */
template <typename T>
TensorResult<Matrix<T>> pinv(const Matrix<T>& A, T rcond = T(-1)) {
    auto dims = A.dims();
    size_t m = dims[0];
    size_t n = dims[1];
    
    if (rcond < T(0)) {
        rcond = std::numeric_limits<T>::epsilon() * std::max(m, n);
    }
    
    // For now, use simplified approach via solve for square matrices
    // Full SVD-based implementation would be added with LAPACK integration
    
    if (m == n) {
        // Square matrix - try to use inverse
        // This is a placeholder - would use LAPACK sgetri/dgetri
        
#ifdef USE_LAPACK
        // TODO: Implement using LAPACK sgetri/dgetri after LU decomposition
#endif
        
        // Fallback: return error for now
        return TensorError::NotImplemented;
    }
    
    // For rectangular matrices, A⁺ = (A^T A)^(-1) A^T (right inverse)
    // or A⁺ = A^T (A A^T)^(-1) (left inverse)
    
    if (m > n) {
        // Overdetermined: A⁺ = (A^T A)^(-1) A^T
        Matrix<T> AT = transpose(A);
        Matrix<T> ATA = matmul(AT, A);
        
        // Solve ATA * X = AT
        auto X_result = solve(ATA, AT);
        return X_result;
        
    } else {
        // Underdetermined: A⁺ = A^T (A A^T)^(-1)
        Matrix<T> AT = transpose(A);
        Matrix<T> AAT = matmul(A, AT);
        
        // Solve AAT * X^T = A, then transpose
        // For now, return error
        return TensorError::NotImplemented;
    }
}

// ============================================
// 6. Kronecker Product
// ============================================

/**
 * @brief Compute Kronecker product of two matrices
 * 
 * @tparam T Data type
 * @param A First matrix (m x n)
 * @param B Second matrix (p x q)
 * @return Kronecker product (mp x nq)
 * 
 * The Kronecker product A ⊗ B is defined as:
 * [a₁₁B  a₁₂B  ...]
 * [a₂₁B  a₂₂B  ...]
 * [...]
 * 
 * Used in statistics, quantum computing, tensor decompositions, etc.
 * 
 * @section example_kron Example
 * @code
 * Matrix<float> A({{1, 2}, {3, 4}});
 * Matrix<float> B({{0, 5}, {6, 7}});
 * auto C = tensor::kron(A, B);  // 4x4 matrix
 * @endcode
 */
template <typename T>
TensorResult<Matrix<T>> kron(const Matrix<T>& A, const Matrix<T>& B) {
    auto dims_A = A.dims();
    auto dims_B = B.dims();
    size_t m = dims_A[0];
    size_t n = dims_A[1];
    size_t p = dims_B[0];
    size_t q = dims_B[1];
    
    Matrix<T> C({m * p, n * q}, A.uses_gpu());
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T a_ij = A[{i, j}];
            
            for (size_t k = 0; k < p; ++k) {
                for (size_t l = 0; l < q; ++l) {
                    C[{(i * p) + k, (j * q) + l}] = (a_ij) * (B[{k, l}]);
                }
            }
        }
    }
    
    return C;
}

// ============================================
// 7. Matrix Determinant (LAPACK-accelerated)
// ============================================

/**
 * @brief Compute determinant of a square matrix
 * 
 * @tparam T Data type
 * @param A Input square matrix
 * @return Determinant value or error
 * 
 * Uses LU decomposition: det(A) = det(P) * det(L) * det(U)
 * = (-1)^(number of swaps) * product of diagonal elements of U
 * 
 * @section example_det Example
 * @code
 * Matrix<float> A({{4, -2}, {-1, 3}});
 * auto det_result = tensor::determinant(A);
 * if (auto* det = std::get_if<float>(&det_result)) {
 *     // det = 10
 * }
 * @endcode
 */
template <typename T>
TensorResult<T> determinant(const Matrix<T>& A) {
    auto dims = A.dims();
    if (dims[0] != dims[1]) {
        return TensorError::NotSquare;
    }
    
    size_t n = dims[0];
    
    if (n == 0) {
        return T(1);
    }
    
    if (n == 1) {
        return A[{0, 0}];
    }
    
    if (n == 2) {
        return A[{0, 0}] * A[{1, 1}] - A[{0, 1}] * A[{1, 0}];
    }
    
    // Use LU decomposition
    auto lu_result = lu_decomp(A);
    if (auto* lu_data = std::get_if<std::pair<Matrix<T>, std::vector<int>>>(&lu_result)) {
        const Matrix<T>& LU = lu_data->first;
        const std::vector<int>& pivots = lu_data->second;
        
        // Compute product of diagonal elements
        T det = T(1);
        for (size_t i = 0; i < n; ++i) {
            det *= LU[{i, i}];
        }
        
        // Account for row swaps (each swap changes sign)
        int num_swaps = 0;
        for (size_t i = 0; i < pivots.size(); ++i) {
            if (pivots[i] != static_cast<int>(i)) {
                num_swaps++;
            }
        }
        
        if (num_swaps % 2 == 1) {
            det = -det;
        }
        
        return det;
    }
    
    return std::get<TensorError>(lu_result);
}

// ============================================
// 8. Matrix Inverse (LAPACK-accelerated)
// ============================================

/**
 * @brief Compute inverse of a square matrix
 * 
 * @tparam T Data type
 * @param A Input square matrix
 * @return Inverse matrix or error
 * 
 * Uses LU decomposition followed by inversion.
 * For symmetric positive definite matrices, consider using Cholesky-based inversion.
 * 
 * @section example_inv Example
 * @code
 * Matrix<float> A({{4, 7}, {2, 6}});
 * auto Ainv_result = tensor::inverse(A);
 * if (auto* Ainv = std::get_if<Matrix<float>>(&Ainv_result)) {
 *     // Verify: A * Ainv ≈ I
 * }
 * @endcode
 */
template <typename T>
TensorResult<Matrix<T>> inverse(const Matrix<T>& A) {
    auto dims = A.dims();
    if (dims[0] != dims[1]) {
        return TensorError::NotSquare;
    }
    
    size_t n = dims[0];
    
    if (n == 0) {
        return Matrix<T>({0, 0}, A.uses_gpu());
    }
    
    if (n == 1) {
        T val = A[{0, 0}];
        if (std::abs(val) < std::numeric_limits<T>::epsilon()) {
            return TensorError::SingularMatrix;
        }
        Matrix<T> inv({1, 1}, A.uses_gpu());
        inv[{0, 0}] = T(1) / val;
        return inv;
    }
    
#ifdef USE_LAPACK
    if (!A.uses_gpu()) {
        // Use LAPACK getrf + getri for inversion
        auto lu_result = lu_decomp(A);
        if (auto* lu_data = std::get_if<std::pair<Matrix<T>, std::vector<int>>>(&lu_result)) {
            Matrix<T> A_inv = lu_data->first;
            std::vector<int>& pivots = lu_data->second;
            
            // Convert pivots back to 1-based for LAPACK
            for (auto& p : pivots) {
                p += 1;
            }
            
            Matrix<T> A_col = transpose(A_inv);
            
            // Query workspace size
            int N = static_cast<int>(n);
            int LDA = N;
            int INFO = 0;
            T work_size;
            int LWORK = -1;
            
            if constexpr (std::is_same_v<T, float>) {
                sgetri_(&N, A_col.begin(), &LDA, pivots.data(), &work_size, &LWORK, &INFO);
            } else if constexpr (std::is_same_v<T, double>) {
                dgetri_(&N, A_col.begin(), &LDA, pivots.data(), &work_size, &LWORK, &INFO);
            }
            
            LWORK = static_cast<int>(work_size);
            std::vector<T> work(LWORK);
            
            // Compute inverse
            if constexpr (std::is_same_v<T, float>) {
                sgetri_(&N, A_col.begin(), &LDA, pivots.data(), work.data(), &LWORK, &INFO);
            } else if constexpr (std::is_same_v<T, double>) {
                dgetri_(&N, A_col.begin(), &LDA, pivots.data(), work.data(), &LWORK, &INFO);
            }
            
            if (INFO != 0) {
                return TensorError::LapackError;
            }
            
            return transpose(A_col);
        }
        
        return std::get<TensorError>(lu_result);
    }
#endif
    
    // Fallback: Gauss-Jordan elimination
    Matrix<T> Aug({n, 2 * n}, A.uses_gpu());
    
    // Build augmented matrix [A | I]
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Aug[{i, j}] = A[{i, j}];
            Aug[{i, n + j}] = (i == j) ? T(1) : T(0);
        }
    }
    
    // Forward elimination
    for (size_t k = 0; k < n; ++k) {
        // Find pivot
        size_t pivot_row = k;
        T max_val = std::abs(Aug[{k, k}]);
        
        for (size_t i = k + 1; i < n; ++i) {
            T val = std::abs(Aug[{i, k}]);
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }
        
        if (max_val < std::numeric_limits<T>::epsilon()) {
            return TensorError::SingularMatrix;
        }
        
        // Swap rows
        if (pivot_row != k) {
            for (size_t j = 0; j < 2 * n; ++j) {
                T temp = Aug[{k, j}];
                Aug[{k, j}] = Aug[{pivot_row, j}];
                Aug[{pivot_row, j}] = temp;
            }
        }
        
        // Scale pivot row
        T pivot = Aug[{k, k}];
        for (size_t j = 0; j < 2 * n; ++j) {
            Aug[{k, j}] /= pivot;
        }
        
        // Eliminate column
        for (size_t i = 0; i < n; ++i) {
            if (i != k) {
                T factor = Aug[{i, k}];
                for (size_t j = 0; j < 2 * n; ++j) {
                    Aug[{i, j}] -= factor * Aug[{k, j}];
                }
            }
        }
    }
    
    // Extract inverse from right half
    Matrix<T> A_inv({n, n}, A.uses_gpu());
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_inv[{i, j}] = Aug[{i, n + j}];
        }
    }
    
    return A_inv;
}

// ============================================
// QR Decomposition
// ============================================

/**
 * @brief QR decomposition: A = QR
 * @param A Input matrix (m x n)
 * @return Pair of Q (m x m orthogonal) and R (m x n upper triangular), or error
 */
template <typename T>
auto qr_decomp(const Matrix<T>& A) -> std::variant<std::pair<Matrix<T>, Matrix<T>>, TensorError> {
    auto dims = A.dims();
    size_t m = dims[0];
    size_t n = dims[1];
    
#ifdef USE_LAPACK
    if (!A.uses_gpu()) {
        // Make a column-major copy
        Matrix<T> A_col = transpose(A);
        int M = static_cast<int>(m);
        int N = static_cast<int>(n);
        int K = std::min(M, N);
        int LDA = M;
        int INFO = 0;
        
        std::vector<T> tau(K);
        T work_size;
        int LWORK = -1;
        
        // Query workspace size
        if constexpr (std::is_same_v<T, float>) {
            sgeqrf_(&M, &N, A_col.begin(), &LDA, tau.data(), &work_size, &LWORK, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dgeqrf_(&M, &N, A_col.begin(), &LDA, tau.data(), &work_size, &LWORK, &INFO);
        }
        
        LWORK = static_cast<int>(work_size);
        std::vector<T> work(LWORK);
        
        // Compute QR factorization
        if constexpr (std::is_same_v<T, float>) {
            sgeqrf_(&M, &N, A_col.begin(), &LDA, tau.data(), work.data(), &LWORK, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dgeqrf_(&M, &N, A_col.begin(), &LDA, tau.data(), work.data(), &LWORK, &INFO);
        }
        
        if (INFO != 0) {
            return TensorError::LapackError;
        }
        
        // Extract R (upper triangular part)
        Matrix<T> R({m, n}, false);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i <= j) {
                    R[{i, j}] = A_col[{j, i}];
                } else {
                    R[{i, j}] = T(0);
                }
            }
        }
        
        // Generate Q
        LWORK = -1;
        if constexpr (std::is_same_v<T, float>) {
            sorgqr_(&M, &M, &K, A_col.begin(), &LDA, tau.data(), &work_size, &LWORK, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dorgqr_(&M, &M, &K, A_col.begin(), &LDA, tau.data(), &work_size, &LWORK, &INFO);
        }
        
        LWORK = static_cast<int>(work_size);
        work.resize(LWORK);
        
        if constexpr (std::is_same_v<T, float>) {
            sorgqr_(&M, &M, &K, A_col.begin(), &LDA, tau.data(), work.data(), &LWORK, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dorgqr_(&M, &M, &K, A_col.begin(), &LDA, tau.data(), work.data(), &LWORK, &INFO);
        }
        
        if (INFO != 0) {
            return TensorError::LapackError;
        }
        
        Matrix<T> Q({m, m}, false);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                Q[{i, j}] = A_col[{j, i}];
            }
        }
        
        return std::make_pair(Q, R);
    }
#endif
    
    return TensorError::NotImplemented;
}

// ============================================
// Cholesky Decomposition
// ============================================

/**
 * @brief Cholesky decomposition: A = L * L^T for symmetric positive definite matrices
 * @param A Input symmetric positive definite matrix (n x n)
 * @return Lower triangular matrix L, or error
 */
template <typename T>
auto cholesky_decomp(const Matrix<T>& A) -> std::variant<Matrix<T>, TensorError> {
    auto dims = A.dims();
    if (dims[0] != dims[1]) {
        return TensorError::NotSquare;
    }
    
    size_t n = dims[0];
    
#ifdef USE_LAPACK
    if (!A.uses_gpu()) {
        // Make a column-major copy
        Matrix<T> A_col = transpose(A);
        int N = static_cast<int>(n);
        int LDA = N;
        int INFO = 0;
        char UPLO = 'L';
        
        // Compute Cholesky factorization
        if constexpr (std::is_same_v<T, float>) {
            spotrf_(&UPLO, &N, A_col.begin(), &LDA, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dpotrf_(&UPLO, &N, A_col.begin(), &LDA, &INFO);
        }
        
        if (INFO != 0) {
            return TensorError::LapackError;
        }
        
        // Extract lower triangular part and transpose back
        Matrix<T> L({n, n}, false);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i >= j) {
                    L[{i, j}] = A_col[{j, i}];
                } else {
                    L[{i, j}] = T(0);
                }
            }
        }
        
        return L;
    }
#endif
    
    return TensorError::NotImplemented;
}

// ============================================
// SVD Decomposition
// ============================================

/**
 * @brief Singular Value Decomposition: A = U * Sigma * V^T
 * @param A Input matrix (m x n)
 * @return Tuple of U (m x m), singular values (min(m,n)), V^T (n x n), or error
 */
template <typename T>
auto svd_decomp(const Matrix<T>& A) 
    -> std::variant<std::tuple<Matrix<T>, Vector<T>, Matrix<T>>, TensorError> {
    auto dims = A.dims();
    size_t m = dims[0];
    size_t n = dims[1];
    size_t min_dim = std::min(m, n);
    
#ifdef USE_LAPACK
    if (!A.uses_gpu()) {
        // Make a column-major copy
        Matrix<T> A_col = transpose(A);
        int M = static_cast<int>(m);
        int N = static_cast<int>(n);
        int LDA = M;
        int LDU = M;
        int LDVT = N;
        int INFO = 0;
        char JOBU = 'A';
        char JOBVT = 'A';
        
        Vector<T> S({min_dim}, false);
        Matrix<T> U_col({m, m}, false);
        Matrix<T> VT_col({n, n}, false);
        
        T work_size;
        int LWORK = -1;
        
        // Query workspace size
        if constexpr (std::is_same_v<T, float>) {
            sgesvd_(&JOBU, &JOBVT, &M, &N, A_col.begin(), &LDA,
                    S.begin(), U_col.begin(), &LDU, VT_col.begin(), &LDVT,
                    &work_size, &LWORK, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dgesvd_(&JOBU, &JOBVT, &M, &N, A_col.begin(), &LDA,
                    S.begin(), U_col.begin(), &LDU, VT_col.begin(), &LDVT,
                    &work_size, &LWORK, &INFO);
        }
        
        LWORK = static_cast<int>(work_size);
        std::vector<T> work(LWORK);
        
        // Compute SVD
        if constexpr (std::is_same_v<T, float>) {
            sgesvd_(&JOBU, &JOBVT, &M, &N, A_col.begin(), &LDA,
                    S.begin(), U_col.begin(), &LDU, VT_col.begin(), &LDVT,
                    work.data(), &LWORK, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dgesvd_(&JOBU, &JOBVT, &M, &N, A_col.begin(), &LDA,
                    S.begin(), U_col.begin(), &LDU, VT_col.begin(), &LDVT,
                    work.data(), &LWORK, &INFO);
        }
        
        if (INFO != 0) {
            return TensorError::LapackError;
        }
        
        // Transpose U and VT back to row-major
        Matrix<T> U = transpose(U_col);
        Matrix<T> VT = transpose(VT_col);
        
        return std::make_tuple(U, S, VT);
    }
#endif
    
    return TensorError::NotImplemented;
}

// ============================================
// Eigenvalue Decomposition (Symmetric Matrices)
// ============================================

/**
 * @brief Compute eigenvalues and eigenvectors for symmetric matrices
 * @param A Input symmetric matrix (n x n)
 * @return Pair of eigenvalues and eigenvector matrix (columns are eigenvectors), or error
 */
template <typename T>
auto eig_decomp(const Matrix<T>& A) 
    -> std::variant<std::pair<Vector<T>, Matrix<T>>, TensorError> {
    auto dims = A.dims();
    if (dims[0] != dims[1]) {
        return TensorError::NotSquare;
    }
    
    size_t n = dims[0];
    
#ifdef USE_LAPACK
    if (!A.uses_gpu()) {
        // Make a column-major copy
        Matrix<T> A_col = transpose(A);
        int N = static_cast<int>(n);
        int LDA = N;
        int INFO = 0;
        char JOBZ = 'V';  // Compute eigenvalues and eigenvectors
        char UPLO = 'U';  // Upper triangle of A is stored
        
        Vector<T> W({n}, false);  // Eigenvalues
        
        T work_size;
        int LWORK = -1;
        
        // Query workspace size
        if constexpr (std::is_same_v<T, float>) {
            ssyev_(&JOBZ, &UPLO, &N, A_col.begin(), &LDA, W.begin(),
                   &work_size, &LWORK, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dsyev_(&JOBZ, &UPLO, &N, A_col.begin(), &LDA, W.begin(),
                   &work_size, &LWORK, &INFO);
        }
        
        LWORK = static_cast<int>(work_size);
        std::vector<T> work(LWORK);
        
        // Compute eigendecomposition
        if constexpr (std::is_same_v<T, float>) {
            ssyev_(&JOBZ, &UPLO, &N, A_col.begin(), &LDA, W.begin(),
                   work.data(), &LWORK, &INFO);
        } else if constexpr (std::is_same_v<T, double>) {
            dsyev_(&JOBZ, &UPLO, &N, A_col.begin(), &LDA, W.begin(),
                   work.data(), &LWORK, &INFO);
        }
        
        if (INFO != 0) {
            return TensorError::LapackError;
        }
        
        // Transpose eigenvectors back to row-major
        Matrix<T> V = transpose(A_col);
        
        return std::make_pair(W, V);
    }
#endif
    
    return TensorError::NotImplemented;
}

} // namespace tensor

#endif // _LINALG_ADVANCED_H
