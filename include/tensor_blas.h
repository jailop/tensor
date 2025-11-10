#ifndef TENSOR_BLAS_H
#define TENSOR_BLAS_H

#ifdef USE_BLAS
extern "C" {
    /// @brief BLAS function declarations for optimized matrix operations
    void cblas_sgemm(const int Order, const int TransA, const int TransB,
                     const int M, const int N, const int K,
                     const float alpha, const float *A, const int lda,
                     const float *B, const int ldb,
                     const float beta, float *C, const int ldc);
    void cblas_dgemm(const int Order, const int TransA, const int TransB,
                     const int M, const int N, const int K,
                     const double alpha, const double *A, const int lda,
                     const double *B, const int ldb,
                     const double beta, double *C, const int ldc);
    float cblas_sdot(const int N, const float *X, const int incX,
                     const float *Y, const int incY);
    double cblas_ddot(const int N, const double *X, const int incX,
                      const double *Y, const int incY);
    void cblas_sscal(const int N, const float alpha, float *X, const int incX);
    void cblas_dscal(const int N, const double alpha, double *X, const int incX);
}

// BLAS constants
constexpr int CblasRowMajor = 101;
constexpr int CblasNoTrans = 111;

// BLAS helper functions to select the correct function based on type
template<typename T>
inline T blas_dot(const int N, const T *X, const int incX, const T *Y, const int incY) {
    // Fallback for unsupported types
    T result = T();
    for (int i = 0; i < N; ++i) {
        result += X[i * incX] * Y[i * incY];
    }
    return result;
}

template<>
inline float blas_dot<float>(const int N, const float *X, const int incX, const float *Y, const int incY) {
    return cblas_sdot(N, X, incX, Y, incY);
}

template<>
inline double blas_dot<double>(const int N, const double *X, const int incX, const double *Y, const int incY) {
    return cblas_ddot(N, X, incX, Y, incY);
}

template<typename T>
inline void blas_scal(const int N, const T alpha, T *X, const int incX) {
    // Fallback for unsupported types
    for (int i = 0; i < N; ++i) {
        X[i * incX] *= alpha;
    }
}

template<>
inline void blas_scal<float>(const int N, const float alpha, float *X, const int incX) {
    cblas_sscal(N, alpha, X, incX);
}

template<>
inline void blas_scal<double>(const int N, const double alpha, double *X, const int incX) {
    cblas_dscal(N, alpha, X, incX);
}

template<typename T>
inline void blas_gemm(const int Order, const int TransA, const int TransB,
                      const int M, const int N, const int K,
                      const T alpha, const T *A, const int lda,
                      const T *B, const int ldb,
                      const T beta, T *C, const int ldc) {
    // Fallback for unsupported types - standard matrix multiplication
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = T();
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
        }
    }
}

template<>
inline void blas_gemm<float>(const int Order, const int TransA, const int TransB,
                             const int M, const int N, const int K,
                             const float alpha, const float *A, const int lda,
                             const float *B, const int ldb,
                             const float beta, float *C, const int ldc) {
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
inline void blas_gemm<double>(const int Order, const int TransA, const int TransB,
                              const int M, const int N, const int K,
                              const double alpha, const double *A, const int lda,
                              const double *B, const int ldb,
                              const double beta, double *C, const int ldc) {
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

#endif // TENSOR_BLAS_H
