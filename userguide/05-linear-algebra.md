# Linear Algebra

## Matrix Multiplication

```cpp
#include "linalg.h"

// Basic matrix multiplication
auto A = Matrix<float>::randn({3, 4});
auto B = Matrix<float>::randn({4, 5});
auto C_var = matmul(A, B);  // Result: 3x5
auto C = std::get<Matrix<float>>(C_var);

// Vector dot product
auto v1 = Vector<float>::randn({10});
auto v2 = Vector<float>::randn({10});
auto dot_var = dot(v1, v2);
float dot_product = std::get<float>(dot_var);

// Cross product (3D vectors only)
auto a = Vector<float>::from_array({1.0f, 0.0f, 0.0f}, {3});
auto b = Vector<float>::from_array({0.0f, 1.0f, 0.0f}, {3});
auto c_var = cross(a, b);  // Result: [0, 0, 1]
auto c = std::get<Vector<float>>(c_var);
```

## Matrix Decompositions

### Singular Value Decomposition (SVD)

```cpp
auto A = Matrix<float>::randn({5, 3});
auto svd_var = svd(A);
auto [U, S, Vt] = std::get<std::tuple<Matrix<float>, Vector<float>, Matrix<float>>>(svd_var);
// A = U * diag(S) * Vt
// U: 5x3 (left singular vectors)
// S: 3 (singular values)
// Vt: 3x3 (right singular vectors, transposed)
```

### QR Decomposition

```cpp
auto A = Matrix<float>::randn({4, 3});
auto qr_var = qr(A);
auto [Q, R] = std::get<std::pair<Matrix<float>, Matrix<float>>>(qr_var);
// A = Q * R
// Q: 4x3 (orthogonal)
// R: 3x3 (upper triangular)
```

### Cholesky Decomposition

```cpp
// For symmetric positive definite matrices
auto A = Matrix<float>::eye({4, 4}) + Matrix<float>::ones({4, 4});  // Make SPD
auto L_var = cholesky(A);
auto L = std::get<Matrix<float>>(L_var);
// A = L * L^T (L is lower triangular)
```

### LU Decomposition

```cpp
auto A = Matrix<float>::randn({4, 4});
auto lu_var = lu(A);
auto [L, U, P] = std::get<std::tuple<Matrix<float>, Matrix<float>, std::vector<int>>>(lu_var);
// PA = LU
// L: lower triangular with unit diagonal
// U: upper triangular
// P: permutation indices
```

### Eigenvalue Decomposition

```cpp
auto A = Matrix<float>::randn({4, 4});
auto eig_var = eigenvalues(A);
auto eigenvals = std::get<Vector<float>>(eig_var);

auto eigvec_var = eigenvectors(A);
auto eigenvecs = std::get<Matrix<float>>(eigvec_var);
```

## Matrix Operations

### Inverse

```cpp
auto A = Matrix<float>::randn({4, 4});
auto A_inv_var = inverse(A);
auto A_inv = std::get<Matrix<float>>(A_inv_var);

// Verify: A * A_inv ≈ I
auto I_var = matmul(A, A_inv);
```

### Determinant

```cpp
auto A = Matrix<float>::randn({3, 3});
auto det_var = determinant(A);
float det = std::get<float>(det_var);
```

### Matrix Rank

```cpp
auto A = Matrix<float>::randn({5, 3});
auto rank_var = matrix_rank(A);
int rank = std::get<int>(rank_var);
```

### Pseudo-Inverse

```cpp
// Moore-Penrose pseudo-inverse (uses SVD)
auto A = Matrix<float>::randn({5, 3});
auto A_pinv_var = pinverse(A);
auto A_pinv = std::get<Matrix<float>>(A_pinv_var);
```

### Transpose

```cpp
auto A = Matrix<float>::randn({3, 4});
auto At = A.transpose();  // 4x3
```

### Kronecker Product

```cpp
auto A = Matrix<float>::randn({2, 3});
auto B = Matrix<float>::randn({4, 5});
auto C_var = kronecker_product(A, B);
auto C = std::get<Matrix<float>>(C_var);  // 8x15
```

## Specialized Types

### Matrix Type

```cpp
// Matrix<T> is alias for Tensor<T, 2>
Matrix<float> m = Matrix<float>::zeros({10, 20});

// Matrix-specific operations
auto row = m.row(0);      // Get first row
auto col = m.col(5);      // Get 6th column
auto diag = m.diag();     // Get diagonal

// Submatrices
auto block = m.block(2, 3, 4, 5);  // 4x5 block starting at (2,3)
auto top_rows = m.topRows(5);
auto bottom_rows = m.bottomRows(3);
auto left_cols = m.leftCols(10);
auto right_cols = m.rightCols(8);
```

### Vector Type

```cpp
// Vector<T> is alias for Tensor<T, 1>
Vector<float> v = Vector<float>::zeros({100});

// Vector-specific operations
auto head = v.head(10);   // First 10 elements
auto tail = v.tail(10);   // Last 10 elements

// Norm operations
float l1_norm = v.norm(1);
float l2_norm = v.norm(2);  // Euclidean norm
float inf_norm = v.norm(INFINITY);
```

## Linear System Solvers

### Direct Solvers

```cpp
// Solve Ax = b using different methods
auto A = Matrix<float>::randn({100, 100});
auto b = Vector<float>::randn({100});

// LU decomposition (general matrices)
auto x_lu_var = solve_lu(A, b);
auto x_lu = std::get<Vector<float>>(x_lu_var);

// QR decomposition (more stable for ill-conditioned systems)
auto x_qr_var = solve_qr(A, b);
auto x_qr = std::get<Vector<float>>(x_qr_var);

// Cholesky (for symmetric positive definite matrices - faster)
auto A_spd = Matrix<float>::eye({100, 100}) + Matrix<float>::ones({100, 100}) * 0.1f;
auto x_chol_var = solve_cholesky(A_spd, b);
auto x_chol = std::get<Vector<float>>(x_chol_var);
```

### Least Squares

```cpp
// Solve overdetermined systems (more equations than unknowns)
auto A = Matrix<float>::randn({200, 50});  // 200 equations, 50 unknowns
auto b = Vector<float>::randn({200});

// QR-based least squares
auto x_qr_var = lstsq_qr(A, b);
auto x_qr = std::get<Vector<float>>(x_qr_var);

// SVD-based least squares (more robust, handles rank-deficient matrices)
auto x_svd_var = lstsq_svd(A, b);
auto x_svd = std::get<Vector<float>>(x_svd_var);

// Verify solution minimizes ||Ax - b||²
auto residual_var = matmul(A, x_svd.unsqueeze(1)) - b.unsqueeze(1);
auto residual = std::get<Matrix<float>>(residual_var);
float error = residual.norm(2);
```

## Practical Examples

### Linear System Solving

```cpp
// Example: Solve a system of linear equations
auto A = Matrix<float>::from_array({
    1.0f, 2.0f, 3.0f,
    2.0f, 5.0f, 3.0f,
    1.0f, 0.0f, 8.0f
}, {3, 3});

auto b = Vector<float>::from_array({14.0f, 18.0f, 20.0f}, {3});

auto x_var = solve_lu(A, b);
auto x = std::get<Vector<float>>(x_var);
// Solution: x ≈ [1, 2, 3]
```

### PCA (Principal Component Analysis)

```cpp
auto X = Matrix<float>::randn({1000, 50});  // 1000 samples, 50 features

// Center the data
auto mean_var = X.mean(0);
auto mean = std::get<Vector<float>>(mean_var);
auto X_centered = X - mean.unsqueeze(0);

// Compute covariance matrix
auto cov_var = covariance_matrix(X_centered);
auto cov = std::get<Matrix<float>>(cov_var);

// Get eigenvalues/eigenvectors
auto eigvals_var = eigenvalues(cov);
auto eigvecs_var = eigenvectors(cov);
auto eigvals = std::get<Vector<float>>(eigvals_var);
auto eigvecs = std::get<Matrix<float>>(eigvecs_var);

// Project to k principal components
int k = 10;
auto proj_matrix = eigvecs.leftCols(k);
auto X_reduced_var = matmul(X_centered, proj_matrix);
auto X_reduced = std::get<Matrix<float>>(X_reduced_var);
```

### Ridge Regression (Regularized Least Squares)

```cpp
// Solve (A^T A + λI)x = A^T b
auto A = Matrix<float>::randn({100, 20});
auto b = Vector<float>::randn({100});
float lambda = 0.1f;

// Form normal equations
auto At_var = A.transpose();
auto At = std::get<Matrix<float>>(At_var);
auto AtA_var = matmul(At, A);
auto AtA = std::get<Matrix<float>>(AtA_var);
auto Atb_var = matmul(At, b.unsqueeze(1));
auto Atb = std::get<Vector<float>>(Atb_var).squeeze();

// Add regularization
auto I = Matrix<float>::eye({20, 20});
auto AtA_reg = AtA + I * lambda;

// Solve using Cholesky (AtA_reg is SPD)
auto x_var = solve_cholesky(AtA_reg, Atb);
auto x = std::get<Vector<float>>(x_var);
```

---

**Previous**: [← Mathematical Operations](04-mathematical-operations.md) | **Next**: [Autograd →](06-autograd.md)
