/**
 * Example C program demonstrating the Tensor C interface
 */

#include "tensor_c.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=== Tensor C Interface Example ===\n\n");
    
    // Example 1: Vector operations
    printf("1. Vector Operations:\n");
    VectorFloatHandle vec1, vec2, vec_result;
    float vec1_data[] = {1.0f, 2.0f, 3.0f};
    float vec2_data[] = {4.0f, 5.0f, 6.0f};
    
    TensorErrorCode err = vector_float_create(3, vec1_data, &vec1);
    if (err != TENSOR_SUCCESS) {
        fprintf(stderr, "Error creating vec1: %s\n", tensor_c_last_error());
        return 1;
    }
    
    vector_float_create(3, vec2_data, &vec2);
    
    // Vector addition
    vector_float_add(vec1, vec2, &vec_result);
    printf("   vec1 + vec2 = [");
    for (size_t i = 0; i < 3; i++) {
        float val;
        vector_float_get(vec_result, i, &val);
        printf("%s%.1f", i > 0 ? ", " : "", val);
    }
    printf("]\n");
    vector_float_destroy(vec_result);
    
    // Dot product
    float dot;
    vector_float_dot(vec1, vec2, &dot);
    printf("   vec1 · vec2 = %.1f\n\n", dot);
    
    vector_float_destroy(vec1);
    vector_float_destroy(vec2);
    
    // Example 2: Matrix operations
    printf("2. Matrix Operations:\n");
    MatrixFloatHandle mat1, mat2, mat_result;
    float mat1_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float mat2_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    
    matrix_float_create(2, 2, mat1_data, &mat1);
    matrix_float_create(2, 2, mat2_data, &mat2);
    
    printf("   Matrix A:\n");
    for (size_t i = 0; i < 2; i++) {
        printf("   [");
        for (size_t j = 0; j < 2; j++) {
            float val;
            matrix_float_get(mat1, i, j, &val);
            printf("%s%.1f", j > 0 ? ", " : "", val);
        }
        printf("]\n");
    }
    
    // Matrix multiplication
    matrix_float_matmul(mat1, mat2, &mat_result);
    printf("   A × B =\n");
    for (size_t i = 0; i < 2; i++) {
        printf("   [");
        for (size_t j = 0; j < 2; j++) {
            float val;
            matrix_float_get(mat_result, i, j, &val);
            printf("%s%.1f", j > 0 ? ", " : "", val);
        }
        printf("]\n");
    }
    matrix_float_destroy(mat_result);
    
    matrix_float_destroy(mat1);
    matrix_float_destroy(mat2);
    
    // Example 3: Identity matrix
    printf("\n3. Identity Matrix (3×3):\n");
    MatrixFloatHandle eye;
    matrix_float_eye(3, &eye);
    
    for (size_t i = 0; i < 3; i++) {
        printf("   [");
        for (size_t j = 0; j < 3; j++) {
            float val;
            matrix_float_get(eye, i, j, &val);
            printf("%s%.0f", j > 0 ? ", " : "", val);
        }
        printf("]\n");
    }
    matrix_float_destroy(eye);
    
    // Example 4: Statistical operations
    printf("\n4. Statistical Operations:\n");
    VectorFloatHandle stats_vec;
    float stats_data[] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    vector_float_create(5, stats_data, &stats_vec);
    
    float mean, variance, std, min, max;
    vector_float_mean(stats_vec, &mean);
    vector_float_variance(stats_vec, &variance);
    vector_float_std(stats_vec, &std);
    vector_float_min(stats_vec, &min);
    vector_float_max(stats_vec, &max);
    
    printf("   Data: [2.0, 4.0, 6.0, 8.0, 10.0]\n");
    printf("   Mean: %.1f\n", mean);
    printf("   Variance: %.1f\n", variance);
    printf("   Std Dev: %.2f\n", std);
    printf("   Min: %.1f, Max: %.1f\n", min, max);
    
    vector_float_destroy(stats_vec);
    
    // Example 5: Matrix-Vector multiplication
    printf("\n5. Matrix-Vector Product:\n");
    MatrixFloatHandle A;
    VectorFloatHandle x, b;
    float A_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float x_data[] = {1.0f, 2.0f, 3.0f};
    
    matrix_float_create(2, 3, A_data, &A);
    vector_float_create(3, x_data, &x);
    
    printf("   A × x = b where\n");
    printf("   A is 2×3 and x is [1.0, 2.0, 3.0]\n");
    
    matrix_float_matvec(A, x, &b);
    printf("   Result b = [");
    for (size_t i = 0; i < 2; i++) {
        float val;
        vector_float_get(b, i, &val);
        printf("%s%.1f", i > 0 ? ", " : "", val);
    }
    printf("]\n");
    
    matrix_float_destroy(A);
    vector_float_destroy(x);
    vector_float_destroy(b);
    
    // Example 6: Linear Algebra - Matrix Inverse and Determinant
    printf("\n6. Linear Algebra Operations:\n");
    MatrixFloatHandle mat_la, mat_inv;
    float la_data[] = {4.0f, 7.0f, 2.0f, 6.0f};
    matrix_float_create(2, 2, la_data, &mat_la);
    
    float det;
    matrix_float_determinant(mat_la, &det);
    printf("   Matrix determinant: %.1f\n", det);
    
    matrix_float_inverse(mat_la, &mat_inv);
    printf("   Matrix inverse computed\n");
    
    matrix_float_destroy(mat_la);
    matrix_float_destroy(mat_inv);
    
    // Example 7: Eigenvalues (for symmetric matrices)
    printf("\n7. Eigenvalue Decomposition:\n");
    MatrixFloatHandle sym_mat;
    float sym_data[] = {4.0f, 2.0f, 2.0f, 3.0f};
    matrix_float_create(2, 2, sym_data, &sym_mat);
    
    VectorFloatHandle eigenvalues;
    MatrixFloatHandle eigenvectors;
    err = matrix_float_eig(sym_mat, &eigenvalues, &eigenvectors);
    
    if (err == TENSOR_SUCCESS) {
        printf("   Eigenvalue decomposition completed\n");
        size_t eig_size;
        vector_float_size(eigenvalues, &eig_size);
        printf("   Number of eigenvalues: %zu\n", eig_size);
        
        vector_float_destroy(eigenvalues);
        matrix_float_destroy(eigenvectors);
    } else {
        printf("   Eigenvalue decomposition failed (may not be implemented)\n");
    }
    
    matrix_float_destroy(sym_mat);
    
    // Example 8: SVD (Singular Value Decomposition)
    printf("\n8. Singular Value Decomposition:\n");
    MatrixFloatHandle svd_mat, U, Vt;
    VectorFloatHandle S;
    float svd_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    matrix_float_create(2, 3, svd_data, &svd_mat);
    
    err = matrix_float_svd(svd_mat, &U, &S, &Vt);
    if (err == TENSOR_SUCCESS) {
        printf("   SVD decomposition completed\n");
        
        size_t s_size;
        vector_float_size(S, &s_size);
        printf("   Number of singular values: %zu\n", s_size);
        printf("   Singular values: [");
        for (size_t i = 0; i < s_size; i++) {
            float val;
            vector_float_get(S, i, &val);
            printf("%s%.3f", i > 0 ? ", " : "", val);
        }
        printf("]\n");
        
        matrix_float_destroy(U);
        vector_float_destroy(S);
        matrix_float_destroy(Vt);
    } else {
        printf("   SVD failed (this may not be implemented for all backends)\n");
    }
    
    matrix_float_destroy(svd_mat);
    
    // Example 9: Correlation
    printf("\n9. Statistical Correlation:\n");
    VectorFloatHandle vec_x, vec_y;
    float corr_x_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float corr_y_data[] = {2.0f, 4.0f, 5.0f, 4.0f, 5.0f};
    
    vector_float_create(5, corr_x_data, &vec_x);
    vector_float_create(5, corr_y_data, &vec_y);
    
    float corr, cov;
    vector_float_correlation(vec_x, vec_y, &corr);
    vector_float_covariance(vec_x, vec_y, &cov);
    
    printf("   Correlation: %.3f\n", corr);
    printf("   Covariance: %.3f\n", cov);
    
    vector_float_destroy(vec_x);
    vector_float_destroy(vec_y);
    
    // Example 10: Vector norm calculation
    printf("\n10. Vector Norms:\n");
    VectorFloatHandle norm_vec;
    float norm_data[] = {3.0f, 4.0f};
    vector_float_create(2, norm_data, &norm_vec);
    
    float l2_norm;
    vector_float_norm(norm_vec, &l2_norm);
    
    printf("   Vector: [3.0, 4.0]\n");
    printf("   L2 norm: %.1f\n", l2_norm);
    
    vector_float_destroy(norm_vec);
    
    // Example 11: Linear system solving (Ax = b)
    printf("\n11. Solve Linear System Ax = b:\n");
    MatrixFloatHandle coeff_mat;
    VectorFloatHandle rhs_vec, solution;
    float coeff_data[] = {3.0f, 2.0f, 1.0f, 5.0f};
    float rhs_data[] = {5.0f, 11.0f};
    
    matrix_float_create(2, 2, coeff_data, &coeff_mat);
    vector_float_create(2, rhs_data, &rhs_vec);
    
    matrix_float_solve(coeff_mat, rhs_vec, &solution);
    printf("   Solution x: [");
    for (size_t i = 0; i < 2; i++) {
        float val;
        vector_float_get(solution, i, &val);
        printf("%s%.1f", i > 0 ? ", " : "", val);
    }
    printf("]\n");
    
    matrix_float_destroy(coeff_mat);
    vector_float_destroy(rhs_vec);
    vector_float_destroy(solution);
    
    // Example 12: QR Decomposition
    printf("\n12. QR Decomposition:\n");
    MatrixFloatHandle qr_mat, Q, R;
    float qr_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    matrix_float_create(2, 2, qr_data, &qr_mat);
    
    err = matrix_float_qr(qr_mat, &Q, &R);
    if (err == TENSOR_SUCCESS) {
        printf("   QR decomposition completed\n");
        printf("   Q is orthogonal, R is upper triangular\n");
        
        matrix_float_destroy(Q);
        matrix_float_destroy(R);
    } else {
        printf("   QR decomposition failed (may not be available)\n");
    }
    
    matrix_float_destroy(qr_mat);
    
    // Example 13: Cholesky Decomposition
    printf("\n13. Cholesky Decomposition (SPD matrix):\n");
    MatrixFloatHandle chol_mat, chol_L;
    float chol_data[] = {4.0f, 2.0f, 2.0f, 3.0f};
    matrix_float_create(2, 2, chol_data, &chol_mat);
    
    err = matrix_float_cholesky(chol_mat, &chol_L);
    if (err == TENSOR_SUCCESS) {
        printf("   Cholesky decomposition completed\n");
        printf("   L matrix (lower triangular):\n");
        for (size_t i = 0; i < 2; i++) {
            printf("   [");
            for (size_t j = 0; j < 2; j++) {
                float val;
                matrix_float_get(chol_L, i, j, &val);
                printf("%s%.3f", j > 0 ? ", " : "", val);
            }
            printf("]\n");
        }
        
        matrix_float_destroy(chol_L);
    } else {
        printf("   Cholesky decomposition failed (matrix may not be SPD)\n");
    }
    
    matrix_float_destroy(chol_mat);
    
    // Example 14: Matrix Rank
    printf("\n14. Matrix Rank Computation:\n");
    MatrixFloatHandle rank_mat;
    float rank_data[] = {1.0f, 2.0f, 2.0f, 4.0f};
    matrix_float_create(2, 2, rank_data, &rank_mat);
    
    size_t rank;
    err = matrix_float_rank(rank_mat, &rank);
    if (err == TENSOR_SUCCESS) {
        printf("   Matrix rank: %zu (expected: 1, rank-deficient)\n", rank);
    } else {
        printf("   Rank computation failed\n");
    }
    
    matrix_float_destroy(rank_mat);
    
    // Example 15: LU Decomposition
    printf("\n15. LU Decomposition:\n");
    MatrixFloatHandle lu_mat, L_lu, U_lu;
    float lu_data[] = {4.0f, 3.0f, 6.0f, 3.0f};
    matrix_float_create(2, 2, lu_data, &lu_mat);
    
    size_t* pivot;
    size_t pivot_size;
    err = matrix_float_lu(lu_mat, &L_lu, &U_lu, &pivot, &pivot_size);
    if (err == TENSOR_SUCCESS) {
        printf("   LU decomposition completed\n");
        printf("   Matrix factorized into lower and upper triangular matrices\n");
        
        matrix_float_destroy(L_lu);
        matrix_float_destroy(U_lu);
        free(pivot);
    } else {
        printf("   LU decomposition failed\n");
    }
    
    matrix_float_destroy(lu_mat);
    
    printf("\n=== Example completed successfully ===\n");
    printf("Library version: %s\n", tensor_c_version());
    
    return 0;
}
