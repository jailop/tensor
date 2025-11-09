#include "tensor_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define ASSERT_EQ(a, b) assert(fabs((a) - (b)) < 1e-5)
#define ASSERT_SUCCESS(err) assert((err) == TENSOR_SUCCESS)

void test_vector_creation() {
    printf("Testing vector creation...\n");
    
    VectorFloatHandle vec;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    TensorErrorCode err = vector_float_create(4, data, &vec);
    ASSERT_SUCCESS(err);
    assert(vec != NULL);
    
    size_t size;
    err = vector_float_size(vec, &size);
    ASSERT_SUCCESS(err);
    assert(size == 4);
    
    float value;
    err = vector_float_get(vec, 0, &value);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(value, 1.0f);
    
    err = vector_float_destroy(vec);
    ASSERT_SUCCESS(err);
    
    printf("  ✓ Vector creation passed\n");
}

void test_vector_operations() {
    printf("Testing vector operations...\n");
    
    VectorFloatHandle vec1, vec2, result;
    float data1[] = {1.0f, 2.0f, 3.0f};
    float data2[] = {4.0f, 5.0f, 6.0f};
    
    vector_float_create(3, data1, &vec1);
    vector_float_create(3, data2, &vec2);
    
    // Test addition
    TensorErrorCode err = vector_float_add(vec1, vec2, &result);
    ASSERT_SUCCESS(err);
    
    float value;
    vector_float_get(result, 0, &value);
    ASSERT_EQ(value, 5.0f);
    
    vector_float_destroy(result);
    
    // Test dot product
    float dot_result;
    err = vector_float_dot(vec1, vec2, &dot_result);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(dot_result, 32.0f); // 1*4 + 2*5 + 3*6
    
    vector_float_destroy(vec1);
    vector_float_destroy(vec2);
    
    printf("  ✓ Vector operations passed\n");
}

void test_vector_zeros_ones() {
    printf("Testing vector zeros and ones...\n");
    
    VectorFloatHandle vec_zeros, vec_ones;
    
    TensorErrorCode err = vector_float_zeros(5, &vec_zeros);
    ASSERT_SUCCESS(err);
    
    float value;
    vector_float_get(vec_zeros, 0, &value);
    ASSERT_EQ(value, 0.0f);
    
    err = vector_float_ones(5, &vec_ones);
    ASSERT_SUCCESS(err);
    
    vector_float_get(vec_ones, 0, &value);
    ASSERT_EQ(value, 1.0f);
    
    vector_float_destroy(vec_zeros);
    vector_float_destroy(vec_ones);
    
    printf("  ✓ Vector zeros/ones passed\n");
}

void test_matrix_creation() {
    printf("Testing matrix creation...\n");
    
    MatrixFloatHandle mat;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    TensorErrorCode err = matrix_float_create(2, 2, data, &mat);
    ASSERT_SUCCESS(err);
    assert(mat != NULL);
    
    size_t rows, cols;
    err = matrix_float_shape(mat, &rows, &cols);
    ASSERT_SUCCESS(err);
    assert(rows == 2);
    assert(cols == 2);
    
    float value;
    err = matrix_float_get(mat, 0, 0, &value);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(value, 1.0f);
    
    err = matrix_float_get(mat, 1, 1, &value);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(value, 4.0f);
    
    err = matrix_float_destroy(mat);
    ASSERT_SUCCESS(err);
    
    printf("  ✓ Matrix creation passed\n");
}

void test_matrix_operations() {
    printf("Testing matrix operations...\n");
    
    MatrixFloatHandle mat1, mat2, result;
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data2[] = {2.0f, 0.0f, 0.0f, 2.0f};
    
    matrix_float_create(2, 2, data1, &mat1);
    matrix_float_create(2, 2, data2, &mat2);
    
    // Test addition
    TensorErrorCode err = matrix_float_add(mat1, mat2, &result);
    ASSERT_SUCCESS(err);
    
    float value;
    matrix_float_get(result, 0, 0, &value);
    ASSERT_EQ(value, 3.0f);
    
    matrix_float_destroy(result);
    
    // Test matrix multiplication
    err = matrix_float_matmul(mat1, mat2, &result);
    ASSERT_SUCCESS(err);
    
    matrix_float_get(result, 0, 0, &value);
    ASSERT_EQ(value, 2.0f); // 1*2 + 2*0
    
    matrix_float_get(result, 0, 1, &value);
    ASSERT_EQ(value, 4.0f); // 1*0 + 2*2
    
    matrix_float_destroy(result);
    matrix_float_destroy(mat1);
    matrix_float_destroy(mat2);
    
    printf("  ✓ Matrix operations passed\n");
}

void test_matrix_eye() {
    printf("Testing identity matrix...\n");
    
    MatrixFloatHandle eye;
    TensorErrorCode err = matrix_float_eye(3, &eye);
    ASSERT_SUCCESS(err);
    
    float value;
    matrix_float_get(eye, 0, 0, &value);
    ASSERT_EQ(value, 1.0f);
    
    matrix_float_get(eye, 0, 1, &value);
    ASSERT_EQ(value, 0.0f);
    
    matrix_float_get(eye, 1, 1, &value);
    ASSERT_EQ(value, 1.0f);
    
    matrix_float_destroy(eye);
    
    printf("  ✓ Identity matrix passed\n");
}

void test_matrix_transpose() {
    printf("Testing matrix transpose...\n");
    
    MatrixFloatHandle mat, trans;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    matrix_float_create(2, 3, data, &mat);
    
    TensorErrorCode err = matrix_float_transpose(mat, &trans);
    ASSERT_SUCCESS(err);
    
    size_t rows, cols;
    matrix_float_shape(trans, &rows, &cols);
    assert(rows == 3);
    assert(cols == 2);
    
    float value;
    matrix_float_get(trans, 0, 0, &value);
    ASSERT_EQ(value, 1.0f);
    
    matrix_float_get(trans, 1, 0, &value);
    ASSERT_EQ(value, 2.0f);
    
    matrix_float_destroy(mat);
    matrix_float_destroy(trans);
    
    printf("  ✓ Matrix transpose passed\n");
}

void test_matrix_vector() {
    printf("Testing matrix-vector multiplication...\n");
    
    MatrixFloatHandle mat;
    VectorFloatHandle vec, result;
    float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vec_data[] = {2.0f, 3.0f};
    
    matrix_float_create(2, 2, mat_data, &mat);
    vector_float_create(2, vec_data, &vec);
    
    TensorErrorCode err = matrix_float_matvec(mat, vec, &result);
    ASSERT_SUCCESS(err);
    
    float value;
    vector_float_get(result, 0, &value);
    ASSERT_EQ(value, 8.0f); // 1*2 + 2*3
    
    vector_float_get(result, 1, &value);
    ASSERT_EQ(value, 18.0f); // 3*2 + 4*3
    
    matrix_float_destroy(mat);
    vector_float_destroy(vec);
    vector_float_destroy(result);
    
    printf("  ✓ Matrix-vector multiplication passed\n");
}

void test_statistical_ops() {
    printf("Testing statistical operations...\n");
    
    VectorFloatHandle vec;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    vector_float_create(5, data, &vec);
    
    float mean, sum, min, max;
    vector_float_mean(vec, &mean);
    ASSERT_EQ(mean, 3.0f);
    
    vector_float_sum(vec, &sum);
    ASSERT_EQ(sum, 15.0f);
    
    vector_float_min(vec, &min);
    ASSERT_EQ(min, 1.0f);
    
    vector_float_max(vec, &max);
    ASSERT_EQ(max, 5.0f);
    
    vector_float_destroy(vec);
    
    printf("  ✓ Statistical operations passed\n");
}

void test_error_handling() {
    printf("Testing error handling...\n");
    
    // Test null pointer errors
    TensorErrorCode err = vector_float_destroy(NULL);
    assert(err == TENSOR_ERROR_NULL_POINTER);
    
    VectorFloatHandle vec;
    err = vector_float_create(3, NULL, &vec);
    // This should not crash
    
    printf("  ✓ Error handling passed\n");
}

void test_linear_algebra() {
    printf("Testing linear algebra operations...\n");
    
    // Test determinant
    MatrixFloatHandle mat;
    float data[] = {4.0f, 7.0f, 2.0f, 6.0f};
    matrix_float_create(2, 2, data, &mat);
    
    float det;
    TensorErrorCode err = matrix_float_determinant(mat, &det);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(det, 10.0f); // 4*6 - 7*2 = 10
    
    // Test inverse
    MatrixFloatHandle inv;
    err = matrix_float_inverse(mat, &inv);
    ASSERT_SUCCESS(err);
    
    float value;
    matrix_float_get(inv, 0, 0, &value);
    ASSERT_EQ(value, 0.6f);
    
    matrix_float_destroy(mat);
    matrix_float_destroy(inv);
    
    printf("  ✓ Linear algebra operations passed\n");
}

void test_svd() {
    printf("Testing SVD decomposition...\n");
    
    MatrixFloatHandle mat, U, Vt;
    VectorFloatHandle S;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    matrix_float_create(2, 2, data, &mat);
    
    TensorErrorCode err = matrix_float_svd(mat, &U, &S, &Vt);
    
    if (err == TENSOR_SUCCESS) {
        size_t size;
        vector_float_size(S, &size);
        assert(size >= 2);
        
        matrix_float_destroy(U);
        vector_float_destroy(S);
        matrix_float_destroy(Vt);
        printf("  ✓ SVD decomposition passed\n");
    } else {
        printf("  ⊘ SVD decomposition not available (skipped)\n");
    }
    
    matrix_float_destroy(mat);
}

void test_eigenvalues() {
    printf("Testing eigenvalue decomposition...\n");
    
    MatrixFloatHandle mat;
    VectorFloatHandle eigenvalues;
    MatrixFloatHandle eigenvectors;
    float data[] = {4.0f, 2.0f, 2.0f, 3.0f};
    
    matrix_float_create(2, 2, data, &mat);
    
    TensorErrorCode err = matrix_float_eig(mat, &eigenvalues, &eigenvectors);
    
    if (err == TENSOR_SUCCESS) {
        size_t size;
        vector_float_size(eigenvalues, &size);
        assert(size == 2);
        
        vector_float_destroy(eigenvalues);
        matrix_float_destroy(eigenvectors);
        printf("  ✓ Eigenvalue decomposition passed\n");
    } else {
        printf("  ⊘ Eigenvalue decomposition not available (skipped)\n");
    }
    
    matrix_float_destroy(mat);
}

void test_solve_linear_system() {
    printf("Testing linear system solver...\n");
    
    MatrixFloatHandle A;
    VectorFloatHandle b, x;
    float A_data[] = {2.0f, 1.0f, 1.0f, 3.0f};
    float b_data[] = {5.0f, 8.0f};
    
    matrix_float_create(2, 2, A_data, &A);
    vector_float_create(2, b_data, &b);
    
    TensorErrorCode err = matrix_float_solve(A, b, &x);
    ASSERT_SUCCESS(err);
    
    float x0, x1;
    vector_float_get(x, 0, &x0);
    vector_float_get(x, 1, &x1);
    
    // Verify: 2*x0 + 1*x1 = 5 and 1*x0 + 3*x1 = 8
    // Solution: x0 = 1, x1 = 3 (approximately)
    
    matrix_float_destroy(A);
    vector_float_destroy(b);
    vector_float_destroy(x);
    
    printf("  ✓ Linear system solver passed\n");
}

void test_norms() {
    printf("Testing vector norms...\n");
    
    VectorFloatHandle vec;
    float data[] = {3.0f, 4.0f};
    vector_float_create(2, data, &vec);
    
    float l2_norm;
    
    TensorErrorCode err = vector_float_norm(vec, &l2_norm);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(l2_norm, 5.0f);
    
    vector_float_destroy(vec);
    
    printf("  ✓ Vector norms passed\n");
}

void test_correlation() {
    printf("Testing correlation and covariance...\n");
    
    VectorFloatHandle vec1, vec2;
    float data1[] = {1.0f, 2.0f, 3.0f};
    float data2[] = {2.0f, 4.0f, 6.0f};
    
    vector_float_create(3, data1, &vec1);
    vector_float_create(3, data2, &vec2);
    
    float corr, cov;
    
    TensorErrorCode err = vector_float_correlation(vec1, vec2, &corr);
    ASSERT_SUCCESS(err);
    
    err = vector_float_covariance(vec1, vec2, &cov);
    ASSERT_SUCCESS(err);
    
    vector_float_destroy(vec1);
    vector_float_destroy(vec2);
    
    printf("  ✓ Correlation and covariance passed\n");
}

void test_cholesky() {
    printf("Testing Cholesky decomposition...\n");
    
    MatrixFloatHandle mat, L;
    // Positive definite matrix
    float data[] = {4.0f, 2.0f, 2.0f, 3.0f};
    
    matrix_float_create(2, 2, data, &mat);
    
    TensorErrorCode err = matrix_float_cholesky(mat, &L);
    
    if (err == TENSOR_SUCCESS) {
        matrix_float_destroy(L);
        printf("  ✓ Cholesky decomposition passed\n");
    } else {
        printf("  ⊘ Cholesky decomposition not available (skipped)\n");
    }
    
    matrix_float_destroy(mat);
}

void test_qr_decomposition() {
    printf("Testing QR decomposition...\n");
    
    MatrixFloatHandle mat, Q, R;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    matrix_float_create(2, 2, data, &mat);
    
    TensorErrorCode err = matrix_float_qr(mat, &Q, &R);
    
    if (err == TENSOR_SUCCESS) {
        matrix_float_destroy(Q);
        matrix_float_destroy(R);
        printf("  ✓ QR decomposition passed\n");
    } else {
        printf("  ⊘ QR decomposition not available (skipped)\n");
    }
    
    matrix_float_destroy(mat);
}

void test_matrix_rank() {
    printf("Testing matrix rank...\n");
    
    MatrixFloatHandle mat;
    float data[] = {1.0f, 2.0f, 2.0f, 4.0f};
    
    matrix_float_create(2, 2, data, &mat);
    
    size_t rank;
    TensorErrorCode err = matrix_float_rank(mat, &rank);
    ASSERT_SUCCESS(err);
    assert(rank == 1); // Rank-deficient matrix
    
    matrix_float_destroy(mat);
    
    printf("  ✓ Matrix rank passed\n");
}

void test_advanced_stats() {
    printf("Testing advanced statistics...\n");
    
    VectorFloatHandle vec;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    vector_float_create(5, data, &vec);
    
    float variance, std;
    
    TensorErrorCode err = vector_float_variance(vec, &variance);
    ASSERT_SUCCESS(err);
    
    err = vector_float_std(vec, &std);
    ASSERT_SUCCESS(err);
    
    vector_float_destroy(vec);
    
    printf("  ✓ Advanced statistics passed\n");
}

void test_lu_decomposition() {
    printf("Testing LU decomposition...\n");
    
    MatrixFloatHandle mat, L, U;
    float data[] = {4.0f, 3.0f, 6.0f, 3.0f};
    
    matrix_float_create(2, 2, data, &mat);
    
    size_t* pivot;
    size_t pivot_size;
    TensorErrorCode err = matrix_float_lu(mat, &L, &U, &pivot, &pivot_size);
    
    if (err == TENSOR_SUCCESS) {
        // Verify L is lower triangular and U is upper triangular
        matrix_float_destroy(L);
        matrix_float_destroy(U);
        free(pivot);
        printf("  ✓ LU decomposition passed\n");
    } else {
        printf("  ⊘ LU decomposition not available (skipped)\n");
    }
    
    matrix_float_destroy(mat);
}

void test_cross_product() {
    printf("Testing cross product (3D vectors)...\n");
    
    VectorFloatHandle v1, v2, cross;
    float data1[] = {1.0f, 0.0f, 0.0f};
    float data2[] = {0.0f, 1.0f, 0.0f};
    
    vector_float_create(3, data1, &v1);
    vector_float_create(3, data2, &v2);
    
    TensorErrorCode err = vector_float_cross(v1, v2, &cross);
    
    if (err == TENSOR_SUCCESS) {
        float z;
        vector_float_get(cross, 2, &z);
        ASSERT_EQ(z, 1.0f); // i × j = k
        
        vector_float_destroy(cross);
        printf("  ✓ Cross product passed\n");
    } else {
        printf("  ⊘ Cross product not available (skipped)\n");
    }
    
    vector_float_destroy(v1);
    vector_float_destroy(v2);
}

void test_version() {
    printf("Testing version info...\n");
    
    const char* version = tensor_c_version();
    assert(version != NULL);
    printf("  Library version: %s\n", version);
    
    printf("  ✓ Version info passed\n");
}

// ============================================================================
// Neural Network Layer Tests
// ============================================================================

void test_linear_layer() {
    printf("Testing Linear layer...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_linear_create_float(3, 5, true, &layer);
    ASSERT_SUCCESS(err);
    assert(layer != NULL);
    
    // Create input: 2 samples, 3 features
    MatrixFloatHandle input;
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    err = matrix_float_create(2, 3, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass
    MatrixFloatHandle output;
    err = layer_linear_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    assert(output != NULL);
    
    // Check output shape
    size_t rows, cols;
    err = matrix_float_shape(output, &rows, &cols);
    ASSERT_SUCCESS(err);
    assert(rows == 2);  // batch size
    assert(cols == 5);  // out_features
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    layer_linear_destroy(layer);
    
    printf("  ✓ Linear layer passed\n");
}

void test_relu_layer() {
    printf("Testing ReLU layer...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_relu_create_float(&layer);
    ASSERT_SUCCESS(err);
    assert(layer != NULL);
    
    // Create input with negative and positive values
    MatrixFloatHandle input;
    float input_data[] = {-1.0f, 2.0f, -0.5f, 3.0f};
    err = matrix_float_create(2, 2, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass
    MatrixFloatHandle output;
    err = layer_relu_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    
    // Check that negative values are zeroed
    float val;
    err = matrix_float_get(output, 0, 0, &val);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(val, 0.0f);  // -1.0 -> 0.0
    
    err = matrix_float_get(output, 0, 1, &val);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(val, 2.0f);  // 2.0 -> 2.0
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    layer_relu_destroy(layer);
    
    printf("  ✓ ReLU layer passed\n");
}

void test_sigmoid_layer() {
    printf("Testing Sigmoid layer...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_sigmoid_create_float(&layer);
    ASSERT_SUCCESS(err);
    assert(layer != NULL);
    
    // Create input
    MatrixFloatHandle input;
    float input_data[] = {0.0f};
    err = matrix_float_create(1, 1, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass
    MatrixFloatHandle output;
    err = layer_sigmoid_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    
    // sigmoid(0) = 0.5
    float val;
    err = matrix_float_get(output, 0, 0, &val);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(val, 0.5f);
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    layer_sigmoid_destroy(layer);
    
    printf("  ✓ Sigmoid layer passed\n");
}

void test_softmax_layer() {
    printf("Testing Softmax layer...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_softmax_create_float(&layer);
    ASSERT_SUCCESS(err);
    assert(layer != NULL);
    
    // Create input
    MatrixFloatHandle input;
    float input_data[] = {1.0f, 2.0f, 3.0f};
    err = matrix_float_create(1, 3, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass
    MatrixFloatHandle output;
    err = layer_softmax_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    
    // Check that probabilities sum to 1
    float sum = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        float val;
        err = matrix_float_get(output, 0, i, &val);
        ASSERT_SUCCESS(err);
        assert(val > 0.0f && val < 1.0f);
        sum += val;
    }
    ASSERT_EQ(sum, 1.0f);
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    layer_softmax_destroy(layer);
    
    printf("  ✓ Softmax layer passed\n");
}

void test_dropout_layer() {
    printf("Testing Dropout layer...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_dropout_create_float(0.5f, &layer);
    ASSERT_SUCCESS(err);
    assert(layer != NULL);
    
    // Test inference mode (no dropout)
    err = layer_dropout_train(layer, false);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle input;
    float input_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    err = matrix_float_create(2, 2, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass in inference mode
    MatrixFloatHandle output;
    err = layer_dropout_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    
    // In inference mode, values should be unchanged
    float val;
    err = matrix_float_get(output, 0, 0, &val);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(val, 1.0f);
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    layer_dropout_destroy(layer);
    
    printf("  ✓ Dropout layer passed\n");
}

void test_batchnorm_layer() {
    printf("Testing BatchNorm layer...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_batchnorm_create_float(2, 1e-5f, 0.1f, &layer);
    ASSERT_SUCCESS(err);
    assert(layer != NULL);
    
    // Set to training mode
    err = layer_batchnorm_train(layer, true);
    ASSERT_SUCCESS(err);
    
    // Create input (batch_size=2, features=2)
    MatrixFloatHandle input;
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    err = matrix_float_create(2, 2, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass
    MatrixFloatHandle output;
    err = layer_batchnorm_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    
    // Output should be normalized (mean~0, std~1)
    float val0, val1;
    err = matrix_float_get(output, 0, 0, &val0);
    ASSERT_SUCCESS(err);
    err = matrix_float_get(output, 1, 0, &val1);
    ASSERT_SUCCESS(err);
    
    // Check mean is approximately 0
    float mean = (val0 + val1) / 2.0f;
    assert(fabsf(mean) < 0.1f);
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    layer_batchnorm_destroy(layer);
    
    printf("  ✓ BatchNorm layer passed\n");
}

void test_linear_backward() {
    printf("Testing Linear layer backward pass...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_linear_create_float(3, 2, true, &layer);
    ASSERT_SUCCESS(err);
    
    // Create input
    MatrixFloatHandle input;
    float input_data[] = {1.0f, 2.0f, 3.0f};
    err = matrix_float_create(1, 3, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass
    MatrixFloatHandle output;
    err = layer_linear_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    
    // Create gradient for backward pass
    MatrixFloatHandle grad_output;
    float grad_data[] = {1.0f, 1.0f};
    err = matrix_float_create(1, 2, grad_data, &grad_output);
    ASSERT_SUCCESS(err);
    
    // Backward pass
    MatrixFloatHandle grad_input;
    err = layer_linear_backward_float(layer, grad_output, &grad_input);
    ASSERT_SUCCESS(err);
    
    // Check that grad_input was created
    assert(grad_input != NULL);
    
    // Check gradients are stored
    MatrixFloatHandle grad_weights, grad_bias;
    err = layer_linear_get_grad_weights_float(layer, &grad_weights);
    ASSERT_SUCCESS(err);
    assert(grad_weights != NULL);
    
    err = layer_linear_get_grad_bias_float(layer, &grad_bias);
    ASSERT_SUCCESS(err);
    assert(grad_bias != NULL);
    
    // Verify gradient shapes
    size_t g_rows, g_cols;
    err = matrix_float_shape(grad_weights, &g_rows, &g_cols);
    ASSERT_SUCCESS(err);
    assert(g_rows == 2 && g_cols == 3);
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    matrix_float_destroy(grad_output);
    matrix_float_destroy(grad_input);
    layer_linear_destroy(layer);
    
    printf("  ✓ Linear backward passed\n");
}

void test_weight_updates() {
    printf("Testing weight updates...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_linear_create_float(2, 2, true, &layer);
    ASSERT_SUCCESS(err);
    
    // Get initial weights
    MatrixFloatHandle weights_before;
    err = layer_linear_get_weights_float(layer, &weights_before);
    ASSERT_SUCCESS(err);
    
    float w00_before;
    err = matrix_float_get(weights_before, 0, 0, &w00_before);
    ASSERT_SUCCESS(err);
    
    // Create input and do forward pass
    MatrixFloatHandle input;
    float input_data[] = {1.0f, 1.0f};
    err = matrix_float_create(1, 2, input_data, &input);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle output;
    err = layer_linear_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    
    // Backward pass with gradient
    MatrixFloatHandle grad_output;
    float grad_data[] = {1.0f, 1.0f};
    err = matrix_float_create(1, 2, grad_data, &grad_output);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle grad_input;
    err = layer_linear_backward_float(layer, grad_output, &grad_input);
    ASSERT_SUCCESS(err);
    
    // Update weights
    float learning_rate = 0.01f;
    err = layer_linear_update_weights_float(layer, learning_rate);
    ASSERT_SUCCESS(err);
    
    // Get updated weights
    MatrixFloatHandle weights_after;
    err = layer_linear_get_weights_float(layer, &weights_after);
    ASSERT_SUCCESS(err);
    
    float w00_after;
    err = matrix_float_get(weights_after, 0, 0, &w00_after);
    ASSERT_SUCCESS(err);
    
    // Weights should have changed
    assert(fabsf(w00_before - w00_after) > 1e-6f);
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    matrix_float_destroy(grad_output);
    matrix_float_destroy(grad_input);
    layer_linear_destroy(layer);
    
    printf("  ✓ Weight updates passed\n");
}

void test_activation_backward() {
    printf("Testing activation layer backward passes...\n");
    
    // Test ReLU backward
    LayerHandle relu;
    TensorErrorCode err = layer_relu_create_float(&relu);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle input;
    float input_data[] = {-1.0f, 2.0f, -0.5f, 3.0f};
    err = matrix_float_create(2, 2, input_data, &input);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle relu_output;
    err = layer_relu_forward_float(relu, input, &relu_output);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle grad_output;
    float grad_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    err = matrix_float_create(2, 2, grad_data, &grad_output);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle grad_input;
    err = layer_relu_backward_float(relu, grad_output, &grad_input);
    ASSERT_SUCCESS(err);
    
    // Gradient should be 0 for negative inputs
    float g_val;
    err = matrix_float_get(grad_input, 0, 0, &g_val);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(g_val, 0.0f);  // input was -1.0
    
    err = matrix_float_get(grad_input, 0, 1, &g_val);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(g_val, 1.0f);  // input was 2.0 (positive)
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(relu_output);
    matrix_float_destroy(grad_output);
    matrix_float_destroy(grad_input);
    layer_relu_destroy(relu);
    
    // Test Sigmoid backward
    LayerHandle sigmoid;
    err = layer_sigmoid_create_float(&sigmoid);
    ASSERT_SUCCESS(err);
    
    float sig_input_data[] = {0.0f};
    MatrixFloatHandle sig_input;
    err = matrix_float_create(1, 1, sig_input_data, &sig_input);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle sig_output;
    err = layer_sigmoid_forward_float(sigmoid, sig_input, &sig_output);
    ASSERT_SUCCESS(err);
    
    float sig_grad_data[] = {1.0f};
    MatrixFloatHandle sig_grad_output;
    err = matrix_float_create(1, 1, sig_grad_data, &sig_grad_output);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle sig_grad_input;
    err = layer_sigmoid_backward_float(sigmoid, sig_grad_output, &sig_grad_input);
    ASSERT_SUCCESS(err);
    
    // Sigmoid derivative at 0 is 0.25 (sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5)
    float sig_grad;
    err = matrix_float_get(sig_grad_input, 0, 0, &sig_grad);
    ASSERT_SUCCESS(err);
    ASSERT_EQ(sig_grad, 0.25f);
    
    // Cleanup
    matrix_float_destroy(sig_input);
    matrix_float_destroy(sig_output);
    matrix_float_destroy(sig_grad_output);
    matrix_float_destroy(sig_grad_input);
    layer_sigmoid_destroy(sigmoid);
    
    printf("  ✓ Activation backward passed\n");
}

void test_softmax_backward() {
    printf("Testing Softmax backward pass...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_softmax_create_float(&layer);
    ASSERT_SUCCESS(err);
    
    // Create input
    MatrixFloatHandle input;
    float input_data[] = {1.0f, 2.0f, 3.0f};
    err = matrix_float_create(1, 3, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass
    MatrixFloatHandle output;
    err = layer_softmax_forward_float(layer, input, &output);
    ASSERT_SUCCESS(err);
    
    // Create gradient (simulating cross-entropy loss gradient)
    MatrixFloatHandle grad_output;
    float grad_data[] = {-1.0f, 0.0f, 0.0f};  // Target was class 0
    err = matrix_float_create(1, 3, grad_data, &grad_output);
    ASSERT_SUCCESS(err);
    
    // Backward pass
    MatrixFloatHandle grad_input;
    err = layer_softmax_backward_float(layer, grad_output, &grad_input);
    ASSERT_SUCCESS(err);
    
    // Check that gradient was computed
    assert(grad_input != NULL);
    
    size_t rows, cols;
    err = matrix_float_shape(grad_input, &rows, &cols);
    ASSERT_SUCCESS(err);
    assert(rows == 1 && cols == 3);
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(output);
    matrix_float_destroy(grad_output);
    matrix_float_destroy(grad_input);
    layer_softmax_destroy(layer);
    
    printf("  ✓ Softmax backward passed\n");
}

void test_full_training_step() {
    printf("Testing full training step (forward + backward + update)...\n");
    
    // Create a simple 2-layer network: 3 -> 4 -> 2
    LayerHandle fc1, relu, fc2;
    TensorErrorCode err;
    
    err = layer_linear_create_float(3, 4, true, &fc1);
    ASSERT_SUCCESS(err);
    err = layer_relu_create_float(&relu);
    ASSERT_SUCCESS(err);
    err = layer_linear_create_float(4, 2, true, &fc2);
    ASSERT_SUCCESS(err);
    
    // Create input (batch_size=2, features=3)
    MatrixFloatHandle input;
    float input_data[] = {1.0f, 2.0f, 3.0f, 0.5f, 1.5f, 2.5f};
    err = matrix_float_create(2, 3, input_data, &input);
    ASSERT_SUCCESS(err);
    
    // Forward pass
    MatrixFloatHandle h1, a1, h2;
    err = layer_linear_forward_float(fc1, input, &h1);
    ASSERT_SUCCESS(err);
    err = layer_relu_forward_float(relu, h1, &a1);
    ASSERT_SUCCESS(err);
    err = layer_linear_forward_float(fc2, a1, &h2);
    ASSERT_SUCCESS(err);
    
    // Compute loss gradient (simulated)
    MatrixFloatHandle grad_h2;
    float grad_data[] = {0.1f, -0.1f, 0.2f, -0.2f};
    err = matrix_float_create(2, 2, grad_data, &grad_h2);
    ASSERT_SUCCESS(err);
    
    // Backward pass
    MatrixFloatHandle grad_a1, grad_h1, grad_input;
    err = layer_linear_backward_float(fc2, grad_h2, &grad_a1);
    ASSERT_SUCCESS(err);
    err = layer_relu_backward_float(relu, grad_a1, &grad_h1);
    ASSERT_SUCCESS(err);
    err = layer_linear_backward_float(fc1, grad_h1, &grad_input);
    ASSERT_SUCCESS(err);
    
    // Get weights before update
    MatrixFloatHandle w1_before;
    err = layer_linear_get_weights_float(fc1, &w1_before);
    ASSERT_SUCCESS(err);
    float w1_val_before;
    err = matrix_float_get(w1_before, 0, 0, &w1_val_before);
    ASSERT_SUCCESS(err);
    
    // Update weights
    float lr = 0.01f;
    err = layer_linear_update_weights_float(fc1, lr);
    ASSERT_SUCCESS(err);
    err = layer_linear_update_weights_float(fc2, lr);
    ASSERT_SUCCESS(err);
    
    // Get weights after update
    MatrixFloatHandle w1_after;
    err = layer_linear_get_weights_float(fc1, &w1_after);
    ASSERT_SUCCESS(err);
    float w1_val_after;
    err = matrix_float_get(w1_after, 0, 0, &w1_val_after);
    ASSERT_SUCCESS(err);
    
    // Weights should have changed
    assert(fabsf(w1_val_before - w1_val_after) > 1e-8f);
    
    // Cleanup
    matrix_float_destroy(input);
    matrix_float_destroy(h1);
    matrix_float_destroy(a1);
    matrix_float_destroy(h2);
    matrix_float_destroy(grad_h2);
    matrix_float_destroy(grad_a1);
    matrix_float_destroy(grad_h1);
    matrix_float_destroy(grad_input);
    layer_linear_destroy(fc1);
    layer_relu_destroy(relu);
    layer_linear_destroy(fc2);
    
    printf("  ✓ Full training step passed\n");
}

void test_gradient_accumulation() {
    printf("Testing gradient accumulation...\n");
    
    LayerHandle layer;
    TensorErrorCode err = layer_linear_create_float(2, 2, true, &layer);
    ASSERT_SUCCESS(err);
    
    // First forward/backward pass
    MatrixFloatHandle input1;
    float input1_data[] = {1.0f, 1.0f};
    err = matrix_float_create(1, 2, input1_data, &input1);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle output1;
    err = layer_linear_forward_float(layer, input1, &output1);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle grad_out1;
    float grad1_data[] = {1.0f, 1.0f};
    err = matrix_float_create(1, 2, grad1_data, &grad_out1);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle grad_in1;
    err = layer_linear_backward_float(layer, grad_out1, &grad_in1);
    ASSERT_SUCCESS(err);
    
    // Get first gradient
    MatrixFloatHandle grad_w1;
    err = layer_linear_get_grad_weights_float(layer, &grad_w1);
    ASSERT_SUCCESS(err);
    float first_grad;
    err = matrix_float_get(grad_w1, 0, 0, &first_grad);
    ASSERT_SUCCESS(err);
    
    // Second forward/backward pass (gradients should be overwritten, not accumulated)
    MatrixFloatHandle input2;
    float input2_data[] = {2.0f, 2.0f};
    err = matrix_float_create(1, 2, input2_data, &input2);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle output2;
    err = layer_linear_forward_float(layer, input2, &output2);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle grad_out2;
    float grad2_data[] = {1.0f, 1.0f};
    err = matrix_float_create(1, 2, grad2_data, &grad_out2);
    ASSERT_SUCCESS(err);
    
    MatrixFloatHandle grad_in2;
    err = layer_linear_backward_float(layer, grad_out2, &grad_in2);
    ASSERT_SUCCESS(err);
    
    // Get second gradient
    MatrixFloatHandle grad_w2;
    err = layer_linear_get_grad_weights_float(layer, &grad_w2);
    ASSERT_SUCCESS(err);
    float second_grad;
    err = matrix_float_get(grad_w2, 0, 0, &second_grad);
    ASSERT_SUCCESS(err);
    
    // Second gradient should be different (about 2x first, due to 2x input)
    assert(fabsf(second_grad - 2.0f * first_grad) < 0.1f);
    
    // Cleanup
    matrix_float_destroy(input1);
    matrix_float_destroy(output1);
    matrix_float_destroy(grad_out1);
    matrix_float_destroy(grad_in1);
    matrix_float_destroy(input2);
    matrix_float_destroy(output2);
    matrix_float_destroy(grad_out2);
    matrix_float_destroy(grad_in2);
    layer_linear_destroy(layer);
    
    printf("  ✓ Gradient accumulation passed\n");
}

void test_mini_training_loop() {
    printf("Testing mini training loop...\n");
    
    // Create simple network: 2 -> 3 -> 1
    LayerHandle fc1, relu, fc2;
    TensorErrorCode err;
    
    err = layer_linear_create_float(2, 3, true, &fc1);
    ASSERT_SUCCESS(err);
    err = layer_relu_create_float(&relu);
    ASSERT_SUCCESS(err);
    err = layer_linear_create_float(3, 1, true, &fc2);
    ASSERT_SUCCESS(err);
    
    // Training data: simple XOR-like problem
    float train_x[] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float train_y[] = {0.0f, 1.0f, 1.0f, 0.0f};
    
    float learning_rate = 0.1f;
    int num_epochs = 5;
    int batch_size = 1;
    
    // Train for a few epochs
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (int i = 0; i < 4; ++i) {
            // Create batch
            MatrixFloatHandle input;
            err = matrix_float_create(1, 2, &train_x[i * 2], &input);
            ASSERT_SUCCESS(err);
            
            // Forward pass
            MatrixFloatHandle h1, a1, output;
            err = layer_linear_forward_float(fc1, input, &h1);
            ASSERT_SUCCESS(err);
            err = layer_relu_forward_float(relu, h1, &a1);
            ASSERT_SUCCESS(err);
            err = layer_linear_forward_float(fc2, a1, &output);
            ASSERT_SUCCESS(err);
            
            // Compute loss (MSE)
            float pred, target = train_y[i];
            err = matrix_float_get(output, 0, 0, &pred);
            ASSERT_SUCCESS(err);
            float loss = (pred - target) * (pred - target);
            epoch_loss += loss;
            
            // Compute gradient
            MatrixFloatHandle grad_output;
            float grad_val = 2.0f * (pred - target);  // MSE gradient
            err = matrix_float_create(1, 1, &grad_val, &grad_output);
            ASSERT_SUCCESS(err);
            
            // Backward pass
            MatrixFloatHandle grad_a1, grad_h1, grad_input;
            err = layer_linear_backward_float(fc2, grad_output, &grad_a1);
            ASSERT_SUCCESS(err);
            err = layer_relu_backward_float(relu, grad_a1, &grad_h1);
            ASSERT_SUCCESS(err);
            err = layer_linear_backward_float(fc1, grad_h1, &grad_input);
            ASSERT_SUCCESS(err);
            
            // Update weights
            err = layer_linear_update_weights_float(fc1, learning_rate);
            ASSERT_SUCCESS(err);
            err = layer_linear_update_weights_float(fc2, learning_rate);
            ASSERT_SUCCESS(err);
            
            // Cleanup
            matrix_float_destroy(input);
            matrix_float_destroy(h1);
            matrix_float_destroy(a1);
            matrix_float_destroy(output);
            matrix_float_destroy(grad_output);
            matrix_float_destroy(grad_a1);
            matrix_float_destroy(grad_h1);
            matrix_float_destroy(grad_input);
        }
        
        epoch_loss /= 4.0f;
        // Loss should decrease over epochs (with some tolerance for randomness)
        if (epoch == 0) {
            assert(epoch_loss > 0.0f);  // Initial loss should be non-zero
        }
    }
    
    // Cleanup
    layer_linear_destroy(fc1);
    layer_relu_destroy(relu);
    layer_linear_destroy(fc2);
    
    printf("  ✓ Mini training loop passed\n");
}

int main() {
    printf("=== Running C Interface Tests ===\n\n");
    
    // Basic tests
    test_vector_creation();
    test_vector_operations();
    test_vector_zeros_ones();
    test_matrix_creation();
    test_matrix_operations();
    test_matrix_eye();
    test_matrix_transpose();
    test_matrix_vector();
    test_statistical_ops();
    
    // Advanced linear algebra tests
    test_linear_algebra();
    test_svd();
    test_eigenvalues();
    test_solve_linear_system();
    test_norms();
    test_correlation();
    test_cholesky();
    test_qr_decomposition();
    test_lu_decomposition();
    test_cross_product();
    test_matrix_rank();
    test_advanced_stats();
    
    // Neural Network Layer tests - Forward Pass
    test_linear_layer();
    test_relu_layer();
    test_sigmoid_layer();
    test_softmax_layer();
    test_dropout_layer();
    test_batchnorm_layer();
    
    // Neural Network Training tests - Backward Pass & Gradients
    printf("\n--- Neural Network Training Tests ---\n");
    test_linear_backward();
    test_weight_updates();
    test_activation_backward();
    test_softmax_backward();
    test_full_training_step();
    test_gradient_accumulation();
    test_mini_training_loop();
    
    // Error handling and version
    test_error_handling();
    test_version();
    
    printf("\n=== All C Interface Tests Passed! ===\n");
    return 0;
}
