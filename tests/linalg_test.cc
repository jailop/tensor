#include <gtest/gtest.h>
#include "linalg.h"
#include <cmath>

TEST(LinalgTest, VectorNorm) {
    Vector<float> v({3});
    v[{0}] = 3.0f;
    v[{1}] = 4.0f;
    v[{2}] = 0.0f;
    
    float norm_val = linalg::norm(v);
    EXPECT_NEAR(norm_val, 5.0f, 1e-5);
}

TEST(LinalgTest, VectorDot) {
    Vector<float> a({3});
    a[{0}] = 1.0f;
    a[{1}] = 2.0f;
    a[{2}] = 3.0f;
    
    Vector<float> b({3});
    b[{0}] = 4.0f;
    b[{1}] = 5.0f;
    b[{2}] = 6.0f;
    
    float dot_product = linalg::dot(a, b);
    EXPECT_NEAR(dot_product, 32.0f, 1e-5);
}

TEST(LinalgTest, MatrixVectorMultiplication) {
    Matrix<float> mat({2, 3});
    mat[{0, 0}] = 1.0f; mat[{0, 1}] = 2.0f; mat[{0, 2}] = 3.0f;
    mat[{1, 0}] = 4.0f; mat[{1, 1}] = 5.0f; mat[{1, 2}] = 6.0f;
    
    Vector<float> vec({3});
    vec[{0}] = 1.0f;
    vec[{1}] = 2.0f;
    vec[{2}] = 3.0f;
    
    auto result = linalg::matvec(mat, vec);
    EXPECT_NEAR((result[{0}]), 14.0f, 1e-5);
    EXPECT_NEAR((result[{1}]), 32.0f, 1e-5);
}

TEST(LinalgTest, MatrixMatrixMultiplication) {
    Matrix<float> a({2, 3});
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f; a[{0, 2}] = 3.0f;
    a[{1, 0}] = 4.0f; a[{1, 1}] = 5.0f; a[{1, 2}] = 6.0f;
    
    Matrix<float> b({3, 2});
    b[{0, 0}] = 7.0f;  b[{0, 1}] = 8.0f;
    b[{1, 0}] = 9.0f;  b[{1, 1}] = 10.0f;
    b[{2, 0}] = 11.0f; b[{2, 1}] = 12.0f;
    
    auto result = linalg::matmul(a, b);
    EXPECT_NEAR((result[{0, 0}]), 58.0f, 1e-5);
    EXPECT_NEAR((result[{0, 1}]), 64.0f, 1e-5);
    EXPECT_NEAR((result[{1, 0}]), 139.0f, 1e-5);
    EXPECT_NEAR((result[{1, 1}]), 154.0f, 1e-5);
}

TEST(LinalgTest, MatrixTranspose) {
    Matrix<float> mat({2, 3});
    mat[{0, 0}] = 1.0f; mat[{0, 1}] = 2.0f; mat[{0, 2}] = 3.0f;
    mat[{1, 0}] = 4.0f; mat[{1, 1}] = 5.0f; mat[{1, 2}] = 6.0f;
    
    auto result = linalg::transpose(mat);
    EXPECT_NEAR((result[{0, 0}]), 1.0f, 1e-5);
    EXPECT_NEAR((result[{1, 1}]), 5.0f, 1e-5);
}

TEST(LinalgTest, IdentityMatrix) {
    auto eye_mat = linalg::eye<float>(3);
    EXPECT_NEAR((eye_mat[{0, 0}]), 1.0f, 1e-5);
    EXPECT_NEAR((eye_mat[{1, 1}]), 1.0f, 1e-5);
    EXPECT_NEAR((eye_mat[{0, 1}]), 0.0f, 1e-5);
}

TEST(LinalgTest, TensorSlice1D) {
    Tensor<float, 1> tensor({10});
    for (size_t i = 0; i < 10; ++i) {
        tensor[{i}] = static_cast<float>(i);
    }
    
    auto view = TensorSlice<float, 1>::slice(tensor, 0, 2, 5);
    EXPECT_NEAR((view[{0}]), 2.0f, 1e-5);
    EXPECT_NEAR((view[{1}]), 3.0f, 1e-5);
    
    view[{0}] = 99.0f;
    EXPECT_NEAR((tensor[{2}]), 99.0f, 1e-5);
}

TEST(LinalgTest, MatrixRowView) {
    Matrix<float> mat({3, 4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            mat[{i, j}] = static_cast<float>(i * 4 + j);
        }
    }
    
    auto row1 = TensorSlice<float, 2>::row(mat, 1);
    EXPECT_NEAR((row1[{0}]), 4.0f, 1e-5);
    EXPECT_NEAR((row1[{1}]), 5.0f, 1e-5);
}

TEST(LinalgTest, MatrixColumnView) {
    Matrix<float> mat({3, 4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            mat[{i, j}] = static_cast<float>(i * 4 + j);
        }
    }
    
    auto col2 = TensorSlice<float, 2>::col(mat, 2);
    EXPECT_NEAR((col2[{0}]), 2.0f, 1e-5);
    EXPECT_NEAR((col2[{1}]), 6.0f, 1e-5);
}

TEST(LinalgTest, MatrixBlockView) {
    Matrix<float> mat({4, 5});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            mat[{i, j}] = static_cast<float>(i * 5 + j);
        }
    }
    
    auto block = TensorSlice<float, 2>::block(mat, 1, 3, 2, 4);
    EXPECT_NEAR((block[{0, 0}]), 7.0f, 1e-5);
    EXPECT_NEAR((block[{1, 1}]), 13.0f, 1e-5);
}
