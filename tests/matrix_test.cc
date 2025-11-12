/**
 * @file matrix_test.cc
 * @brief Test the new Matrix class using Google Test
 */

#include <gtest/gtest.h>
#include "tensor_matrix.h"

using namespace tensor;

class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MatrixTest, Construction) {
    Matrix<float> m1(3, 3);
    m1.fill(1.0f);
    
    EXPECT_EQ(m1.rows(), 3);
    EXPECT_EQ(m1.cols(), 3);
    EXPECT_TRUE(m1.is_square());
    EXPECT_NEAR((m1[{0, 0}]), 1.0f, 1e-5f);
}

TEST_F(MatrixTest, InitializerList) {
    Matrix<float> m2 = {{1, 2, 3}, 
                        {4, 5, 6}, 
                        {7, 8, 9}};
    
    EXPECT_NEAR((m2[{0, 0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((m2[{1, 1}]), 5.0f, 1e-5f);
    EXPECT_NEAR((m2[{2, 2}]), 9.0f, 1e-5f);
    EXPECT_NEAR((m2[{0, 2}]), 3.0f, 1e-5f);
}

TEST_F(MatrixTest, IdentityMatrix) {
    auto identity = Matrix<float>::eye(3);
    
    EXPECT_EQ(identity.rows(), 3);
    EXPECT_EQ(identity.cols(), 3);
    EXPECT_NEAR(identity.trace(), 3.0f, 1e-5f);
    EXPECT_NEAR((identity[{0, 0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((identity[{1, 1}]), 1.0f, 1e-5f);
    EXPECT_NEAR((identity[{0, 1}]), 0.0f, 1e-5f);
}

TEST_F(MatrixTest, DiagonalExtraction) {
    Matrix<float> m2 = {{1, 2, 3}, 
                        {4, 5, 6}, 
                        {7, 8, 9}};
    auto diag = m2.diag();
    
    ASSERT_EQ(diag.dims()[0], 3);
    EXPECT_NEAR((diag[{0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((diag[{1}]), 5.0f, 1e-5f);
    EXPECT_NEAR((diag[{2}]), 9.0f, 1e-5f);
}

TEST_F(MatrixTest, Transpose) {
    Matrix<float> m3 = {{1, 2}, {3, 4}, {5, 6}};
    auto m3_t = m3.transpose();
    
    EXPECT_EQ(m3.rows(), 3);
    EXPECT_EQ(m3.cols(), 2);
    EXPECT_EQ(m3_t.rows(), 2);
    EXPECT_EQ(m3_t.cols(), 3);
    EXPECT_NEAR((m3_t[{0, 0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((m3_t[{1, 0}]), 2.0f, 1e-5f);
    EXPECT_NEAR((m3_t[{0, 2}]), 5.0f, 1e-5f);
}

TEST_F(MatrixTest, MatrixMultiplication) {
    Matrix<float> a = {{1, 2}, {3, 4}};
    Matrix<float> b = {{5, 6}, {7, 8}};
    auto c = a.matmul(b);
    
    EXPECT_NEAR((c[{0, 0}]), 19.0f, 1e-5f);
    EXPECT_NEAR((c[{0, 1}]), 22.0f, 1e-5f);
    EXPECT_NEAR((c[{1, 0}]), 43.0f, 1e-5f);
    EXPECT_NEAR((c[{1, 1}]), 50.0f, 1e-5f);
}

TEST_F(MatrixTest, MatrixVectorMultiplication) {
    Matrix<float> a = {{1, 2}, {3, 4}};
    Tensor<float, 1> v({2});
    v[{0}] = 1.0f;
    v[{1}] = 2.0f;
    auto result = a.matvec(v);
    
    EXPECT_NEAR((result[{0}]), 5.0f, 1e-5f);
    EXPECT_NEAR((result[{1}]), 11.0f, 1e-5f);
}

TEST_F(MatrixTest, RowExtraction) {
    Matrix<float> m2 = {{1, 2, 3}, 
                        {4, 5, 6}, 
                        {7, 8, 9}};
    auto row1 = m2.row(1);
    
    EXPECT_NEAR((row1[{0}]), 4.0f, 1e-5f);
    EXPECT_NEAR((row1[{1}]), 5.0f, 1e-5f);
    EXPECT_NEAR((row1[{2}]), 6.0f, 1e-5f);
}

TEST_F(MatrixTest, ColumnExtraction) {
    Matrix<float> m2 = {{1, 2, 3}, 
                        {4, 5, 6}, 
                        {7, 8, 9}};
    auto col1 = m2.col(1);
    
    EXPECT_NEAR((col1[{0}]), 2.0f, 1e-5f);
    EXPECT_NEAR((col1[{1}]), 5.0f, 1e-5f);
    EXPECT_NEAR((col1[{2}]), 8.0f, 1e-5f);
}

TEST_F(MatrixTest, BlockExtraction) {
    Matrix<float> m2 = {{1, 2, 3}, 
                        {4, 5, 6}, 
                        {7, 8, 9}};
    auto block = m2.block(0, 0, 2, 2);
    
    EXPECT_EQ(block.rows(), 2);
    EXPECT_EQ(block.cols(), 2);
    EXPECT_NEAR((block[{0, 0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((block[{1, 1}]), 5.0f, 1e-5f);
}

TEST_F(MatrixTest, FrobeniusNorm) {
    Matrix<float> m2 = {{1, 2, 3}, 
                        {4, 5, 6}, 
                        {7, 8, 9}};
    float norm = m2.frobenius_norm();
    
    EXPECT_NEAR(norm, 16.8819f, 0.001f);
}

TEST_F(MatrixTest, L1Norm) {
    Matrix<float> m2 = {{1, 2, 3}, 
                        {4, 5, 6}, 
                        {7, 8, 9}};
    float norm = m2.norm_l1();
    
    EXPECT_NEAR(norm, 18.0f, 1e-5f);
}

TEST_F(MatrixTest, InfNorm) {
    Matrix<float> m2 = {{1, 2, 3}, 
                        {4, 5, 6}, 
                        {7, 8, 9}};
    float norm = m2.norm_inf();
    
    EXPECT_NEAR(norm, 24.0f, 1e-5f);
}

TEST_F(MatrixTest, SymmetryCheck) {
    Matrix<float> symmetric = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};
    Matrix<float> nonsymmetric = {{1, 2}, {3, 4}};
    
    EXPECT_TRUE(symmetric.is_symmetric());
    EXPECT_FALSE(nonsymmetric.is_symmetric());
}

TEST_F(MatrixTest, Trace) {
    Matrix<float> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    
    EXPECT_NEAR(m.trace(), 15.0f, 1e-5f);
}

TEST_F(MatrixTest, ZerosFactory) {
    auto zeros = Matrix<float>::zeros(2, 3);
    
    EXPECT_EQ(zeros.rows(), 2);
    EXPECT_EQ(zeros.cols(), 3);
    EXPECT_NEAR((zeros[{0, 0}]), 0.0f, 1e-5f);
    EXPECT_NEAR((zeros[{1, 2}]), 0.0f, 1e-5f);
}

TEST_F(MatrixTest, OnesFactory) {
    auto ones = Matrix<float>::ones(2, 3);
    
    EXPECT_EQ(ones.rows(), 2);
    EXPECT_EQ(ones.cols(), 3);
    EXPECT_NEAR((ones[{0, 0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((ones[{1, 2}]), 1.0f, 1e-5f);
}

TEST_F(MatrixTest, FromDiag) {
    Tensor<float, 1> diag_vec({3});
    diag_vec[{0}] = 1.0f;
    diag_vec[{1}] = 2.0f;
    diag_vec[{2}] = 3.0f;
    auto diag_mat = Matrix<float>::from_diag(diag_vec);
    
    EXPECT_NEAR((diag_mat[{0, 0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((diag_mat[{1, 1}]), 2.0f, 1e-5f);
    EXPECT_NEAR((diag_mat[{2, 2}]), 3.0f, 1e-5f);
    EXPECT_NEAR((diag_mat[{0, 1}]), 0.0f, 1e-5f);
}

TEST_F(MatrixTest, CopyConstructor) {
    Matrix<float> m1 = {{1, 2}, {3, 4}};
    Matrix<float> m2 = m1;
    
    EXPECT_EQ(m2.rows(), 2);
    EXPECT_EQ(m2.cols(), 2);
    EXPECT_NEAR((m2[{0, 0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((m2[{1, 1}]), 4.0f, 1e-5f);
}

TEST_F(MatrixTest, ConversionFromTensor) {
    Tensor<float, 2> t({2, 2});
    t[{0, 0}] = 1.0f;
    t[{0, 1}] = 2.0f;
    t[{1, 0}] = 3.0f;
    t[{1, 1}] = 4.0f;
    
    Matrix<float> m(t);
    
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 2);
    EXPECT_NEAR((m[{0, 0}]), 1.0f, 1e-5f);
    EXPECT_NEAR((m[{1, 1}]), 4.0f, 1e-5f);
}
