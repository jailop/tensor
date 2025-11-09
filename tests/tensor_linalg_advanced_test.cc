/**
 * @file tensor_linalg_advanced_test.cc
 * @brief Tests for advanced linear algebra operations
 */

#include <gtest/gtest.h>
#include "linalg_advanced.h"
#include <cmath>

using namespace linalg;

class LinalgAdvancedTest : public ::testing::Test {
protected:
    const float eps = 1e-4f;
    
    bool near(float a, float b, float epsilon = 1e-4f) {
        return std::abs(a - b) < epsilon;
    }
};

// ============================================
// LU Decomposition Tests
// ============================================

TEST_F(LinalgAdvancedTest, LUDecomp3x3) {
    Matrix<float> A({3, 3});
    A[{0, 0}] = 2; A[{0, 1}] = -1; A[{0, 2}] = 0;
    A[{1, 0}] = -1; A[{1, 1}] = 2; A[{1, 2}] = -1;
    A[{2, 0}] = 0; A[{2, 1}] = -1; A[{2, 2}] = 2;
    
    auto result = lu_decomp(A);
    ASSERT_TRUE((std::holds_alternative<std::pair<Matrix<float>, std::vector<int>>>(result)));
    
    auto& [LU, pivots] = std::get<std::pair<Matrix<float>, std::vector<int>>>(result);
    
    // Check dimensions
    auto dims = LU.dims();
    EXPECT_EQ(dims[0], 3);
    EXPECT_EQ(dims[1], 3);
    EXPECT_EQ(pivots.size(), 3);
}

TEST_F(LinalgAdvancedTest, LUSingular) {
    Matrix<float> A({2, 2});
    A[{0, 0}] = 1; A[{0, 1}] = 2;
    A[{1, 0}] = 2; A[{1, 1}] = 4;  // Rank deficient
    
    auto result = lu_decomp(A);
    ASSERT_TRUE(std::holds_alternative<TensorError>(result));
}

// ============================================
// Linear System Solver Tests
// ============================================

TEST_F(LinalgAdvancedTest, SolveSquareSystem) {
    // Solve: 2x + y = 5
    //        x + 2y = 4
    Matrix<float> A({2, 2});
    A[{0, 0}] = 2; A[{0, 1}] = 1;
    A[{1, 0}] = 1; A[{1, 1}] = 2;
    
    Vector<float> b({2});
    b[{0}] = 5;
    b[{1}] = 4;
    
    auto result = solve(A, b);
    ASSERT_TRUE(std::holds_alternative<Vector<float>>(result));
    
    auto& x = std::get<Vector<float>>(result);
    EXPECT_TRUE(near(x[{0}], 2.0f));  // x = 2
    EXPECT_TRUE(near(x[{1}], 1.0f));  // y = 1
}

TEST_F(LinalgAdvancedTest, SolveWithLU) {
    Matrix<float> A({3, 3});
    A[{0, 0}] = 1; A[{0, 1}] = 2; A[{0, 2}] = 3;
    A[{1, 0}] = 2; A[{1, 1}] = 5; A[{1, 2}] = 3;
    A[{2, 0}] = 1; A[{2, 1}] = 0; A[{2, 2}] = 8;
    
    Vector<float> b({3});
    b[{0}] = 1;
    b[{1}] = 2;
    b[{2}] = 3;
    
    auto result = solve(A, b, SolverMethod::LU);
    ASSERT_TRUE(std::holds_alternative<Vector<float>>(result));
    
    auto& x = std::get<Vector<float>>(result);
    
    // Verify Ax = b
    for (size_t i = 0; i < 3; ++i) {
        float sum = 0;
        for (size_t j = 0; j < 3; ++j) {
            sum += A[{i, j}] * x[{j}];
        }
        EXPECT_TRUE(near(sum, b[{i}]));
    }
}

TEST_F(LinalgAdvancedTest, SolveSPDWithCholesky) {
    // Symmetric positive definite matrix
    Matrix<float> A({2, 2});
    A[{0, 0}] = 4; A[{0, 1}] = 2;
    A[{1, 0}] = 2; A[{1, 1}] = 3;
    
    Vector<float> b({2});
    b[{0}] = 6;
    b[{1}] = 5;
    
    auto result = solve(A, b, SolverMethod::Cholesky);
    ASSERT_TRUE(std::holds_alternative<Vector<float>>(result));
    
    auto& x = std::get<Vector<float>>(result);
    
    // Verify Ax = b
    float b0 = A[{0, 0}] * x[{0}] + A[{0, 1}] * x[{1}];
    float b1 = A[{1, 0}] * x[{0}] + A[{1, 1}] * x[{1}];
    
    EXPECT_TRUE(near(b0, 6.0f));
    EXPECT_TRUE(near(b1, 5.0f));
}

TEST_F(LinalgAdvancedTest, SolveMultipleRHS) {
    Matrix<float> A({2, 2});
    A[{0, 0}] = 3; A[{0, 1}] = 1;
    A[{1, 0}] = 1; A[{1, 1}] = 2;
    
    Matrix<float> B({2, 2});
    B[{0, 0}] = 9; B[{0, 1}] = 5;
    B[{1, 0}] = 8; B[{1, 1}] = 7;
    
    auto result = solve(A, B);
    ASSERT_TRUE(std::holds_alternative<Matrix<float>>(result));
    
    auto& X = std::get<Matrix<float>>(result);
    
    // Verify AX = B for first column
    float b00 = A[{0, 0}] * X[{0, 0}] + A[{0, 1}] * X[{1, 0}];
    EXPECT_TRUE(near(b00, 9.0f));
}

// ============================================
// Least Squares Tests
// ============================================

TEST_F(LinalgAdvancedTest, LeastSquaresOverdetermined) {
    // Fit line y = mx + b through points (1,1), (2,2), (3,2), (4,3)
    Matrix<float> A({4, 2});
    A[{0, 0}] = 1; A[{0, 1}] = 1;
    A[{1, 0}] = 2; A[{1, 1}] = 1;
    A[{2, 0}] = 3; A[{2, 1}] = 1;
    A[{3, 0}] = 4; A[{3, 1}] = 1;
    
    Vector<float> b({4});
    b[{0}] = 1;
    b[{1}] = 2;
    b[{2}] = 2;
    b[{3}] = 3;
    
    auto result = lstsq(A, b);
    ASSERT_TRUE(std::holds_alternative<Vector<float>>(result));
    
    auto& x = std::get<Vector<float>>(result);
    
    // Should get approximately m ≈ 0.6, b ≈ 0.5
    EXPECT_GT(x[{0}], 0.4f);
    EXPECT_LT(x[{0}], 0.8f);
}

TEST_F(LinalgAdvancedTest, LeastSquaresPerfectFit) {
    // Exact fit: y = 2x + 1
    Matrix<float> A({3, 2});
    A[{0, 0}] = 1; A[{0, 1}] = 1;
    A[{1, 0}] = 2; A[{1, 1}] = 1;
    A[{2, 0}] = 3; A[{2, 1}] = 1;
    
    Vector<float> b({3});
    b[{0}] = 3;  // 2*1 + 1
    b[{1}] = 5;  // 2*2 + 1
    b[{2}] = 7;  // 2*3 + 1
    
    auto result = lstsq(A, b);
    ASSERT_TRUE(std::holds_alternative<Vector<float>>(result));
    
    auto& x = std::get<Vector<float>>(result);
    
    EXPECT_TRUE(near(x[{0}], 2.0f));  // m = 2
    EXPECT_TRUE(near(x[{1}], 1.0f));  // b = 1
}

// ============================================
// Matrix Rank Tests
// ============================================

TEST_F(LinalgAdvancedTest, RankFullRank) {
    Matrix<float> A({3, 3});
    A[{0, 0}] = 1; A[{0, 1}] = 0; A[{0, 2}] = 0;
    A[{1, 0}] = 0; A[{1, 1}] = 1; A[{1, 2}] = 0;
    A[{2, 0}] = 0; A[{2, 1}] = 0; A[{2, 2}] = 1;
    
    size_t r = matrix_rank(A);
    EXPECT_EQ(r, 3);
}

TEST_F(LinalgAdvancedTest, RankDeficient) {
    Matrix<float> A({3, 3});
    A[{0, 0}] = 1; A[{0, 1}] = 2; A[{0, 2}] = 3;
    A[{1, 0}] = 2; A[{1, 1}] = 4; A[{1, 2}] = 6;
    A[{2, 0}] = 1; A[{2, 1}] = 0; A[{2, 2}] = 1;
    
    size_t r = matrix_rank(A);
    EXPECT_EQ(r, 2);
}

// ============================================
// Kronecker Product Tests
// ============================================

TEST_F(LinalgAdvancedTest, Kron2x2) {
    Matrix<float> A({2, 2});
    A[{0, 0}] = 1; A[{0, 1}] = 2;
    A[{1, 0}] = 3; A[{1, 1}] = 4;
    
    Matrix<float> B({2, 2});
    B[{0, 0}] = 0; B[{0, 1}] = 5;
    B[{1, 0}] = 6; B[{1, 1}] = 7;
    
    auto result = kron(A, B);
    ASSERT_TRUE(std::holds_alternative<Matrix<float>>(result));
    
    auto& C = std::get<Matrix<float>>(result);
    
    // Check dimensions
    auto dims = C.dims();
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(dims[1], 4);
    
    // Check some elements
    EXPECT_FLOAT_EQ((C[{0, 0}]), 0.0f);   // A[0,0] * B[0,0] = 1*0
    EXPECT_FLOAT_EQ((C[{0, 1}]), 5.0f);   // A[0,0] * B[0,1] = 1*5
    EXPECT_FLOAT_EQ((C[{2, 1}]), 15.0f);  // A[1,0] * B[0,1] = 3*5
    EXPECT_FLOAT_EQ((C[{2, 3}]), 20.0f);  // A[1,1] * B[0,1] = 4*5
}

TEST_F(LinalgAdvancedTest, KronIdentity) {
    Matrix<float> I({2, 2});
    I[{0, 0}] = 1; I[{0, 1}] = 0;
    I[{1, 0}] = 0; I[{1, 1}] = 1;
    
    Matrix<float> A({2, 2});
    A[{0, 0}] = 2; A[{0, 1}] = 3;
    A[{1, 0}] = 4; A[{1, 1}] = 5;
    
    auto result = kron(I, A);
    ASSERT_TRUE(std::holds_alternative<Matrix<float>>(result));
    
    auto& C = std::get<Matrix<float>>(result);
    
    // I ⊗ A should be block diagonal with A blocks
    EXPECT_FLOAT_EQ((C[{0, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((C[{0, 1}]), 3.0f);
    EXPECT_FLOAT_EQ((C[{1, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((C[{1, 1}]), 5.0f);
    
    EXPECT_FLOAT_EQ((C[{2, 2}]), 2.0f);
    EXPECT_FLOAT_EQ((C[{2, 3}]), 3.0f);
    EXPECT_FLOAT_EQ((C[{3, 2}]), 4.0f);
    EXPECT_FLOAT_EQ((C[{3, 3}]), 5.0f);
}

// ============================================
// Determinant Tests
// ============================================

TEST_F(LinalgAdvancedTest, Determinant2x2) {
    Matrix<float> A({2, 2});
    A[{0, 0}] = 4; A[{0, 1}] = 7;
    A[{1, 0}] = 2; A[{1, 1}] = 6;
    
    auto result = determinant(A);
    ASSERT_TRUE(std::holds_alternative<float>(result));
    
    float det = std::get<float>(result);
    EXPECT_TRUE(near(det, 10.0f));  // 4*6 - 7*2 = 24 - 14 = 10
}

TEST_F(LinalgAdvancedTest, Determinant3x3) {
    Matrix<float> A({3, 3});
    A[{0, 0}] = 1; A[{0, 1}] = 2; A[{0, 2}] = 3;
    A[{1, 0}] = 0; A[{1, 1}] = 1; A[{1, 2}] = 4;
    A[{2, 0}] = 5; A[{2, 1}] = 6; A[{2, 2}] = 0;
    
    auto result = determinant(A);
    ASSERT_TRUE(std::holds_alternative<float>(result));
    
    float det = std::get<float>(result);
    EXPECT_TRUE(near(det, 1.0f));  // Computed determinant
}

TEST_F(LinalgAdvancedTest, DeterminantIdentity) {
    Matrix<float> I({3, 3});
    I[{0, 0}] = 1; I[{0, 1}] = 0; I[{0, 2}] = 0;
    I[{1, 0}] = 0; I[{1, 1}] = 1; I[{1, 2}] = 0;
    I[{2, 0}] = 0; I[{2, 1}] = 0; I[{2, 2}] = 1;
    
    auto result = determinant(I);
    ASSERT_TRUE(std::holds_alternative<float>(result));
    
    float det = std::get<float>(result);
    EXPECT_TRUE(near(det, 1.0f));
}

// ============================================
// Matrix Inverse Tests
// ============================================

TEST_F(LinalgAdvancedTest, Inverse2x2) {
    Matrix<float> A({2, 2});
    A[{0, 0}] = 4; A[{0, 1}] = 7;
    A[{1, 0}] = 2; A[{1, 1}] = 6;
    
    auto result = inverse(A);
    ASSERT_TRUE(std::holds_alternative<Matrix<float>>(result));
    
    auto& Ainv = std::get<Matrix<float>>(result);
    
    // Verify A * Ainv = I
    auto I = matmul(A, Ainv);
    EXPECT_TRUE(near(I[{0, 0}], 1.0f));
    EXPECT_TRUE(near(I[{0, 1}], 0.0f));
    EXPECT_TRUE(near(I[{1, 0}], 0.0f));
    EXPECT_TRUE(near(I[{1, 1}], 1.0f));
}

TEST_F(LinalgAdvancedTest, Inverse3x3) {
    Matrix<float> A({3, 3});
    A[{0, 0}] = 1; A[{0, 1}] = 2; A[{0, 2}] = 0;
    A[{1, 0}] = 0; A[{1, 1}] = 1; A[{1, 2}] = 1;
    A[{2, 0}] = 1; A[{2, 1}] = 0; A[{2, 2}] = 1;
    
    auto result = inverse(A);
    ASSERT_TRUE(std::holds_alternative<Matrix<float>>(result));
    
    auto& Ainv = std::get<Matrix<float>>(result);
    
    // Verify A * Ainv = I
    auto I = matmul(A, Ainv);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_TRUE(near(I[{i, j}], expected));
        }
    }
}

TEST_F(LinalgAdvancedTest, InverseSingular) {
    Matrix<float> A({2, 2});
    A[{0, 0}] = 1; A[{0, 1}] = 2;
    A[{1, 0}] = 2; A[{1, 1}] = 4;  // Singular
    
    auto result = inverse(A);
    ASSERT_TRUE(std::holds_alternative<TensorError>(result));
}

TEST_F(LinalgAdvancedTest, InverseIdentity) {
    Matrix<float> I({2, 2});
    I[{0, 0}] = 1; I[{0, 1}] = 0;
    I[{1, 0}] = 0; I[{1, 1}] = 1;
    
    auto result = inverse(I);
    ASSERT_TRUE(std::holds_alternative<Matrix<float>>(result));
    
    auto& Iinv = std::get<Matrix<float>>(result);
    
    // Inverse of identity is identity
    EXPECT_TRUE(near(Iinv[{0, 0}], 1.0f));
    EXPECT_TRUE(near(Iinv[{0, 1}], 0.0f));
    EXPECT_TRUE(near(Iinv[{1, 0}], 0.0f));
    EXPECT_TRUE(near(Iinv[{1, 1}], 1.0f));
}

// ============================================
// Helper Function Tests
// ============================================

TEST_F(LinalgAdvancedTest, IsSymmetric) {
    Matrix<float> A({3, 3});
    A[{0, 0}] = 1; A[{0, 1}] = 2; A[{0, 2}] = 3;
    A[{1, 0}] = 2; A[{1, 1}] = 4; A[{1, 2}] = 5;
    A[{2, 0}] = 3; A[{2, 1}] = 5; A[{2, 2}] = 6;
    
    EXPECT_TRUE(detail::is_symmetric(A));
}

TEST_F(LinalgAdvancedTest, IsNotSymmetric) {
    Matrix<float> A({2, 2});
    A[{0, 0}] = 1; A[{0, 1}] = 2;
    A[{1, 0}] = 3; A[{1, 1}] = 4;
    
    EXPECT_FALSE(detail::is_symmetric(A));
}

TEST_F(LinalgAdvancedTest, IsPositiveDefinite) {
    Matrix<float> A({2, 2});
    A[{0, 0}] = 4; A[{0, 1}] = 1;
    A[{1, 0}] = 1; A[{1, 1}] = 3;
    
    EXPECT_TRUE(detail::is_positive_definite(A));
}
