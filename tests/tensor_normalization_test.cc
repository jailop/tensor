/**
 * @file tensor_normalization_test.cc
 * @brief Test suite for tensor normalization functions and enhanced submatrix views
 */

#include <gtest/gtest.h>
#include <cmath>
#include <tensor_normalize.h>

using namespace tensor;

class TensorNormalizationTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

// ============================================================================
// L1 Normalization Tests
// ============================================================================

TEST_F(TensorNormalizationTest, L1Normalize1D) {
    Tensor<float, 1> x({5}, false);
    x[{0}] = 1.0f; x[{1}] = 2.0f; x[{2}] = 3.0f; x[{3}] = 4.0f; x[{4}] = 5.0f;
    
    auto result = normalize_l1(x);
    
    // Sum of absolute values = 15
    EXPECT_NEAR((result[{0}]), 1.0f/15.0f, 1e-5f);
    EXPECT_NEAR((result[{1}]), 2.0f/15.0f, 1e-5f);
    EXPECT_NEAR((result[{2}]), 3.0f/15.0f, 1e-5f);
}

TEST_F(TensorNormalizationTest, L2Normalize1D) {
    Tensor<float, 1> x({3}, false);
    x[{0}] = 3.0f; x[{1}] = 4.0f; x[{2}] = 0.0f;
    
    auto result = normalize_l2(x);
    
    // L2 norm = sqrt(9 + 16) = 5
    EXPECT_NEAR((result[{0}]), 3.0f/5.0f, 1e-5f);
    EXPECT_NEAR((result[{1}]), 4.0f/5.0f, 1e-5f);
    EXPECT_NEAR((result[{2}]), 0.0f, 1e-5f);
}

TEST_F(TensorNormalizationTest, ZScoreNormalize1D) {
    Tensor<float, 1> x({5}, false);
    x[{0}] = 1.0f; x[{1}] = 2.0f; x[{2}] = 3.0f; x[{3}] = 4.0f; x[{4}] = 5.0f;
    
    auto result = normalize_zscore(x);
    
    // Mean = 3, middle element should be ~0
    EXPECT_NEAR((result[{2}]), 0.0f, 1e-4f);
}

TEST_F(TensorNormalizationTest, MinMaxNormalize1D) {
    Tensor<float, 1> x({5}, false);
    x[{0}] = 0.0f; x[{1}] = 2.0f; x[{2}] = 5.0f; x[{3}] = 8.0f; x[{4}] = 10.0f;
    
    auto result = normalize_minmax(x);
    
    // Min = 0, Max = 10
    EXPECT_NEAR((result[{0}]), 0.0f, 1e-5f);
    EXPECT_NEAR((result[{2}]), 0.5f, 1e-5f);
    EXPECT_NEAR((result[{4}]), 1.0f, 1e-5f);
}

// ============================================================================
// Enhanced Submatrix View Tests
// ============================================================================

TEST_F(TensorNormalizationTest, RowView) {
    Tensor<float, 2> matrix({3, 4}, false);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            matrix[{i, j}] = i * 4 + j;
        }
    }
    
    auto row1 = matrix.row(1);
    
    ASSERT_EQ(row1.dims()[0], 4);
    EXPECT_FLOAT_EQ((row1[{0}]), 4.0f);
    EXPECT_FLOAT_EQ((row1[{1}]), 5.0f);
    EXPECT_FLOAT_EQ((row1[{2}]), 6.0f);
    EXPECT_FLOAT_EQ((row1[{3}]), 7.0f);
}

TEST_F(TensorNormalizationTest, ColView) {
    Tensor<float, 2> matrix({3, 4}, false);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            matrix[{i, j}] = i * 4 + j;
        }
    }
    
    auto col2 = matrix.col(2);
    
    ASSERT_EQ(col2.dims()[0], 3);
    EXPECT_FLOAT_EQ((col2[{0}]), 2.0f);
    EXPECT_FLOAT_EQ((col2[{1}]), 6.0f);
    EXPECT_FLOAT_EQ((col2[{2}]), 10.0f);
}

TEST_F(TensorNormalizationTest, DiagExtract) {
    Tensor<float, 2> matrix({3, 3}, false);
    matrix[{0, 0}] = 1.0f; matrix[{0, 1}] = 2.0f; matrix[{0, 2}] = 3.0f;
    matrix[{1, 0}] = 4.0f; matrix[{1, 1}] = 5.0f; matrix[{1, 2}] = 6.0f;
    matrix[{2, 0}] = 7.0f; matrix[{2, 1}] = 8.0f; matrix[{2, 2}] = 9.0f;
    
    auto diagonal = matrix.diag();
    
    ASSERT_EQ(diagonal.dims()[0], 3);
    EXPECT_FLOAT_EQ((diagonal[{0}]), 1.0f);
    EXPECT_FLOAT_EQ((diagonal[{1}]), 5.0f);
    EXPECT_FLOAT_EQ((diagonal[{2}]), 9.0f);
}

TEST_F(TensorNormalizationTest, DiagCreateMatrix) {
    Tensor<float, 1> vec({3}, false);
    vec[{0}] = 1.0f; vec[{1}] = 2.0f; vec[{2}] = 3.0f;
    
    auto matrix = diag(vec);
    
    ASSERT_EQ(matrix.dims()[0], 3);
    ASSERT_EQ(matrix.dims()[1], 3);
    EXPECT_FLOAT_EQ((matrix[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((matrix[{1, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((matrix[{2, 2}]), 3.0f);
    EXPECT_FLOAT_EQ((matrix[{0, 1}]), 0.0f);
}

TEST_F(TensorNormalizationTest, BlockView) {
    Tensor<float, 2> matrix({5, 6}, false);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            matrix[{i, j}] = i * 6 + j;
        }
    }
    
    auto block_view = matrix.block(1, 2, 2, 3);
    
    ASSERT_EQ(block_view.dims()[0], 2);
    ASSERT_EQ(block_view.dims()[1], 3);
    EXPECT_FLOAT_EQ((block_view[{0, 0}]), 8.0f);   // matrix[1, 2]
    EXPECT_FLOAT_EQ((block_view[{0, 1}]), 9.0f);   // matrix[1, 3]
    EXPECT_FLOAT_EQ((block_view[{1, 0}]), 14.0f);  // matrix[2, 2]
}

TEST_F(TensorNormalizationTest, HeadElements) {
    Tensor<float, 1> vec({10}, false);
    for (size_t i = 0; i < 10; ++i) {
        vec[{i}] = i;
    }
    
    auto head5 = vec.head(5);
    
    ASSERT_EQ(head5.dims()[0], 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ((head5[{i}]), static_cast<float>(i));
    }
}

TEST_F(TensorNormalizationTest, TailElements) {
    Tensor<float, 1> vec({10}, false);
    for (size_t i = 0; i < 10; ++i) {
        vec[{i}] = i;
    }
    
    auto tail5 = vec.tail(5);
    
    ASSERT_EQ(tail5.dims()[0], 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ((tail5[{i}]), static_cast<float>(5 + i));
    }
}

TEST_F(TensorNormalizationTest, TopRows) {
    Tensor<float, 2> matrix({5, 3}, false);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            matrix[{i, j}] = i * 3 + j;
        }
    }
    
    auto top2 = matrix.topRows(2);
    
    ASSERT_EQ(top2.dims()[0], 2);
    ASSERT_EQ(top2.dims()[1], 3);
    EXPECT_FLOAT_EQ((top2[{0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((top2[{1, 2}]), 5.0f);
}

TEST_F(TensorNormalizationTest, BottomRows) {
    Tensor<float, 2> matrix({5, 3}, false);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            matrix[{i, j}] = i * 3 + j;
        }
    }
    
    auto bottom2 = matrix.bottomRows(2);
    
    ASSERT_EQ(bottom2.dims()[0], 2);
    ASSERT_EQ(bottom2.dims()[1], 3);
    EXPECT_FLOAT_EQ((bottom2[{0, 0}]), 9.0f);   // matrix[3, 0]
    EXPECT_FLOAT_EQ((bottom2[{1, 2}]), 14.0f);  // matrix[4, 2]
}

TEST_F(TensorNormalizationTest, LeftCols) {
    Tensor<float, 2> matrix({3, 5}, false);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            matrix[{i, j}] = i * 5 + j;
        }
    }
    
    auto left2 = matrix.leftCols(2);
    
    ASSERT_EQ(left2.dims()[0], 3);
    ASSERT_EQ(left2.dims()[1], 2);
    EXPECT_FLOAT_EQ((left2[{0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((left2[{2, 1}]), 11.0f);  // matrix[2, 1]
}

TEST_F(TensorNormalizationTest, RightCols) {
    Tensor<float, 2> matrix({3, 5}, false);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            matrix[{i, j}] = i * 5 + j;
        }
    }
    
    auto right2 = matrix.rightCols(2);
    
    ASSERT_EQ(right2.dims()[0], 3);
    ASSERT_EQ(right2.dims()[1], 2);
    EXPECT_FLOAT_EQ((right2[{0, 0}]), 3.0f);   // matrix[0, 3]
    EXPECT_FLOAT_EQ((right2[{2, 1}]), 14.0f);  // matrix[2, 4]
}
