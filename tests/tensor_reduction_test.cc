#include "tensor.h"
#include <gtest/gtest.h>

class TensorReductionTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Cumulative Operations
// ============================================================================

TEST_F(TensorReductionTest, CumsumFlat) {
    Tensor<float, 1> t({5});
    for (size_t i = 0; i < 5; ++i) {
        t[{i}] = static_cast<float>(i + 1);  // [1, 2, 3, 4, 5]
    }
    
    auto result = t.cumsum();
    
    EXPECT_FLOAT_EQ((result[{0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 10.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 15.0f);
}

TEST_F(TensorReductionTest, CumprodFlat) {
    Tensor<float, 1> t({5});
    for (size_t i = 0; i < 5; ++i) {
        t[{i}] = static_cast<float>(i + 1);  // [1, 2, 3, 4, 5]
    }
    
    auto result = t.cumprod();
    
    EXPECT_FLOAT_EQ((result[{0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 24.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 120.0f);
}

TEST_F(TensorReductionTest, CumsumAxis0) {
    Tensor<float, 2> t({3, 2});
    t[{0, 0}] = 1.0f; t[{0, 1}] = 2.0f;
    t[{1, 0}] = 3.0f; t[{1, 1}] = 4.0f;
    t[{2, 0}] = 5.0f; t[{2, 1}] = 6.0f;
    
    auto result = t.cumsum_axis(0);
    
    // Cumulative sum along rows
    EXPECT_FLOAT_EQ((result[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{2, 0}]), 9.0f);
    EXPECT_FLOAT_EQ((result[{2, 1}]), 12.0f);
}

TEST_F(TensorReductionTest, CumsumAxis1) {
    Tensor<float, 2> t({2, 3});
    t[{0, 0}] = 1.0f; t[{0, 1}] = 2.0f; t[{0, 2}] = 3.0f;
    t[{1, 0}] = 4.0f; t[{1, 1}] = 5.0f; t[{1, 2}] = 6.0f;
    
    auto result = t.cumsum_axis(1);
    
    // Cumulative sum along columns
    EXPECT_FLOAT_EQ((result[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{0, 2}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 9.0f);
    EXPECT_FLOAT_EQ((result[{1, 2}]), 15.0f);
}

TEST_F(TensorReductionTest, CumprodAxis0) {
    Tensor<float, 2> t({3, 2});
    t[{0, 0}] = 1.0f; t[{0, 1}] = 2.0f;
    t[{1, 0}] = 3.0f; t[{1, 1}] = 4.0f;
    t[{2, 0}] = 5.0f; t[{2, 1}] = 6.0f;
    
    auto result = t.cumprod_axis(0);
    
    // Cumulative product along rows
    EXPECT_FLOAT_EQ((result[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 8.0f);
    EXPECT_FLOAT_EQ((result[{2, 0}]), 15.0f);
    EXPECT_FLOAT_EQ((result[{2, 1}]), 48.0f);
}

TEST_F(TensorReductionTest, CumprodAxis1) {
    Tensor<float, 2> t({2, 3});
    t[{0, 0}] = 1.0f; t[{0, 1}] = 2.0f; t[{0, 2}] = 3.0f;
    t[{1, 0}] = 2.0f; t[{1, 1}] = 3.0f; t[{1, 2}] = 4.0f;
    
    auto result = t.cumprod_axis(1);
    
    // Cumulative product along columns
    EXPECT_FLOAT_EQ((result[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{0, 2}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{1, 2}]), 24.0f);
}

TEST_F(TensorReductionTest, CumsumAxisOutOfBounds) {
    Tensor<float, 2> t({2, 3});
    
    EXPECT_THROW(t.cumsum_axis(5), std::out_of_range);
}

TEST_F(TensorReductionTest, CumprodAxisOutOfBounds) {
    Tensor<float, 2> t({2, 3});
    
    EXPECT_THROW(t.cumprod_axis(5), std::out_of_range);
}

// ============================================================================
// Product Reduction
// ============================================================================

TEST_F(TensorReductionTest, Prod1D) {
    Tensor<float, 1> t({4});
    t[{0}] = 2.0f;
    t[{1}] = 3.0f;
    t[{2}] = 4.0f;
    t[{3}] = 5.0f;
    
    float result = t.prod();
    
    EXPECT_FLOAT_EQ(result, 120.0f);
}

TEST_F(TensorReductionTest, Prod2D) {
    Tensor<float, 2> t({2, 3});
    t.fill(2.0f);
    
    float result = t.prod();
    
    EXPECT_FLOAT_EQ(result, 64.0f);  // 2^6
}

TEST_F(TensorReductionTest, ProdWithZero) {
    Tensor<float, 1> t({3});
    t[{0}] = 1.0f;
    t[{1}] = 0.0f;
    t[{2}] = 5.0f;
    
    float result = t.prod();
    
    EXPECT_FLOAT_EQ(result, 0.0f);
}

// ============================================================================
// Argmin/Argmax Operations
// ============================================================================

TEST_F(TensorReductionTest, Argmax1D) {
    Tensor<float, 1> t({5});
    t[{0}] = 1.0f;
    t[{1}] = 5.0f;
    t[{2}] = 3.0f;
    t[{3}] = 2.0f;
    t[{4}] = 4.0f;
    
    size_t idx = t.argmax();
    
    EXPECT_EQ(idx, 1);
}

TEST_F(TensorReductionTest, Argmin1D) {
    Tensor<float, 1> t({5});
    t[{0}] = 5.0f;
    t[{1}] = 2.0f;
    t[{2}] = 3.0f;
    t[{3}] = 1.0f;
    t[{4}] = 4.0f;
    
    size_t idx = t.argmin();
    
    EXPECT_EQ(idx, 3);
}

TEST_F(TensorReductionTest, ArgmaxAxis) {
    Tensor<float, 2> t({3, 4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            t[{i, j}] = static_cast<float>(i * 4 + j);
        }
    }
    
    // Find argmax along axis 0 (across rows)
    auto result = t.argmax_axis(0);
    
    ASSERT_EQ(result.dims()[0], 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result[{i}], 2);  // Last row has maximum
    }
}

TEST_F(TensorReductionTest, ArgminAxis) {
    Tensor<float, 2> t({3, 4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            t[{i, j}] = static_cast<float>(i * 4 + j);
        }
    }
    
    // Find argmin along axis 1 (across columns)
    auto result = t.argmin_axis(1);
    
    ASSERT_EQ(result.dims()[0], 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(result[{i}], 0);  // First column has minimum
    }
}

// ============================================================================
// Boolean Reductions
// ============================================================================

TEST_F(TensorReductionTest, AnyAllTrue) {
    Tensor<float, 1> t({5});
    t.fill(1.0f);
    
    EXPECT_TRUE(t.any());
    EXPECT_TRUE(t.all());
}

TEST_F(TensorReductionTest, AnyAllFalse) {
    Tensor<float, 1> t({5});
    t.fill(0.0f);
    
    EXPECT_FALSE(t.any());
    EXPECT_FALSE(t.all());
}

TEST_F(TensorReductionTest, AnySomeFalse) {
    Tensor<float, 1> t({5});
    t.fill(0.0f);
    t[{2}] = 1.0f;
    
    EXPECT_TRUE(t.any());
    EXPECT_FALSE(t.all());
}

TEST_F(TensorReductionTest, AllSomeFalse) {
    Tensor<float, 1> t({5});
    t.fill(1.0f);
    t[{2}] = 0.0f;
    
    EXPECT_TRUE(t.any());
    EXPECT_FALSE(t.all());
}