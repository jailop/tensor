#include "tensor.h"
#include <gtest/gtest.h>

using namespace tensor;

class TensorIndexingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Take/Put Operations
// ============================================================================

TEST_F(TensorIndexingTest, TakeBasic) {
    Tensor<float, 1> t({10});
    for (size_t i = 0; i < 10; ++i) {
        t[{i}] = static_cast<float>(i);
    }
    
    std::vector<size_t> indices = {0, 2, 5, 9};
    auto result = t.take(indices);
    
    ASSERT_EQ(result.dims()[0], 4);
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 2.0f);
    EXPECT_FLOAT_EQ(result[{2}], 5.0f);
    EXPECT_FLOAT_EQ(result[{3}], 9.0f);
}

TEST_F(TensorIndexingTest, Take2D) {
    Tensor<float, 2> t({3, 4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            t[{i, j}] = static_cast<float>(i * 4 + j);
        }
    }
    
    // Take elements at flat indices 0, 5, 11
    std::vector<size_t> indices = {0, 5, 11};
    auto result = t.take(indices);
    
    ASSERT_EQ(result.dims()[0], 3);
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 5.0f);
    EXPECT_FLOAT_EQ(result[{2}], 11.0f);
}

TEST_F(TensorIndexingTest, TakeOutOfBounds) {
    Tensor<float, 1> t({5});
    for (size_t i = 0; i < 5; ++i) {
        t[{i}] = static_cast<float>(i);
    }
    std::vector<size_t> indices = {0, 10};
    
    // Out of bounds indices return 0
    auto result = t.take(indices);
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 0.0f);  // Out of bounds -> 0
}

TEST_F(TensorIndexingTest, PutBasic) {
    Tensor<float, 1> t({10});
    t.fill(0.0f);
    
    std::vector<size_t> indices = {0, 2, 5};
    std::vector<float> values = {10.0f, 20.0f, 30.0f};
    t.put(indices, values);
    
    EXPECT_FLOAT_EQ(t[{0}], 10.0f);
    EXPECT_FLOAT_EQ(t[{1}], 0.0f);
    EXPECT_FLOAT_EQ(t[{2}], 20.0f);
    EXPECT_FLOAT_EQ(t[{5}], 30.0f);
}

TEST_F(TensorIndexingTest, PutSingleValue) {
    Tensor<float, 1> t({10});
    t.fill(0.0f);
    
    std::vector<size_t> indices = {1, 3, 7};
    std::vector<float> values = {99.0f, 99.0f, 99.0f};
    t.put(indices, values);
    
    EXPECT_FLOAT_EQ(t[{1}], 99.0f);
    EXPECT_FLOAT_EQ(t[{3}], 99.0f);
    EXPECT_FLOAT_EQ(t[{7}], 99.0f);
    EXPECT_FLOAT_EQ(t[{0}], 0.0f);
}

TEST_F(TensorIndexingTest, PutOutOfBounds) {
    Tensor<float, 1> t({5});
    t.fill(0.0f);
    std::vector<size_t> indices = {0, 10};
    std::vector<float> values = {1.0f, 2.0f};
    
    // Out of bounds indices are silently ignored
    t.put(indices, values);
    EXPECT_FLOAT_EQ(t[{0}], 1.0f);
    EXPECT_FLOAT_EQ(t[{1}], 0.0f);  // Not modified
}

TEST_F(TensorIndexingTest, PutSizeMismatch) {
    Tensor<float, 1> t({5});
    t.fill(0.0f);
    std::vector<size_t> indices = {0, 1};
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    
    // Uses min of both sizes, extra values ignored
    t.put(indices, values);
    EXPECT_FLOAT_EQ(t[{0}], 1.0f);
    EXPECT_FLOAT_EQ(t[{1}], 2.0f);
}

// ============================================================================
// Boolean Indexing / Masked Operations
// ============================================================================

TEST_F(TensorIndexingTest, MaskedSelect) {
    Tensor<float, 1> t({5});
    for (size_t i = 0; i < 5; ++i) {
        t[{i}] = static_cast<float>(i);
    }
    
    Tensor<bool, 1> mask({5});
    mask[{0}] = true;
    mask[{1}] = false;
    mask[{2}] = true;
    mask[{3}] = false;
    mask[{4}] = true;
    
    auto result = t.masked_select(mask);
    
    ASSERT_EQ(result.dims()[0], 3);
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 2.0f);
    EXPECT_FLOAT_EQ(result[{2}], 4.0f);
}

TEST_F(TensorIndexingTest, MaskedSelect2D) {
    Tensor<float, 2> t({2, 3});
    t[{0, 0}] = 1.0f;
    t[{0, 1}] = 2.0f;
    t[{0, 2}] = 3.0f;
    t[{1, 0}] = 4.0f;
    t[{1, 1}] = 5.0f;
    t[{1, 2}] = 6.0f;
    
    Tensor<bool, 2> mask({2, 3});
    mask[{0, 0}] = true;
    mask[{0, 1}] = false;
    mask[{0, 2}] = true;
    mask[{1, 0}] = false;
    mask[{1, 1}] = true;
    mask[{1, 2}] = true;
    
    auto result = t.masked_select(mask);
    
    ASSERT_EQ(result.dims()[0], 4);
    EXPECT_FLOAT_EQ(result[{0}], 1.0f);
    EXPECT_FLOAT_EQ(result[{1}], 3.0f);
    EXPECT_FLOAT_EQ(result[{2}], 5.0f);
    EXPECT_FLOAT_EQ(result[{3}], 6.0f);
}

TEST_F(TensorIndexingTest, MaskedSelectAllFalse) {
    Tensor<float, 1> t({5});
    t.fill(1.0f);
    
    Tensor<bool, 1> mask({5});
    mask.fill(false);
    
    auto result = t.masked_select(mask);
    
    ASSERT_EQ(result.dims()[0], 0);
}

TEST_F(TensorIndexingTest, MaskedSelectShapeMismatch) {
    Tensor<float, 1> t({5});
    t.fill(1.0f);
    Tensor<bool, 1> mask({3});
    
    // Shape mismatch returns empty tensor
    auto result = t.masked_select(mask);
    EXPECT_EQ(result.dims()[0], 0);
}

TEST_F(TensorIndexingTest, MaskedFill) {
    Tensor<float, 1> t({5});
    t.fill(1.0f);
    
    Tensor<bool, 1> mask({5});
    mask[{0}] = false;
    mask[{1}] = true;
    mask[{2}] = false;
    mask[{3}] = true;
    mask[{4}] = false;
    
    auto result = t.masked_fill(mask, 99.0f);
    
    EXPECT_FLOAT_EQ(result[{0}], 1.0f);
    EXPECT_FLOAT_EQ(result[{1}], 99.0f);
    EXPECT_FLOAT_EQ(result[{2}], 1.0f);
    EXPECT_FLOAT_EQ(result[{3}], 99.0f);
    EXPECT_FLOAT_EQ(result[{4}], 1.0f);
}

TEST_F(TensorIndexingTest, MaskedFill2D) {
    Tensor<float, 2> t({2, 2});
    t.fill(0.0f);
    
    Tensor<bool, 2> mask({2, 2});
    mask[{0, 0}] = true;
    mask[{0, 1}] = false;
    mask[{1, 0}] = false;
    mask[{1, 1}] = true;
    
    auto result = t.masked_fill(mask, 5.0f);
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 5.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 5.0f);
}

// ============================================================================
// Where (Conditional Selection)
// ============================================================================

TEST_F(TensorIndexingTest, Where1D) {
    Tensor<bool, 1> condition({5});
    condition[{0}] = true;
    condition[{1}] = false;
    condition[{2}] = true;
    condition[{3}] = false;
    condition[{4}] = true;
    
    Tensor<float, 1> x({5});
    x.fill(1.0f);
    
    Tensor<float, 1> y({5});
    y.fill(2.0f);
    
    auto result = Tensor<float, 1>::where(condition, x, y);
    
    EXPECT_FLOAT_EQ(result[{0}], 1.0f);
    EXPECT_FLOAT_EQ(result[{1}], 2.0f);
    EXPECT_FLOAT_EQ(result[{2}], 1.0f);
    EXPECT_FLOAT_EQ(result[{3}], 2.0f);
    EXPECT_FLOAT_EQ(result[{4}], 1.0f);
}

TEST_F(TensorIndexingTest, Where2D) {
    Tensor<bool, 2> condition({2, 2});
    condition[{0, 0}] = true;
    condition[{0, 1}] = false;
    condition[{1, 0}] = false;
    condition[{1, 1}] = true;
    
    Tensor<float, 2> x({2, 2});
    x.fill(10.0f);
    
    Tensor<float, 2> y({2, 2});
    y.fill(20.0f);
    
    auto result = Tensor<float, 2>::where(condition, x, y);
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 10.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 20.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 20.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 10.0f);
}

TEST_F(TensorIndexingTest, WhereShapeMismatch) {
    Tensor<bool, 1> condition({5});
    Tensor<float, 1> x({5});
    Tensor<float, 1> y({3});
    
    EXPECT_THROW((Tensor<float, 1>::where(condition, x, y)), std::invalid_argument);
}

// ============================================================================
// Select (Dimension Selection)
// ============================================================================

TEST_F(TensorIndexingTest, SelectRow) {
    Tensor<float, 2> t({3, 4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            t[{i, j}] = static_cast<float>(i * 4 + j);
        }
    }
    
    // Select row 1
    auto result = t.select(0, 1);
    
    ASSERT_EQ(result.dims()[0], 4);
    EXPECT_FLOAT_EQ(result[{0}], 4.0f);
    EXPECT_FLOAT_EQ(result[{1}], 5.0f);
    EXPECT_FLOAT_EQ(result[{2}], 6.0f);
    EXPECT_FLOAT_EQ(result[{3}], 7.0f);
}

TEST_F(TensorIndexingTest, SelectColumn) {
    Tensor<float, 2> t({3, 4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            t[{i, j}] = static_cast<float>(i * 4 + j);
        }
    }
    
    // Select column 2
    auto result = t.select(1, 2);
    
    ASSERT_EQ(result.dims()[0], 3);
    EXPECT_FLOAT_EQ(result[{0}], 2.0f);
    EXPECT_FLOAT_EQ(result[{1}], 6.0f);
    EXPECT_FLOAT_EQ(result[{2}], 10.0f);
}

TEST_F(TensorIndexingTest, Select3D) {
    Tensor<float, 3> t({2, 3, 4});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                t[{i, j, k}] = static_cast<float>(i * 12 + j * 4 + k);
            }
        }
    }
    
    // Select first dimension at index 1
    auto result = t.select(0, 1);
    
    ASSERT_EQ(result.dims()[0], 3);
    ASSERT_EQ(result.dims()[1], 4);
    EXPECT_FLOAT_EQ((result[{0, 0}]), 12.0f);
    EXPECT_FLOAT_EQ((result[{1, 2}]), 18.0f);
    EXPECT_FLOAT_EQ((result[{2, 3}]), 23.0f);
}

TEST_F(TensorIndexingTest, SelectOutOfBounds) {
    Tensor<float, 2> t({3, 4});
    
    // Out of bounds returns empty tensor
    auto result1 = t.select(0, 10);
    EXPECT_EQ(result1.dims()[0], 0);
    
    auto result2 = t.select(5, 0);
    EXPECT_EQ(result2.dims()[0], 0);
}

// ============================================================================
// Clip/Clamp Operations
// ============================================================================

TEST_F(TensorIndexingTest, ClipBasic) {
    Tensor<float, 1> t({5});
    t[{0}] = -10.0f;
    t[{1}] = -1.0f;
    t[{2}] = 5.0f;
    t[{3}] = 8.0f;
    t[{4}] = 15.0f;
    
    auto result = t.clip(0.0f, 10.0f);
    
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 0.0f);
    EXPECT_FLOAT_EQ(result[{2}], 5.0f);
    EXPECT_FLOAT_EQ(result[{3}], 8.0f);
    EXPECT_FLOAT_EQ(result[{4}], 10.0f);
}

TEST_F(TensorIndexingTest, ClipMin) {
    Tensor<float, 1> t({5});
    t[{0}] = -5.0f;
    t[{1}] = 0.0f;
    t[{2}] = 5.0f;
    t[{3}] = 10.0f;
    t[{4}] = 15.0f;
    
    auto result = t.clip_min(0.0f);
    
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 0.0f);
    EXPECT_FLOAT_EQ(result[{2}], 5.0f);
    EXPECT_FLOAT_EQ(result[{3}], 10.0f);
    EXPECT_FLOAT_EQ(result[{4}], 15.0f);
}

TEST_F(TensorIndexingTest, ClipMax) {
    Tensor<float, 1> t({5});
    t[{0}] = -5.0f;
    t[{1}] = 0.0f;
    t[{2}] = 5.0f;
    t[{3}] = 10.0f;
    t[{4}] = 15.0f;
    
    auto result = t.clip_max(10.0f);
    
    EXPECT_FLOAT_EQ(result[{0}], -5.0f);
    EXPECT_FLOAT_EQ(result[{1}], 0.0f);
    EXPECT_FLOAT_EQ(result[{2}], 5.0f);
    EXPECT_FLOAT_EQ(result[{3}], 10.0f);
    EXPECT_FLOAT_EQ(result[{4}], 10.0f);
}

TEST_F(TensorIndexingTest, ClipWithGradient) {
    Tensor<float, 1> t({5}, false, true);
    t[{0}] = -5.0f;
    t[{1}] = 1.0f;
    t[{2}] = 5.0f;
    t[{3}] = 8.0f;
    t[{4}] = 15.0f;
    
    auto result = t.clip(0.0f, 10.0f);
    
    // Check forward pass
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 1.0f);
    EXPECT_FLOAT_EQ(result[{2}], 5.0f);
    EXPECT_FLOAT_EQ(result[{3}], 8.0f);
    EXPECT_FLOAT_EQ(result[{4}], 10.0f);
    
    // Test backward pass
    Tensor<float, 1> grad({5});
    grad.fill(1.0f);
    
    result.backward(&grad);
    
    // Gradients should flow only for non-clamped values
    EXPECT_FLOAT_EQ((*t.grad())[{0}], 0.0f);  // Clamped to min
    EXPECT_FLOAT_EQ((*t.grad())[{1}], 1.0f);  // Not clamped
    EXPECT_FLOAT_EQ((*t.grad())[{2}], 1.0f);  // Not clamped
    EXPECT_FLOAT_EQ((*t.grad())[{3}], 1.0f);  // Not clamped
    EXPECT_FLOAT_EQ((*t.grad())[{4}], 0.0f);  // Clamped to max
}

