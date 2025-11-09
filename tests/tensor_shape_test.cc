/**
 * @file tensor_shape_test.cc
 * @brief Unit tests for tensor shape manipulation operations (flatten, repeat, tile)
 * @note reshape, squeeze, unsqueeze, permute, concatenate are tested in tensor_test.cc
 */

#include <gtest/gtest.h>
#include "tensor.h"
#include <cmath>

class TensorShapeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};
// ============================================
// Flatten Tests
// ============================================

TEST_F(TensorShapeTest, Flatten2D) {
    Tensor<float, 2> A({3, 4});
    for (size_t i = 0; i < 12; ++i) {
        A.data()[i] = static_cast<float>(i * 2);
    }
    
    auto B = A.flatten();
    EXPECT_EQ(B.dims()[0], 12);
    
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(B.data()[i], static_cast<float>(i * 2));
    }
}

TEST_F(TensorShapeTest, Flatten3D) {
    Tensor<float, 3> A({2, 3, 4});
    size_t val = 0;
    for (size_t i = 0; i < 24; ++i) {
        A.data()[i] = static_cast<float>(val++);
    }
    
    auto B = A.flatten();
    EXPECT_EQ(B.dims()[0], 24);
    
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(B.data()[i], static_cast<float>(i));
    }
}

TEST_F(TensorShapeTest, Flatten1D) {
    Tensor<float, 1> A({10});
    for (size_t i = 0; i < 10; ++i) {
        A.data()[i] = static_cast<float>(i);
    }
    
    auto B = A.flatten();
    EXPECT_EQ(B.dims()[0], 10);
    
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(B.data()[i], static_cast<float>(i));
    }
}

// ============================================
// Repeat Tests
// ============================================

TEST_F(TensorShapeTest, Repeat2D) {
    Tensor<float, 2> A({2, 3});
    float val = 1.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A[{i, j}] = val++;
        }
    }
    
    auto B = A.repeat({2, 2});
    auto dims = B.dims();
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(dims[1], 6);
    
    // Check pattern is repeated
    EXPECT_FLOAT_EQ(B[{0, 0}], 1.0f);
    EXPECT_FLOAT_EQ(B[{0, 3}], 1.0f);  // Repeated in column
    EXPECT_FLOAT_EQ(B[{2, 0}], 1.0f);  // Repeated in row
}

TEST_F(TensorShapeTest, Repeat1D) {
    Tensor<float, 1> A({3});
    A[{0}] = 1.0f;
    A[{1}] = 2.0f;
    A[{2}] = 3.0f;
    
    auto B = A.repeat({3});
    EXPECT_EQ(B.dims()[0], 9);
    
    EXPECT_FLOAT_EQ(B[{0}], 1.0f);
    EXPECT_FLOAT_EQ(B[{1}], 2.0f);
    EXPECT_FLOAT_EQ(B[{2}], 3.0f);
    EXPECT_FLOAT_EQ(B[{3}], 1.0f);  // Repeated
    EXPECT_FLOAT_EQ(B[{4}], 2.0f);
    EXPECT_FLOAT_EQ(B[{5}], 3.0f);
}

TEST_F(TensorShapeTest, RepeatNoChange) {
    Tensor<float, 2> A({2, 3});
    for (size_t i = 0; i < 6; ++i) {
        A.data()[i] = static_cast<float>(i);
    }
    
    auto B = A.repeat({1, 1});
    auto dims = B.dims();
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 3);
    
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(B.data()[i], A.data()[i]);
    }
}

// ============================================
// Tile Tests
// ============================================

TEST_F(TensorShapeTest, Tile2D) {
    Tensor<float, 2> A({2, 2});
    A[{0, 0}] = 1.0f;
    A[{0, 1}] = 2.0f;
    A[{1, 0}] = 3.0f;
    A[{1, 1}] = 4.0f;
    
    auto B = A.tile({2, 3});
    auto dims = B.dims();
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(dims[1], 6);
    
    // Check tiling pattern
    EXPECT_FLOAT_EQ(B[{0, 0}], 1.0f);
    EXPECT_FLOAT_EQ(B[{0, 2}], 1.0f);
    EXPECT_FLOAT_EQ(B[{2, 0}], 1.0f);
}

TEST_F(TensorShapeTest, Tile1D) {
    Tensor<float, 1> A({2});
    A[{0}] = 5.0f;
    A[{1}] = 10.0f;
    
    auto B = A.tile({4});
    EXPECT_EQ(B.dims()[0], 8);
    
    EXPECT_FLOAT_EQ(B[{0}], 5.0f);
    EXPECT_FLOAT_EQ(B[{1}], 10.0f);
    EXPECT_FLOAT_EQ(B[{2}], 5.0f);
    EXPECT_FLOAT_EQ(B[{3}], 10.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
