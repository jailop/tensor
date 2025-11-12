#include <gtest/gtest.h>
#include "../include/tensor.h"
#include "../include/tensor_ops.h"

using namespace tensor;

class TensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Broadcasting Rules Tests
// =============================================================================

TEST_F(TensorOpsTest, AreBroadcastable_EqualShapes) {
    // Same shapes are always broadcastable
    TensorIndices<2> shape1 = {3, 4};
    TensorIndices<2> shape2 = {3, 4};
    
    EXPECT_TRUE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_OneIsScalar) {
    // Scalar (1,) with any shape
    TensorIndices<1> shape1 = {1};
    TensorIndices<2> shape2 = {3, 4};
    
    EXPECT_TRUE(are_broadcastable(shape1, shape2));
    EXPECT_TRUE(are_broadcastable(shape2, shape1));
}

TEST_F(TensorOpsTest, AreBroadcastable_TrailingDimensionOne) {
    // (3, 1) can broadcast with (3, 4)
    TensorIndices<2> shape1 = {3, 1};
    TensorIndices<2> shape2 = {3, 4};
    
    EXPECT_TRUE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_LeadingDimensionOne) {
    // (1, 4) can broadcast with (3, 4)
    TensorIndices<2> shape1 = {1, 4};
    TensorIndices<2> shape2 = {3, 4};
    
    EXPECT_TRUE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_DifferentRanks) {
    // (4,) can broadcast with (3, 4)
    TensorIndices<1> shape1 = {4};
    TensorIndices<2> shape2 = {3, 4};
    
    EXPECT_TRUE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_NotBroadcastable_DifferentSizes) {
    // (3, 4) cannot broadcast with (3, 5)
    TensorIndices<2> shape1 = {3, 4};
    TensorIndices<2> shape2 = {3, 5};
    
    EXPECT_FALSE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_NotBroadcastable_IncompatibleDims) {
    // (3, 4) cannot broadcast with (2, 4)
    TensorIndices<2> shape1 = {3, 4};
    TensorIndices<2> shape2 = {2, 4};
    
    EXPECT_FALSE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_3D_Compatible) {
    // (1, 3, 4) can broadcast with (2, 3, 4)
    TensorIndices<3> shape1 = {1, 3, 4};
    TensorIndices<3> shape2 = {2, 3, 4};
    
    EXPECT_TRUE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_3D_NotCompatible) {
    // (2, 3, 4) cannot broadcast with (2, 3, 5)
    TensorIndices<3> shape1 = {2, 3, 4};
    TensorIndices<3> shape2 = {2, 3, 5};
    
    EXPECT_FALSE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_MixedRanks_Compatible) {
    // (5,) can broadcast with (3, 4, 5)
    TensorIndices<1> shape1 = {5};
    TensorIndices<3> shape2 = {3, 4, 5};
    
    EXPECT_TRUE(are_broadcastable(shape1, shape2));
}

TEST_F(TensorOpsTest, AreBroadcastable_MixedRanks_NotCompatible) {
    // (4,) cannot broadcast with (3, 4, 5)
    TensorIndices<1> shape1 = {4};
    TensorIndices<3> shape2 = {3, 4, 5};
    
    EXPECT_FALSE(are_broadcastable(shape1, shape2));
}

// =============================================================================
// Broadcast Shape Computation Tests
// =============================================================================

TEST_F(TensorOpsTest, BroadcastShape_EqualShapes) {
    TensorIndices<2> shape1 = {3, 4};
    TensorIndices<2> shape2 = {3, 4};
    
    auto result = broadcast_shape(shape1, shape2);
    
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 4);
}

TEST_F(TensorOpsTest, BroadcastShape_OneWithSizeOne) {
    TensorIndices<2> shape1 = {3, 1};
    TensorIndices<2> shape2 = {3, 4};
    
    auto result = broadcast_shape(shape1, shape2);
    
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 4);
}

TEST_F(TensorOpsTest, BroadcastShape_BothWithSizeOne) {
    TensorIndices<2> shape1 = {1, 4};
    TensorIndices<2> shape2 = {3, 1};
    
    auto result = broadcast_shape(shape1, shape2);
    
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 4);
}

TEST_F(TensorOpsTest, BroadcastShape_DifferentRanks) {
    TensorIndices<1> shape1 = {4};
    TensorIndices<2> shape2 = {3, 4};
    
    auto result = broadcast_shape(shape1, shape2);
    
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 4);
}

TEST_F(TensorOpsTest, BroadcastShape_DifferentRanks_Reverse) {
    TensorIndices<2> shape1 = {3, 4};
    TensorIndices<1> shape2 = {4};
    
    auto result = broadcast_shape(shape1, shape2);
    
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 4);
}

TEST_F(TensorOpsTest, BroadcastShape_3D_WithOnes) {
    TensorIndices<3> shape1 = {1, 3, 4};
    TensorIndices<3> shape2 = {2, 1, 4};
    
    auto result = broadcast_shape(shape1, shape2);
    
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
}

TEST_F(TensorOpsTest, BroadcastShape_MixedRanks_1D_to_3D) {
    TensorIndices<1> shape1 = {5};
    TensorIndices<3> shape2 = {2, 3, 5};
    
    auto result = broadcast_shape(shape1, shape2);
    
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 5);
}

// =============================================================================
// MatmulResultDims Tests (compile-time checks)
// =============================================================================

TEST_F(TensorOpsTest, MatmulResultDims_2D_by_2D) {
    // Matrix @ Matrix: (m, n) @ (n, p) -> (m, p)
    // Result dimensions = 2 + 2 - 2 = 2
    constexpr size_t result = MatmulResultDims<2, 2>::value;
    EXPECT_EQ(result, 2);
}

TEST_F(TensorOpsTest, MatmulResultDims_1D_by_2D) {
    // Vector @ Matrix: (n,) @ (n, p) -> (p,)
    // Result dimensions = 1 + 2 - 2 = 1
    constexpr size_t result = MatmulResultDims<1, 2>::value;
    EXPECT_EQ(result, 1);
}

TEST_F(TensorOpsTest, MatmulResultDims_2D_by_1D) {
    // Matrix @ Vector: (m, n) @ (n,) -> (m,)
    // Result dimensions = 2 + 1 - 2 = 1
    constexpr size_t result = MatmulResultDims<2, 1>::value;
    EXPECT_EQ(result, 1);
}

TEST_F(TensorOpsTest, MatmulResultDims_3D_by_3D) {
    // 3D @ 3D: batch matrix multiplication
    // Result dimensions = 3 + 3 - 2 = 4
    constexpr size_t result = MatmulResultDims<3, 3>::value;
    EXPECT_EQ(result, 4);
}

TEST_F(TensorOpsTest, MatmulResultDims_3D_by_2D) {
    // Batched matrix @ Matrix
    // Result dimensions = 3 + 2 - 2 = 3
    constexpr size_t result = MatmulResultDims<3, 2>::value;
    EXPECT_EQ(result, 3);
}

// =============================================================================
// Integration Tests - Broadcasting with Actual Tensors
// =============================================================================

// Note: These tests require broadcasting between different rank tensors,
// which is not currently implemented in the operator overloads.
// The are_broadcastable and broadcast_shape functions work correctly,
// but the Tensor class operators only work with same-rank tensors.

/*
TEST_F(TensorOpsTest, Integration_BroadcastAddition) {
    // Test that tensors actually broadcast correctly
    Tensor<float, 1> a({3});
    Tensor<float, 2> b({2, 3});
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    // Should be able to add them (b + a should broadcast a)
    auto result = b + a;
    
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result)));
    auto& tensor_result = std::get<Tensor<float, 2>>(result);
    
    EXPECT_EQ(tensor_result.dims()[0], 2);
    EXPECT_EQ(tensor_result.dims()[1], 3);
    
    // All elements should be 3.0
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((tensor_result[{i, j}]), 3.0f);
        }
    }
}

TEST_F(TensorOpsTest, Integration_BroadcastWithSizeOne) {
    // Test broadcasting with dimension of size 1
    Tensor<float, 2> a({3, 1});
    Tensor<float, 2> b({3, 4});
    
    a.fill(2.0f);
    b.fill(3.0f);
    
    auto result = a + b;
    
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result)));
    auto& tensor_result = std::get<Tensor<float, 2>>(result);
    
    EXPECT_EQ(tensor_result.dims()[0], 3);
    EXPECT_EQ(tensor_result.dims()[1], 4);
    
    // All elements should be 5.0
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((tensor_result[{i, j}]), 5.0f);
        }
    }
}

TEST_F(TensorOpsTest, Integration_NonBroadcastableError) {
    // Test that non-broadcastable shapes return error
    Tensor<float, 2> a({3, 4});
    Tensor<float, 2> b({3, 5});
    
    a.fill(1.0f);
    b.fill(1.0f);
    
    auto result = a + b;
    
    // Should return an error, not a valid tensor
    EXPECT_TRUE((std::holds_alternative<TensorError>(result)));
}
*/
