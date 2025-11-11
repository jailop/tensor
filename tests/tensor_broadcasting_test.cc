/**
 * @file tensor_broadcasting_test.cc
 * @brief Unit tests for tensor broadcasting enhancements and NumPy compatibility
 */

#include <gtest/gtest.h>
#include "tensor.h"
#include "linalg.h"
#include <variant>
#include <cmath>

using namespace tensor;

// Test fixture for broadcasting tests
class TensorBroadcastingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST_F(TensorBroadcastingTest, BroadcastableShapes) {
    TensorIndices<2> shape1{3, 1};
    TensorIndices<2> shape2{3, 4};
    
    std::string error_msg;
    ASSERT_TRUE(are_broadcastable(shape1, shape2, &error_msg));
}

TEST_F(TensorBroadcastingTest, NonBroadcastableShapes) {
    TensorIndices<2> shape1{3, 5};
    TensorIndices<2> shape2{3, 4};
    
    std::string error_msg;
    ASSERT_FALSE(are_broadcastable(shape1, shape2, &error_msg));
    ASSERT_FALSE(error_msg.empty());
}

TEST_F(TensorBroadcastingTest, ComputeBroadcastShape) {
    TensorIndices<2> shape1{3, 1};
    TensorIndices<2> shape2{1, 4};
    
    auto result = compute_broadcast_shape(shape1, shape2);
    ASSERT_TRUE((std::holds_alternative<TensorIndices<2>>(result)));
    
    auto broadcast_shape = std::get<TensorIndices<2>>(result);
    ASSERT_EQ(broadcast_shape[0], 3);
    ASSERT_EQ(broadcast_shape[1], 4);
}

TEST_F(TensorBroadcastingTest, BroadcastTo1Dto2D) {
    Tensor<float, 1> x({3});
    x[{0}] = 1.0f;
    x[{1}] = 2.0f;
    x[{2}] = 3.0f;
    
    auto result_var = broadcast_to(x, TensorIndices<2>{4, 3});
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_var)));
    
    auto result = std::get<Tensor<float, 2>>(result_var);
    ASSERT_EQ(result.shape()[0], 4);
    ASSERT_EQ(result.shape()[1], 3);
    
    // Each row should be [1, 2, 3]
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ((result[{i, 0}]), 1.0f);
        EXPECT_FLOAT_EQ((result[{i, 1}]), 2.0f);
        EXPECT_FLOAT_EQ((result[{i, 2}]), 3.0f);
    }
}

TEST_F(TensorBroadcastingTest, BroadcastTo2Dto3D) {
    Tensor<float, 2> x({2, 3});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            x[{i, j}] = static_cast<float>(i * 3 + j);
        }
    }
    
    auto result_var = broadcast_to(x, TensorIndices<3>{4, 2, 3});
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 3>>( result_var)));
    
    auto result = std::get<Tensor<float, 3>>(result_var);
    ASSERT_EQ(result.shape()[0], 4);
    ASSERT_EQ(result.shape()[1], 2);
    ASSERT_EQ(result.shape()[2], 3);
    
    // Each slice along first dimension should be the same as x
    for (size_t k = 0; k < 4; ++k) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_FLOAT_EQ((result[{k, i, j}]), (x[{i, j}]));
            }
        }
    }
}

TEST_F(TensorBroadcastingTest, BroadcastToInvalidShape) {
    Tensor<float, 1> x({3});
    
    // Try to broadcast (3,) to (4,) - invalid
    auto result_var = broadcast_to(x, TensorIndices<1>{4});
    ASSERT_TRUE((std::holds_alternative<TensorError>(result_var)));
}

TEST_F(TensorBroadcastingTest, BroadcastToMethod) {
    Tensor<float, 1> x({3});
    x[{0}] = 1.0f;
    x[{1}] = 2.0f;
    x[{2}] = 3.0f;
    
    auto result_var = x.broadcast_to<2>({2, 3});
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>( result_var)));
    
    auto result = std::get<Tensor<float, 2>>(result_var);
    ASSERT_EQ(result.shape()[0], 2);
    ASSERT_EQ(result.shape()[1], 3);
}

// ============================================================================
// Type Casting Tests
// ============================================================================

TEST_F(TensorBroadcastingTest, AstypeIntToFloat) {
    Tensor<int, 2> x({2, 3});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            x[{i, j}] = static_cast<int>(i * 3 + j);
        }
    }
    
    auto y = astype<float>(x);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 3);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((y[{i, j}]), static_cast<float>(x[{i, j}]));
        }
    }
}

TEST_F(TensorBroadcastingTest, AstypeFloatToInt) {
    Tensor<float, 2> x({2, 3});
    x.fill(3.7f);
    
    auto y = astype<int>(x);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 3);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ((y[{i, j}]), 3);  // Truncation
        }
    }
}

TEST_F(TensorBroadcastingTest, AstypeMethod) {
    Tensor<int, 1> x({5});
    for (size_t i = 0; i < 5; ++i) {
        x[{i}] = static_cast<int>(i);
    }
    
    auto y = x.astype<double>();
    ASSERT_EQ(y.shape()[0], 5);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(y[{i}], static_cast<double>(i));
    }
}

// ============================================================================
// NumPy Compatibility Tests
// ============================================================================

TEST_F(TensorBroadcastingTest, ZerosFunction) {
    auto x = zeros<float, 2>({3, 4});
    ASSERT_EQ(x.shape()[0], 3);
    ASSERT_EQ(x.shape()[1], 4);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((x[{i, j}]), 0.0f);
        }
    }
}

TEST_F(TensorBroadcastingTest, OnesFunction) {
    auto x = ones<float, 2>({2, 5});
    ASSERT_EQ(x.shape()[0], 2);
    ASSERT_EQ(x.shape()[1], 5);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            EXPECT_FLOAT_EQ((x[{i, j}]), 1.0f);
        }
    }
}

TEST_F(TensorBroadcastingTest, CopyFunction) {
    Tensor<float, 2> x({2, 3});
    x.fill(2.5f);
    
    auto y = copy(x);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 3);
    
    // Verify values
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((y[{i, j}]), 2.5f);
        }
    }
    
    // Verify it's a separate copy
    x[{0, 0}] = 10.0f;
    EXPECT_FLOAT_EQ((y[{0, 0}]), 2.5f);
}

TEST_F(TensorBroadcastingTest, ArangeFunction) {
    auto x = arange<float>(0.0f, 10.0f, 1.0f);
    ASSERT_EQ(x.shape()[0], 10);
    
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ((x[{i}]), static_cast<float>(i));
    }
}

TEST_F(TensorBroadcastingTest, ArangeWithStep) {
    auto x = arange<float>(0.0f, 5.0f, 0.5f);
    ASSERT_EQ(x.shape()[0], 10);
    
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ((x[{i}]), static_cast<float>(i) * 0.5f);
    }
}

TEST_F(TensorBroadcastingTest, LinspaceFunction) {
    auto x = linspace<float>(0.0f, 10.0f, 11);
    ASSERT_EQ(x.shape()[0], 11);
    
    for (size_t i = 0; i < 11; ++i) {
        EXPECT_NEAR(x[{i}], static_cast<float>(i), 1e-5f);
    }
}

TEST_F(TensorBroadcastingTest, LinspaceSinglePoint) {
    auto x = linspace<float>(5.0f, 10.0f, 1);
    ASSERT_EQ(x.shape()[0], 1);
    EXPECT_FLOAT_EQ((x[{0}]), 5.0f);
}

TEST_F(TensorBroadcastingTest, LogspaceFunction) {
    auto x = logspace<float>(0.0f, 2.0f, 3);
    ASSERT_EQ(x.shape()[0], 3);
    
    EXPECT_NEAR(x[{0}], 1.0f, 1e-5f);      // 10^0
    EXPECT_NEAR(x[{1}], 10.0f, 1e-5f);     // 10^1
    EXPECT_NEAR(x[{2}], 100.0f, 1e-4f);    // 10^2
}

TEST_F(TensorBroadcastingTest, EyeFunction) {
    auto x = eye<float>(3);
    ASSERT_EQ(x.shape()[0], 3);
    ASSERT_EQ(x.shape()[1], 3);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_FLOAT_EQ((x[{i, j}]), 1.0f);
            } else {
                EXPECT_FLOAT_EQ((x[{i, j}]), 0.0f);
            }
        }
    }
}

TEST_F(TensorBroadcastingTest, ReshapeToFunction) {
    Tensor<float, 2> x({2, 6});
    for (size_t i = 0; i < 12; ++i) {
        x.data_ptr()[i] = static_cast<float>(i);
    }
    
    auto result_var = reshape_to(x, TensorIndices<3>{2, 3, 2});
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 3>>( result_var)));
    
    auto result = std::get<Tensor<float, 3>>(result_var);
    ASSERT_EQ(result.shape()[0], 2);
    ASSERT_EQ(result.shape()[1], 3);
    ASSERT_EQ(result.shape()[2], 2);
    
    // Verify data is preserved
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(result.data_ptr()[i], static_cast<float>(i));
    }
}

TEST_F(TensorBroadcastingTest, ReshapeToInvalidSize) {
    Tensor<float, 2> x({2, 6});
    
    auto result_var = reshape_to(x, TensorIndices<2>{3, 5});  // 15 != 12
    ASSERT_TRUE((std::holds_alternative<TensorError>(result_var)));
}

// ============================================================================
// Complex Broadcasting Scenarios
// ============================================================================

TEST_F(TensorBroadcastingTest, BroadcastScalarLike) {
    Tensor<float, 1> x({1});
    x[{0}] = 5.0f;
    
    auto result_var = broadcast_to(x, TensorIndices<2>{3, 4});
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>( result_var)));
    
    auto result = std::get<Tensor<float, 2>>(result_var);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), 5.0f);
        }
    }
}

TEST_F(TensorBroadcastingTest, BroadcastColumn) {
    Tensor<float, 2> x({3, 1});
    x[{0, 0}] = 1.0f;
    x[{1, 0}] = 2.0f;
    x[{2, 0}] = 3.0f;
    
    auto result_var = broadcast_to(x, TensorIndices<2>{3, 4});
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>( result_var)));
    
    auto result = std::get<Tensor<float, 2>>(result_var);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), static_cast<float>(i + 1));
        }
    }
}

TEST_F(TensorBroadcastingTest, BroadcastRow) {
    Tensor<float, 2> x({1, 4});
    for (size_t j = 0; j < 4; ++j) {
        x[{0, j}] = static_cast<float>(j + 1);
    }
    
    auto result_var = broadcast_to(x, TensorIndices<2>{3, 4});
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>( result_var)));
    
    auto result = std::get<Tensor<float, 2>>(result_var);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), static_cast<float>(j + 1));
        }
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(TensorBroadcastingTest, BroadcastAndCast) {
    Tensor<int, 1> x({3});
    x[{0}] = 1;
    x[{1}] = 2;
    x[{2}] = 3;
    
    // First broadcast
    auto broadcast_var = broadcast_to(x, TensorIndices<2>{2, 3});
    ASSERT_TRUE((std::holds_alternative<Tensor<int, 2>>( broadcast_var)));
    auto broadcast = std::get<Tensor<int, 2>>(broadcast_var);
    
    // Then cast
    auto result = astype<float>(broadcast);
    ASSERT_EQ(result.shape()[0], 2);
    ASSERT_EQ(result.shape()[1], 3);
    
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ((result[{i, 0}]), 1.0f);
        EXPECT_FLOAT_EQ((result[{i, 1}]), 2.0f);
        EXPECT_FLOAT_EQ((result[{i, 2}]), 3.0f);
    }
}

TEST_F(TensorBroadcastingTest, CombineWithArange) {
    auto x = arange<float>(1.0f, 4.0f, 1.0f);  // [1, 2, 3]
    
    auto broadcast_var = broadcast_to(x, TensorIndices<2>{4, 3});
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>( broadcast_var)));
    
    auto result = std::get<Tensor<float, 2>>(broadcast_var);
    
    // Each row should be [1, 2, 3]
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ((result[{i, 0}]), 1.0f);
        EXPECT_FLOAT_EQ((result[{i, 1}]), 2.0f);
        EXPECT_FLOAT_EQ((result[{i, 2}]), 3.0f);
    }
}

