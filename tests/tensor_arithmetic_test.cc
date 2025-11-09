/**
 * @file tensor_arithmetic_test.cc
 * @brief Unit tests for tensor arithmetic operations
 */

#include "tensor.h"
#include <gtest/gtest.h>

class TensorArithmeticTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorArithmeticTest, AddTensorToTensor1D) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape);
    Tensor<float, 1> b(shape);
    
    for (size_t i = 0; i < 5; ++i) {
        a.data()[i] = static_cast<float>(i);
        b.data()[i] = static_cast<float>(i * 2);
    }
    
    auto result_var = a + b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result_var)));
    auto& c = std::get<Tensor<float, 1>>(result_var);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], static_cast<float>(i * 3));
    }
}

TEST_F(TensorArithmeticTest, AddScalarToTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>(i);
    }
    
    auto c = a + 5.0f;
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], static_cast<float>(i) + 5.0f);
    }
}

TEST_F(TensorArithmeticTest, SubtractTensorFromTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>(i * 3);
        b.data()[i] = static_cast<float>(i);
    }
    
    auto result_var = a - b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_var)));
    auto& c = std::get<Tensor<float, 2>>(result_var);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], static_cast<float>(i * 2));
    }
}

TEST_F(TensorArithmeticTest, MultiplyTensorByTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
        b.data()[i] = static_cast<float>(i + 2);
    }
    
    auto result_var = a * b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_var)));
    auto& c = std::get<Tensor<float, 2>>(result_var);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], static_cast<float>((i + 1) * (i + 2)));
    }
}

TEST_F(TensorArithmeticTest, MultiplyTensorByScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>(i);
    }
    
    auto c = a * 3.0f;
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], static_cast<float>(i * 3));
    }
}

TEST_F(TensorArithmeticTest, DivideTensorByTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>((i + 1) * 6);
        b.data()[i] = static_cast<float>(i + 2);
    }
    
    auto result_var = a / b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_var)));
    auto& c = std::get<Tensor<float, 2>>(result_var);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], static_cast<float>((i + 1) * 6) / static_cast<float>(i + 2));
    }
}

TEST_F(TensorArithmeticTest, AddEqualsTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>(i);
        b.data()[i] = static_cast<float>(i * 2);
    }
    
    a += b;
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a.data()[i], static_cast<float>(i * 3));
    }
}

TEST_F(TensorArithmeticTest, AddEqualsScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>(i);
    }
    
    a += 10.0f;
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a.data()[i], static_cast<float>(i) + 10.0f);
    }
}

TEST_F(TensorArithmeticTest, UnaryNegation) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>(i);
    }
    
    auto b = -a;
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(b.data()[i], -static_cast<float>(i));
    }
}
