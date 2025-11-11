#include "tensor.h"
#include <gtest/gtest.h>

using namespace tensor;

class TensorMathTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorMathTest, ExpFunction) {
    Tensor<float, 1> x({3});
    x[{0}] = 0.0f; x[{1}] = 1.0f; x[{2}] = 2.0f;
    
    auto result = x.exp();
    
    EXPECT_NEAR(result[{0}], 1.0f, 1e-5);
    EXPECT_NEAR(result[{1}], std::exp(1.0f), 1e-5);
    EXPECT_NEAR(result[{2}], std::exp(2.0f), 1e-5);
}

TEST_F(TensorMathTest, LogFunction) {
    Tensor<float, 1> x({3});
    x[{0}] = 1.0f; x[{1}] = std::exp(1.0f); x[{2}] = std::exp(2.0f);
    
    auto result = x.log();
    
    EXPECT_NEAR(result[{0}], 0.0f, 1e-5);
    EXPECT_NEAR(result[{1}], 1.0f, 1e-5);
    EXPECT_NEAR(result[{2}], 2.0f, 1e-5);
}

TEST_F(TensorMathTest, SqrtFunction) {
    Tensor<float, 1> x({3});
    x[{0}] = 1.0f; x[{1}] = 4.0f; x[{2}] = 9.0f;
    
    auto result = x.sqrt();
    
    EXPECT_FLOAT_EQ(result[{0}], 1.0f);
    EXPECT_FLOAT_EQ(result[{1}], 2.0f);
    EXPECT_FLOAT_EQ(result[{2}], 3.0f);
}

TEST_F(TensorMathTest, PowFunction) {
    Tensor<float, 1> x({3});
    x[{0}] = 2.0f; x[{1}] = 3.0f; x[{2}] = 4.0f;
    
    auto result = x.pow(2.0f);
    
    EXPECT_FLOAT_EQ(result[{0}], 4.0f);
    EXPECT_FLOAT_EQ(result[{1}], 9.0f);
    EXPECT_FLOAT_EQ(result[{2}], 16.0f);
}

TEST_F(TensorMathTest, SigmoidFunction) {
    Tensor<float, 1> x({3});
    x[{0}] = 0.0f; x[{1}] = 1.0f; x[{2}] = -1.0f;
    
    auto result = x.sigmoid();
    
    EXPECT_FLOAT_EQ(result[{0}], 0.5f);
    EXPECT_NEAR(result[{1}], 1.0f / (1.0f + std::exp(-1.0f)), 1e-5);
    EXPECT_NEAR(result[{2}], 1.0f / (1.0f + std::exp(1.0f)), 1e-5);
}

TEST_F(TensorMathTest, ReluFunction) {
    Tensor<float, 1> x({4});
    x[{0}] = -2.0f; x[{1}] = -0.5f; x[{2}] = 0.0f; x[{3}] = 2.0f;
    
    auto result = x.relu();
    
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 0.0f);
    EXPECT_FLOAT_EQ(result[{2}], 0.0f);
    EXPECT_FLOAT_EQ(result[{3}], 2.0f);
}
