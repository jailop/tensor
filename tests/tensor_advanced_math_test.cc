#include "tensor.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace tensor;

class TensorAdvancedMathTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorAdvancedMathTest, SignFunction) {
    Tensor<float, 1> x({5});
    x[{0}] = -2.5f;
    x[{1}] = -0.5f;
    x[{2}] = 0.0f;
    x[{3}] = 0.5f;
    x[{4}] = 2.5f;
    
    auto result = x.sign();
    
    EXPECT_FLOAT_EQ(result[{0}], -1.0f);
    EXPECT_FLOAT_EQ(result[{1}], -1.0f);
    EXPECT_FLOAT_EQ(result[{2}], 0.0f);
    EXPECT_FLOAT_EQ(result[{3}], 1.0f);
    EXPECT_FLOAT_EQ(result[{4}], 1.0f);
}

TEST_F(TensorAdvancedMathTest, RoundFunction) {
    Tensor<float, 1> x({5});
    x[{0}] = 1.2f;
    x[{1}] = 1.5f;
    x[{2}] = 1.8f;
    x[{3}] = -1.5f;
    x[{4}] = -1.2f;
    
    auto result = x.round();
    
    EXPECT_FLOAT_EQ(result[{0}], 1.0f);
    EXPECT_FLOAT_EQ(result[{1}], 2.0f);
    EXPECT_FLOAT_EQ(result[{2}], 2.0f);
    EXPECT_FLOAT_EQ(result[{3}], -2.0f);
    EXPECT_FLOAT_EQ(result[{4}], -1.0f);
}

TEST_F(TensorAdvancedMathTest, ErfFunction) {
    Tensor<float, 1> x({3});
    x[{0}] = 0.0f;
    x[{1}] = 1.0f;
    x[{2}] = -1.0f;
    
    auto result = x.erf();
    
    EXPECT_NEAR(result[{0}], 0.0f, 1e-6);
    EXPECT_NEAR(result[{1}], std::erf(1.0f), 1e-6);
    EXPECT_NEAR(result[{2}], std::erf(-1.0f), 1e-6);
}

TEST_F(TensorAdvancedMathTest, Log1pFunction) {
    Tensor<float, 1> x({3});
    x[{0}] = 0.0f;
    x[{1}] = 0.1f;
    x[{2}] = 1.0f;
    
    auto result = x.log1p();
    
    EXPECT_NEAR(result[{0}], 0.0f, 1e-6);
    EXPECT_NEAR(result[{1}], std::log1p(0.1f), 1e-6);
    EXPECT_NEAR(result[{2}], std::log1p(1.0f), 1e-6);
}

TEST_F(TensorAdvancedMathTest, Expm1Function) {
    Tensor<float, 1> x({3});
    x[{0}] = 0.0f;
    x[{1}] = 0.1f;
    x[{2}] = 1.0f;
    
    auto result = x.expm1();
    
    EXPECT_NEAR(result[{0}], 0.0f, 1e-6);
    EXPECT_NEAR(result[{1}], std::expm1(0.1f), 1e-6);
    EXPECT_NEAR(result[{2}], std::expm1(1.0f), 1e-6);
}

TEST_F(TensorAdvancedMathTest, IsNanFunction) {
    Tensor<float, 1> x({4});
    x[{0}] = 1.0f;
    x[{1}] = std::numeric_limits<float>::quiet_NaN();
    x[{2}] = std::numeric_limits<float>::infinity();
    x[{3}] = -2.0f;
    
    auto result = x.isnan();
    
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 1.0f);
    EXPECT_FLOAT_EQ(result[{2}], 0.0f);
    EXPECT_FLOAT_EQ(result[{3}], 0.0f);
}

TEST_F(TensorAdvancedMathTest, IsInfFunction) {
    Tensor<float, 1> x({5});
    x[{0}] = 1.0f;
    x[{1}] = std::numeric_limits<float>::quiet_NaN();
    x[{2}] = std::numeric_limits<float>::infinity();
    x[{3}] = -std::numeric_limits<float>::infinity();
    x[{4}] = -2.0f;
    
    auto result = x.isinf();
    
    EXPECT_FLOAT_EQ(result[{0}], 0.0f);
    EXPECT_FLOAT_EQ(result[{1}], 0.0f);
    EXPECT_FLOAT_EQ(result[{2}], 1.0f);
    EXPECT_FLOAT_EQ(result[{3}], 1.0f);
    EXPECT_FLOAT_EQ(result[{4}], 0.0f);
}

TEST_F(TensorAdvancedMathTest, IsFiniteFunction) {
    Tensor<float, 1> x({5});
    x[{0}] = 1.0f;
    x[{1}] = std::numeric_limits<float>::quiet_NaN();
    x[{2}] = std::numeric_limits<float>::infinity();
    x[{3}] = -std::numeric_limits<float>::infinity();
    x[{4}] = -2.0f;
    
    auto result = x.isfinite();
    
    EXPECT_FLOAT_EQ(result[{0}], 1.0f);
    EXPECT_FLOAT_EQ(result[{1}], 0.0f);
    EXPECT_FLOAT_EQ(result[{2}], 0.0f);
    EXPECT_FLOAT_EQ(result[{3}], 0.0f);
    EXPECT_FLOAT_EQ(result[{4}], 1.0f);
}

TEST_F(TensorAdvancedMathTest, SignWith2DTensor) {
    Tensor<float, 2> x({2, 3});
    x[{0, 0}] = -1.5f; x[{0, 1}] = 0.0f; x[{0, 2}] = 2.5f;
    x[{1, 0}] = -0.1f; x[{1, 1}] = 0.1f; x[{1, 2}] = 0.0f;
    
    auto result = x.sign();
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), -1.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{0, 2}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), -1.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1, 2}]), 0.0f);
}

TEST_F(TensorAdvancedMathTest, Log1pSmallValues) {
    // Test numerical stability for small values
    Tensor<float, 1> x({3});
    x[{0}] = 1e-10f;
    x[{1}] = 1e-8f;
    x[{2}] = 1e-6f;
    
    auto result = x.log1p();
    
    // For small x, log1p(x) ≈ x (more accurate than log(1+x))
    EXPECT_NEAR(result[{0}], 1e-10f, 1e-15);
    EXPECT_NEAR(result[{1}], 1e-8f, 1e-13);
    EXPECT_NEAR(result[{2}], 1e-6f, 1e-11);
}

TEST_F(TensorAdvancedMathTest, Expm1SmallValues) {
    // Test numerical stability for small values
    Tensor<float, 1> x({3});
    x[{0}] = 1e-10f;
    x[{1}] = 1e-8f;
    x[{2}] = 1e-6f;
    
    auto result = x.expm1();
    
    // For small x, expm1(x) ≈ x (more accurate than exp(x)-1)
    EXPECT_NEAR(result[{0}], 1e-10f, 1e-15);
    EXPECT_NEAR(result[{1}], 1e-8f, 1e-13);
    EXPECT_NEAR(result[{2}], 1e-6f, 1e-11);
}

TEST_F(TensorAdvancedMathTest, ChainedOperations) {
    Tensor<float, 1> x({3});
    x[{0}] = -2.5f;
    x[{1}] = 0.0f;
    x[{2}] = 2.5f;
    
    // Test chaining: abs -> round -> sign
    auto result = x.abs().round().sign();
    
    EXPECT_FLOAT_EQ(result[{0}], 1.0f);
    EXPECT_FLOAT_EQ(result[{1}], 0.0f);
    EXPECT_FLOAT_EQ(result[{2}], 1.0f);
}
