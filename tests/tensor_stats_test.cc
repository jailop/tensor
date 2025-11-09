#include "tensor.h"
#include <gtest/gtest.h>

class TensorStatsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorStatsTest, MeanCalculation) {
    Tensor<float, 1> x({5});
    x[{0}] = 1.0f; x[{1}] = 2.0f; x[{2}] = 3.0f; x[{3}] = 4.0f; x[{4}] = 5.0f;
    
    float mean = x.mean();
    EXPECT_FLOAT_EQ(mean, 3.0f);
}
