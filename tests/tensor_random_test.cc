#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, UniformDistribution) {
    TensorIndices<1> shape = {100};

TEST_F(TensorTest, BernoulliDistribution) {
    TensorIndices<1> shape = {100};

