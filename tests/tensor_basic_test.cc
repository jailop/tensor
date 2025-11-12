#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

using namespace tensor;

class TensorBasicTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorBasicTest, Constructor1DInitializesShape) {
    Tensor<float, 1> tensor({10});
    ASSERT_EQ(tensor.dims()[0], 10);
}

TEST_F(TensorBasicTest, Constructor2DInitializesShape) {
    Tensor<float, 2> tensor({3, 4});
    ASSERT_EQ(tensor.dims()[0], 3);
    ASSERT_EQ(tensor.dims()[1], 4);
}

TEST_F(TensorBasicTest, Constructor3DInitializesShape) {
    Tensor<float, 3> tensor({2, 3, 4});
    ASSERT_EQ(tensor.dims()[0], 2);
    ASSERT_EQ(tensor.dims()[1], 3);
    ASSERT_EQ(tensor.dims()[2], 4);
}

TEST_F(TensorBasicTest, FillMethod1DSetsAllElements) {
    Tensor<float, 1> tensor({10});
    tensor.fill(5.0f);
    
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_FLOAT_EQ(tensor[{i}], 5.0f);
    }
}

TEST_F(TensorBasicTest, FillMethod2DSetsAllElements) {
    Tensor<float, 2> tensor({3, 4});
    tensor.fill(2.5f);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            ASSERT_FLOAT_EQ((tensor[{i, j}]), 2.5f);
        }
    }
}

TEST_F(TensorBasicTest, SetAndGet2DElement) {
    Tensor<float, 2> tensor({3, 4});
    tensor[{1, 2}] = 7.5f;
    ASSERT_FLOAT_EQ((tensor[{1, 2}]), 7.5f);
}

TEST_F(TensorBasicTest, CopyConstructor) {
    Tensor<float, 2> tensor({3, 3});
    tensor.fill(1.0f);
    
    Tensor<float, 2> copy = tensor;
    
    ASSERT_EQ(copy.shape()[0], tensor.shape()[0]);
    ASSERT_EQ(copy.shape()[1], tensor.shape()[1]);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            ASSERT_FLOAT_EQ((copy[{i, j}]), (tensor[{i, j}]));
        }
    }
}
