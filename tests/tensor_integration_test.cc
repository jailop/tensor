#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, NeuralNetworkForwardPass) {
    // Simulate: y = softmax(W @ x + b)

TEST_F(TensorTest, BackpropagationExample) {
    // Simulate a simple forward and backward pass

TEST_F(TensorTest, NeuralNetworkActivationSequence) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, MatmulBasic) {
    TensorIndices<2> shape_a = {2, 3};

