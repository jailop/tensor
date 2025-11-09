#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, LossMSE) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, LossL1) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, LossBCE) {
    TensorIndices<1> shape = {4};

TEST_F(TensorTest, MSELoss) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, MSELossGradient) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, BinaryCrossentropyLoss) {
    TensorIndices<1> shape = {4};

