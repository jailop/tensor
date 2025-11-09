#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, AutogradBasicAddition) {
    TensorIndices<1> shape = {1};  // Scalar tensor

TEST_F(TensorTest, AutogradMultiplication) {
    TensorIndices<1> shape = {1};  // Scalar tensor

TEST_F(TensorTest, AutogradSigmoid) {
    TensorIndices<1> shape = {1};  // Scalar tensor

TEST_F(TensorTest, AutogradReLU) {
    TensorIndices<1> shape = {1};  // Scalar tensor

TEST_F(TensorTest, AutogradReLUNegative) {
    TensorIndices<1> shape = {1};

TEST_F(TensorTest, AutogradWithGradientArg) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, AutogradChainedOperations) {
    TensorIndices<1> shape = {1};  // Scalar

TEST_F(TensorTest, AutogradCompositeFunction) {
    // Test: f(x) = sigmoid(x * x) for scalar

TEST_F(TensorTest, AutogradZeroGrad) {
    TensorIndices<1> shape = {1};  // Scalar

TEST_F(TensorTest, AutogradDetach) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, AutogradNoGradTracking) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, AutogradSumReduction) {
    // Test backward through a reduction operation

TEST_F(TensorTest, AutogradLeafTensor) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, AutogradMultipleBackward) {
    TensorIndices<1> shape = {1};

TEST_F(TensorTest, ChainRule) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, SoftmaxAutograd) {
    TensorIndices<1> shape = {3};

TEST_F(TensorTest, MatmulAutograd) {
    TensorIndices<2> shape_w = {3, 2};

TEST_F(TensorTest, MatmulChained) {
    // Test: y = (A @ B) @ C

TEST_F(TensorTest, TransposeWithAutograd) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, PermuteWithAutograd) {
    TensorIndices<3> shape = {2, 3, 4};

TEST_F(TensorTest, ReshapeWithAutograd) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, ConcatenateWithAutograd) {
    TensorIndices<1> shape1 = {3};

