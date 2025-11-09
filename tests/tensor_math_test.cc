#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, ExpFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, LogFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, SqrtFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, PowFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, SinCosFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, SigmoidFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, TanhFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, ReluFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, LeakyReluDerivative) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, AbsFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, CeilFloorFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, ClampFunction) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, ClipValues) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, ClipWithGradient) {
    TensorIndices<1> shape = {3};

TEST_F(TensorTest, SquareFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, ExpDerivative) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, LogDerivative) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, PowDerivative) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, SinCosDerivative) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, SigmoidDerivative) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, SigmoidDerivativeFromInput) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, TanhDerivative) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, ReluDerivative) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, SoftmaxBasic) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, LogSoftmax) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, MathFunctionsOn3DTensor) {
    TensorIndices<3> shape = {2, 2, 2};

TEST_F(TensorTest, ChainedMathOperations) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, MapCustomFunction) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, MapInplaceCustomFunction) {
    TensorIndices<2> shape = {2, 2};

