#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, Constructor1DInitializesShape) {
    TensorIndices<1> shape = {10};

TEST_F(TensorTest, Constructor2DInitializesShape) {
    TensorIndices<2> shape = {3, 4};

TEST_F(TensorTest, Constructor3DInitializesShape) {
    TensorIndices<3> shape = {2, 3, 4};

TEST_F(TensorTest, Constructor4DInitializesShape) {
    TensorIndices<4> shape = {2, 3, 4, 5};

TEST_F(TensorTest, SetAndGet1DElement) {
    TensorIndices<1> shape = {10};

TEST_F(TensorTest, SetAndGet2DElement) {
    TensorIndices<2> shape = {3, 4};

TEST_F(TensorTest, SetAndGet3DElement) {
    TensorIndices<3> shape = {2, 3, 4};

TEST_F(TensorTest, SetAndGetMultiple2DElements) {
    TensorIndices<2> shape = {3, 3};

TEST_F(TensorTest, SetAndGetMultiple3DElements) {
    TensorIndices<3> shape = {2, 2, 2};

TEST_F(TensorTest, CopyConstructor1DCreatesIndependentCopy) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, CopyConstructor2DCreatesIndependentCopy) {
    TensorIndices<2> shape = {3, 3};

TEST_F(TensorTest, CopyConstructorCopiesAllElements) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, DifferentShapeSizes) {
    TensorIndices<1> shape1 = {100};

TEST_F(TensorTest, IntegerTypeWorks) {
    TensorIndices<2> shape = {3, 3};

TEST_F(TensorTest, DoubleTypeWorks) {
    TensorIndices<2> shape = {3, 3};

TEST_F(TensorTest, ConstAccessorWorks) {
    TensorIndices<2> shape = {3, 3};

TEST_F(TensorTest, BoundaryElements) {
    TensorIndices<3> shape = {3, 4, 5};

TEST_F(TensorTest, FillMethod1DSetsAllElements) {
    TensorIndices<1> shape = {10};

TEST_F(TensorTest, FillMethod2DSetsAllElements) {
    TensorIndices<2> shape = {3, 4};

TEST_F(TensorTest, FillMethod3DSetsAllElements) {
    TensorIndices<3> shape = {2, 2, 2};

TEST_F(TensorTest, FillWithIntegerType) {
    TensorIndices<2> shape = {2, 3};

