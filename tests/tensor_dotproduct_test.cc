#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, DotProduct1DSimple) {
    TensorIndices<1> shape = {3};

TEST_F(TensorTest, DotProduct1DOrthogonal) {
    TensorIndices<1> shape = {4};

TEST_F(TensorTest, DotProduct1DZero) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, DotProduct1DDimensionMismatch) {
    TensorIndices<1> shape_a = {3};

TEST_F(TensorTest, DotProduct2DSimple) {
    TensorIndices<2> shape_a = {2, 3};

TEST_F(TensorTest, DotProduct2DIdentityMatrix) {
    TensorIndices<2> shape = {3, 3};

TEST_F(TensorTest, DotProduct2DSquareMatrices) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, DotProduct2DDimensionMismatch) {
    TensorIndices<2> shape_a = {2, 3};

TEST_F(TensorTest, DotProduct2DWithIntegers) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, DotProduct3DWith1D) {
    // Test 3D tensor (2,3,4) dot 1D tensor (4) = 2D tensor (2,3)

TEST_F(TensorTest, DotProduct3DWith2D) {
    // Test 3D tensor (2,3,4) dot 2D tensor (4,5) = 3D tensor (2,3,5)

TEST_F(TensorTest, DotProduct3DContractionMismatch) {
    TensorIndices<3> shape_a = {2, 3, 4};

TEST_F(TensorTest, DotProduct4DWith1D) {
    // Test 4D tensor (2,2,2,3) dot 1D tensor (3) = 3D tensor (2,2,2)

