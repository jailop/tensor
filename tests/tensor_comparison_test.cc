#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, ComparisonLessThan) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, ComparisonGreaterThan) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, ComparisonLessEqual) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, ComparisonGreaterEqual) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, ComparisonEquality) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, ComparisonInequality) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, ComparisonGreaterThanScalar) {
    TensorIndices<1> shape = {5};

