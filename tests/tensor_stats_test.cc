#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, SumReduction) {
    TensorIndices<2> shape = {2, 3};

TEST_F(TensorTest, MeanCalculation) {
    TensorIndices<2> shape = {2, 2};

TEST_F(TensorTest, VarianceAndStd) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, MinMax) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, Median) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, Quantile) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, QuantileInvalidRange) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, PearsonCorrelation) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, SpearmanCorrelation) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, Covariance) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, CorrelationDimensionMismatch) {
    TensorIndices<1> shape1 = {5};

TEST_F(TensorTest, CovarianceDimensionMismatch) {
    TensorIndices<1> shape1 = {5};

TEST_F(TensorTest, Normalize) {
    TensorIndices<1> shape = {5};

TEST_F(TensorTest, Standardize) {
    TensorIndices<1> shape = {5};

