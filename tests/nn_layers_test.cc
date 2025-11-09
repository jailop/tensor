/**
 * @file nn_layers_test.cc
 * @brief Unit tests for neural network layers
 */

#include "nn_layers.h"
#include "tensor_types.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace tensor4d;
using namespace tensor4d::nn;

class NNLayersTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper function to check if two floats are approximately equal
    bool approx_equal(float a, float b, float epsilon = 1e-5f) {
        return std::abs(a - b) < epsilon;
    }
};

// ============================================================================
// Linear Layer Tests
// ============================================================================

TEST_F(NNLayersTest, LinearLayerConstructor) {
    Linearf linear(3, 5, true);
    
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2);  // weights and bias
    
    auto& weights = linear.weights();
    auto& bias = linear.bias();
    
    EXPECT_EQ(weights.shape()[0], 5);  // out_features
    EXPECT_EQ(weights.shape()[1], 3);  // in_features
    EXPECT_EQ(bias.shape()[0], 1);
    EXPECT_EQ(bias.shape()[1], 5);
}

TEST_F(NNLayersTest, LinearLayerForwardShape) {
    Linearf linear(4, 6, true);
    
    // Batch of 2 samples, 4 features each
    Matrixf input({2, 4});
    input.fill(1.0f);
    
    auto output = linear.forward(input);
    
    EXPECT_EQ(output.shape()[0], 2);  // batch size preserved
    EXPECT_EQ(output.shape()[1], 6);  // out_features
}

TEST_F(NNLayersTest, LinearLayerNoBias) {
    Linearf linear(3, 5, false);
    
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 1);  // only weights, no bias
}

TEST_F(NNLayersTest, LinearLayerBackwardShape) {
    Linearf linear(4, 6, true);
    
    Matrixf input({2, 4});
    input.fill(1.0f);
    
    auto output = linear.forward(input);
    
    Matrixf grad_output({2, 6});
    grad_output.fill(1.0f);
    
    auto grad_input = linear.backward(grad_output);
    
    EXPECT_EQ(grad_input.shape()[0], 2);
    EXPECT_EQ(grad_input.shape()[1], 4);
}

// ============================================================================
// ReLU Tests
// ============================================================================

TEST_F(NNLayersTest, ReLUForwardPositive) {
    ReLUf relu;
    
    Matrixf input({1, 3});
    input[{0, 0}] = 1.0f;
    input[{0, 1}] = 2.0f;
    input[{0, 2}] = 3.0f;
    
    auto output = relu.forward(input);
    
    EXPECT_FLOAT_EQ(((output[{0, 0}])), 1.0f);
    EXPECT_FLOAT_EQ(((output[{0, 1}])), 2.0f);
    EXPECT_FLOAT_EQ(((output[{0, 2}])), 3.0f);
}

TEST_F(NNLayersTest, ReLUForwardNegative) {
    ReLUf relu;
    
    Matrixf input({1, 3});
    input[{0, 0}] = -1.0f;
    input[{0, 1}] = -2.0f;
    input[{0, 2}] = -3.0f;
    
    auto output = relu.forward(input);
    
    EXPECT_FLOAT_EQ(((output[{0, 0}])), 0.0f);
    EXPECT_FLOAT_EQ(((output[{0, 1}])), 0.0f);
    EXPECT_FLOAT_EQ(((output[{0, 2}])), 0.0f);
}

TEST_F(NNLayersTest, ReLUForwardMixed) {
    ReLUf relu;
    
    Matrixf input({1, 4});
    input[{0, 0}] = -1.0f;
    input[{0, 1}] = 0.0f;
    input[{0, 2}] = 1.0f;
    input[{0, 3}] = -0.5f;
    
    auto output = relu.forward(input);
    
    EXPECT_FLOAT_EQ(((output[{0, 0}])), 0.0f);
    EXPECT_FLOAT_EQ(((output[{0, 1}])), 0.0f);
    EXPECT_FLOAT_EQ(((output[{0, 2}])), 1.0f);
    EXPECT_FLOAT_EQ(((output[{0, 3}])), 0.0f);
}

TEST_F(NNLayersTest, ReLUBackward) {
    ReLUf relu;
    
    Matrixf input({1, 3});
    input[{0, 0}] = -1.0f;
    input[{0, 1}] = 2.0f;
    input[{0, 2}] = 0.0f;
    
    relu.forward(input);
    
    Matrixf grad_output({1, 3});
    grad_output.fill(1.0f);
    
    auto grad_input = relu.backward(grad_output);
    
    EXPECT_FLOAT_EQ(((grad_input[{0, 0}])), 0.0f);  // negative -> 0
    EXPECT_FLOAT_EQ(((grad_input[{0, 1}])), 1.0f);  // positive -> 1
    EXPECT_FLOAT_EQ(((grad_input[{0, 2}])), 0.0f);  // zero -> 0
}

// ============================================================================
// Sigmoid Tests
// ============================================================================

TEST_F(NNLayersTest, SigmoidForwardZero) {
    Sigmoidf sigmoid;
    
    Matrixf input({1, 1});
    input[{0, 0}] = 0.0f;
    
    auto output = sigmoid.forward(input);
    
    EXPECT_FLOAT_EQ(((output[{0, 0}])), 0.5f);
}

TEST_F(NNLayersTest, SigmoidForwardRange) {
    Sigmoidf sigmoid;
    
    Matrixf input({1, 3});
    input[{0, 0}] = -10.0f;
    input[{0, 1}] = 0.0f;
    input[{0, 2}] = 10.0f;
    
    auto output = sigmoid.forward(input);
    
    EXPECT_GT(((output[{0, 0}])), 0.0f);
    EXPECT_LT(((output[{0, 0}])), 0.01f);  // Very close to 0
    EXPECT_FLOAT_EQ(((output[{0, 1}])), 0.5f);
    EXPECT_GT(((output[{0, 2}])), 0.99f);  // Very close to 1
    EXPECT_LT(((output[{0, 2}])), 1.0f);
}

TEST_F(NNLayersTest, SigmoidBackward) {
    Sigmoidf sigmoid;
    
    Matrixf input({1, 1});
    input[{0, 0}] = 0.0f;
    
    auto output = sigmoid.forward(input);
    
    Matrixf grad_output({1, 1});
    grad_output[{0, 0}] = 1.0f;
    
    auto grad_input = sigmoid.backward(grad_output);
    
    // At x=0, sigmoid(0)=0.5, derivative = 0.5 * (1 - 0.5) = 0.25
    EXPECT_TRUE(approx_equal((grad_input[{0, 0}]), 0.25f));
}

// ============================================================================
// Tanh Tests
// ============================================================================

TEST_F(NNLayersTest, TanhForwardZero) {
    Tanhf tanh;
    
    Matrixf input({1, 1});
    input[{0, 0}] = 0.0f;
    
    auto output = tanh.forward(input);
    
    EXPECT_FLOAT_EQ(((output[{0, 0}])), 0.0f);
}

TEST_F(NNLayersTest, TanhForwardRange) {
    Tanhf tanh;
    
    Matrixf input({1, 3});
    input[{0, 0}] = -5.0f;
    input[{0, 1}] = 0.0f;
    input[{0, 2}] = 5.0f;
    
    auto output = tanh.forward(input);
    
    EXPECT_GT(((output[{0, 0}])), -1.0f);
    EXPECT_LT(((output[{0, 0}])), -0.99f);  // Very close to -1
    EXPECT_FLOAT_EQ(((output[{0, 1}])), 0.0f);
    EXPECT_GT(((output[{0, 2}])), 0.99f);   // Very close to 1
    EXPECT_LT(((output[{0, 2}])), 1.0f);
}

// ============================================================================
// Softmax Tests
// ============================================================================

TEST_F(NNLayersTest, SoftmaxSumToOne) {
    Softmaxf softmax;
    
    Matrixf input({1, 3});
    input[{0, 0}] = 1.0f;
    input[{0, 1}] = 2.0f;
    input[{0, 2}] = 3.0f;
    
    auto output = softmax.forward(input);
    
    float sum = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        sum += output[{0, i}];
        EXPECT_GT(((output[{0, i}])), 0.0f);
        EXPECT_LT(((output[{0, i}])), 1.0f);
    }
    
    EXPECT_TRUE(approx_equal(sum, 1.0f));
}

TEST_F(NNLayersTest, SoftmaxBatchProcessing) {
    Softmaxf softmax;
    
    Matrixf input({2, 3});
    input[{0, 0}] = 1.0f; input[{0, 1}] = 2.0f; input[{0, 2}] = 3.0f;
    input[{1, 0}] = 3.0f; input[{1, 1}] = 2.0f; input[{1, 2}] = 1.0f;
    
    auto output = softmax.forward(input);
    
    // Each row should sum to 1
    for (size_t i = 0; i < 2; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            sum += output[{i, j}];
        }
        EXPECT_TRUE(approx_equal(sum, 1.0f));
    }
}

TEST_F(NNLayersTest, SoftmaxMonotonic) {
    Softmaxf softmax;
    
    Matrixf input({1, 3});
    input[{0, 0}] = 1.0f;
    input[{0, 1}] = 2.0f;
    input[{0, 2}] = 3.0f;
    
    auto output = softmax.forward(input);
    
    // Larger inputs should give larger probabilities
    EXPECT_LT(((output[{0, 0}])), (output[{0, 1}]));
    EXPECT_LT(((output[{0, 1}])), (output[{0, 2}]));
}

// ============================================================================
// Dropout Tests
// ============================================================================

TEST_F(NNLayersTest, DropoutInferenceMode) {
    Dropoutf dropout(0.5f);
    dropout.train(false);  // Inference mode
    
    Matrixf input({2, 4});
    input.fill(1.0f);
    
    auto output = dropout.forward(input);
    
    // In inference mode, no dropout should occur
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(((output[{i, j}])), 1.0f);
        }
    }
}

TEST_F(NNLayersTest, DropoutTrainingMode) {
    Dropoutf dropout(0.5f);
    dropout.train(true);  // Training mode
    
    Matrixf input({10, 10});
    input.fill(1.0f);
    
    auto output = dropout.forward(input);
    
    // Count non-zero elements
    int non_zero_count = 0;
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            if (output[{i, j}] > 0.0f) {
                non_zero_count++;
            }
        }
    }
    
    // With p=0.5, approximately half should be non-zero
    // Allow some variance: between 30 and 70 out of 100
    EXPECT_GT(non_zero_count, 30);
    EXPECT_LT(non_zero_count, 70);
}

TEST_F(NNLayersTest, DropoutInvertedScaling) {
    Dropoutf dropout(0.5f);
    dropout.train(true);
    
    Matrixf input({100, 1});
    input.fill(1.0f);
    
    auto output = dropout.forward(input);
    
    // Calculate mean of non-zero elements
    float sum = 0.0f;
    int count = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (output[{i, 0}] > 0.0f) {
            sum += output[{i, 0}];
            count++;
        }
    }
    
    if (count > 0) {
        float mean = sum / count;
        // With inverted dropout, non-zero values should be scaled by 1/(1-p) = 2
        EXPECT_TRUE(approx_equal(mean, 2.0f, 0.1f));
    }
}

// ============================================================================
// Batch Normalization Tests
// ============================================================================

TEST_F(NNLayersTest, BatchNormTrainingMode) {
    BatchNorm1df bn(3);
    bn.train(true);
    
    Matrixf input({2, 3});
    input[{0, 0}] = 1.0f; input[{0, 1}] = 2.0f; input[{0, 2}] = 3.0f;
    input[{1, 0}] = 4.0f; input[{1, 1}] = 5.0f; input[{1, 2}] = 6.0f;
    
    auto output = bn.forward(input);
    
    // Each feature should be normalized (mean ≈ 0, std ≈ 1)
    for (size_t j = 0; j < 3; ++j) {
        float mean = (output[{0, j}] + output[{1, j}]) / 2.0f;
        EXPECT_TRUE(approx_equal(mean, 0.0f, 0.01f));
    }
}

TEST_F(NNLayersTest, BatchNormZeroMeanUnitVariance) {
    BatchNorm1df bn(2);
    bn.train(true);
    
    Matrixf input({4, 2});
    // Feature 0: values 1, 2, 3, 4 (mean=2.5, var=1.25)
    input[{0, 0}] = 1.0f; input[{0, 1}] = 10.0f;
    input[{1, 0}] = 2.0f; input[{1, 1}] = 20.0f;
    input[{2, 0}] = 3.0f; input[{2, 1}] = 30.0f;
    input[{3, 0}] = 4.0f; input[{3, 1}] = 40.0f;
    
    auto output = bn.forward(input);
    
    // Compute mean of each feature
    for (size_t j = 0; j < 2; ++j) {
        float mean = 0.0f;
        for (size_t i = 0; i < 4; ++i) {
            mean += output[{i, j}];
        }
        mean /= 4.0f;
        EXPECT_TRUE(approx_equal(mean, 0.0f, 0.01f));
    }
}

// ============================================================================
// Layer Base Class Tests
// ============================================================================

TEST_F(NNLayersTest, LayerTrainingMode) {
    Dropoutf dropout(0.5f);
    
    EXPECT_TRUE(dropout.is_training());
    
    dropout.train(false);
    EXPECT_FALSE(dropout.is_training());
    
    dropout.train(true);
    EXPECT_TRUE(dropout.is_training());
}

TEST_F(NNLayersTest, LayerParametersEmpty) {
    ReLUf relu;
    
    auto params = relu.parameters();
    EXPECT_EQ(params.size(), 0);  // Activation layers have no parameters
}

TEST_F(NNLayersTest, LayerParametersNonEmpty) {
    Linearf linear(3, 5, true);
    
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2);  // weights and bias
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(NNLayersTest, SimpleNetworkForwardPass) {
    Linearf fc1(2, 4, true);
    ReLUf relu;
    Linearf fc2(4, 1, true);
    
    Matrixf input({1, 2});
    input[{0, 0}] = 1.0f;
    input[{0, 1}] = 2.0f;
    
    auto h1 = fc1.forward(input);
    EXPECT_EQ(h1.shape()[0], 1);
    EXPECT_EQ(h1.shape()[1], 4);
    
    auto h2 = relu.forward(h1);
    EXPECT_EQ(h2.shape()[0], 1);
    EXPECT_EQ(h2.shape()[1], 4);
    
    auto output = fc2.forward(h2);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
}

TEST_F(NNLayersTest, NetworkBackwardPass) {
    Linearf fc1(2, 3, true);
    ReLUf relu;
    Linearf fc2(3, 1, true);
    
    Matrixf input({1, 2});
    input.fill(1.0f);
    
    // Forward pass
    auto h1 = fc1.forward(input);
    auto h2 = relu.forward(h1);
    auto output = fc2.forward(h2);
    
    // Backward pass
    Matrixf grad_output({1, 1});
    grad_output.fill(1.0f);
    
    auto grad_h2 = fc2.backward(grad_output);
    EXPECT_EQ(grad_h2.shape()[0], 1);
    EXPECT_EQ(grad_h2.shape()[1], 3);
    
    auto grad_h1 = relu.backward(grad_h2);
    EXPECT_EQ(grad_h1.shape()[0], 1);
    EXPECT_EQ(grad_h1.shape()[1], 3);
    
    auto grad_input = fc1.backward(grad_h1);
    EXPECT_EQ(grad_input.shape()[0], 1);
    EXPECT_EQ(grad_input.shape()[1], 2);
}

TEST_F(NNLayersTest, ClassificationNetwork) {
    Linearf fc1(4, 8, true);
    ReLUf relu;
    Linearf fc2(8, 3, true);
    Softmaxf softmax;
    
    // Batch of 2 samples
    Matrixf input({2, 4});
    input.fill(0.5f);
    
    auto h1 = fc1.forward(input);
    auto h2 = relu.forward(h1);
    auto h3 = fc2.forward(h2);
    auto output = softmax.forward(h3);
    
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    
    // Check that each row sums to 1
    for (size_t i = 0; i < 2; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            sum += output[{i, j}];
        }
        EXPECT_TRUE(approx_equal(sum, 1.0f));
    }
}

// ============================================================================
// Softmax Jacobian Tests
// ============================================================================

TEST_F(NNLayersTest, SoftmaxJacobianShape) {
    Tensor<float, 1> softmax_output({5});
    softmax_output[{0}] = 0.1f;
    softmax_output[{1}] = 0.2f;
    softmax_output[{2}] = 0.3f;
    softmax_output[{3}] = 0.25f;
    softmax_output[{4}] = 0.15f;
    
    auto jacobian = softmax_jacobian(softmax_output);
    
    EXPECT_EQ(jacobian.shape()[0], 5);
    EXPECT_EQ(jacobian.shape()[1], 5);
}

TEST_F(NNLayersTest, SoftmaxJacobianDiagonal) {
    Tensor<float, 1> softmax_output({3});
    softmax_output[{0}] = 0.2f;
    softmax_output[{1}] = 0.5f;
    softmax_output[{2}] = 0.3f;
    
    auto jacobian = softmax_jacobian(softmax_output);
    
    // J_ii = s_i * (1 - s_i)
    EXPECT_TRUE(approx_equal(jacobian[{0, 0}], 0.2f * (1.0f - 0.2f), 1e-5f));
    EXPECT_TRUE(approx_equal(jacobian[{1, 1}], 0.5f * (1.0f - 0.5f), 1e-5f));
    EXPECT_TRUE(approx_equal(jacobian[{2, 2}], 0.3f * (1.0f - 0.3f), 1e-5f));
}

TEST_F(NNLayersTest, SoftmaxJacobianOffDiagonal) {
    Tensor<float, 1> softmax_output({3});
    softmax_output[{0}] = 0.2f;
    softmax_output[{1}] = 0.5f;
    softmax_output[{2}] = 0.3f;
    
    auto jacobian = softmax_jacobian(softmax_output);
    
    // J_ij = -s_i * s_j for i != j
    EXPECT_TRUE(approx_equal(jacobian[{0, 1}], -0.2f * 0.5f, 1e-5f));
    EXPECT_TRUE(approx_equal(jacobian[{0, 2}], -0.2f * 0.3f, 1e-5f));
    EXPECT_TRUE(approx_equal(jacobian[{1, 0}], -0.5f * 0.2f, 1e-5f));
    EXPECT_TRUE(approx_equal(jacobian[{1, 2}], -0.5f * 0.3f, 1e-5f));
    EXPECT_TRUE(approx_equal(jacobian[{2, 0}], -0.3f * 0.2f, 1e-5f));
    EXPECT_TRUE(approx_equal(jacobian[{2, 1}], -0.3f * 0.5f, 1e-5f));
}

TEST_F(NNLayersTest, SoftmaxJacobianSymmetry) {
    Tensor<float, 1> softmax_output({4});
    softmax_output[{0}] = 0.1f;
    softmax_output[{1}] = 0.3f;
    softmax_output[{2}] = 0.4f;
    softmax_output[{3}] = 0.2f;
    
    auto jacobian = softmax_jacobian(softmax_output);
    
    // Off-diagonal elements should be symmetric
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i != j) {
                EXPECT_TRUE(approx_equal(jacobian[{i, j}], jacobian[{j, i}], 1e-5f));
            }
        }
    }
}

TEST_F(NNLayersTest, SoftmaxJacobianRowSum) {
    Tensor<float, 1> softmax_output({3});
    softmax_output[{0}] = 0.2f;
    softmax_output[{1}] = 0.5f;
    softmax_output[{2}] = 0.3f;
    
    auto jacobian = softmax_jacobian(softmax_output);
    
    // Each row of the Jacobian should sum to 0
    for (size_t i = 0; i < 3; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            row_sum += jacobian[{i, j}];
        }
        EXPECT_TRUE(approx_equal(row_sum, 0.0f, 1e-5f));
    }
}

TEST_F(NNLayersTest, SoftmaxJacobianBatch) {
    Matrixf softmax_output({2, 3});
    
    // First sample
    softmax_output[{0, 0}] = 0.2f;
    softmax_output[{0, 1}] = 0.5f;
    softmax_output[{0, 2}] = 0.3f;
    
    // Second sample
    softmax_output[{1, 0}] = 0.1f;
    softmax_output[{1, 1}] = 0.6f;
    softmax_output[{1, 2}] = 0.3f;
    
    auto jacobians = softmax_jacobian_batch(softmax_output);
    
    EXPECT_EQ(jacobians.size(), 2);
    EXPECT_EQ(jacobians[0].shape()[0], 3);
    EXPECT_EQ(jacobians[0].shape()[1], 3);
    EXPECT_EQ(jacobians[1].shape()[0], 3);
    EXPECT_EQ(jacobians[1].shape()[1], 3);
    
    // Check first sample Jacobian diagonal
    EXPECT_TRUE(approx_equal(jacobians[0][{0, 0}], 0.2f * (1.0f - 0.2f), 1e-5f));
    EXPECT_TRUE(approx_equal(jacobians[0][{1, 1}], 0.5f * (1.0f - 0.5f), 1e-5f));
    
    // Check second sample Jacobian diagonal
    EXPECT_TRUE(approx_equal(jacobians[1][{0, 0}], 0.1f * (1.0f - 0.1f), 1e-5f));
    EXPECT_TRUE(approx_equal(jacobians[1][{1, 1}], 0.6f * (1.0f - 0.6f), 1e-5f));
}

TEST_F(NNLayersTest, SoftmaxJacobianWithTensorOperations) {
    Matrixf logits({1, 4});
    logits[{0, 0}] = 1.0f;
    logits[{0, 1}] = 2.0f;
    logits[{0, 2}] = 3.0f;
    logits[{0, 3}] = 4.0f;
    
    // Apply softmax
    auto probs = logits.softmax_rows();
    
    // Extract first row as 1D tensor
    Tensor<float, 1> prob_vec({4});
    const float* prob_data = probs.data_ptr();
    std::copy_n(prob_data, 4, prob_vec.data_ptr());
    
    auto jacobian = softmax_jacobian(prob_vec);
    
    // Verify properties
    EXPECT_EQ(jacobian.shape()[0], 4);
    EXPECT_EQ(jacobian.shape()[1], 4);
    
    // Row sums should be zero
    for (size_t i = 0; i < 4; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 4; ++j) {
            row_sum += jacobian[{i, j}];
        }
        EXPECT_TRUE(approx_equal(row_sum, 0.0f, 1e-5f));
    }
}

// ============================================================================
// compute_accuracy Tests
// ============================================================================

TEST_F(NNLayersTest, ComputeAccuracyPerfectPredictions) {
    Matrixf predictions({5, 3});
    predictions[{0, 0}] = 0.9f; predictions[{0, 1}] = 0.05f; predictions[{0, 2}] = 0.05f;
    predictions[{1, 0}] = 0.1f; predictions[{1, 1}] = 0.8f;  predictions[{1, 2}] = 0.1f;
    predictions[{2, 0}] = 0.1f; predictions[{2, 1}] = 0.1f;  predictions[{2, 2}] = 0.8f;
    predictions[{3, 0}] = 0.7f; predictions[{3, 1}] = 0.2f;  predictions[{3, 2}] = 0.1f;
    predictions[{4, 0}] = 0.1f; predictions[{4, 1}] = 0.7f;  predictions[{4, 2}] = 0.2f;
    
    std::vector<uint8_t> labels = {0, 1, 2, 0, 1};
    
    float acc = compute_accuracy(predictions, labels);
    
    EXPECT_TRUE(approx_equal(acc, 1.0f, 1e-5f));
}

TEST_F(NNLayersTest, ComputeAccuracyPartialCorrect) {
    Matrixf predictions({4, 3});
    predictions[{0, 0}] = 0.9f; predictions[{0, 1}] = 0.05f; predictions[{0, 2}] = 0.05f;
    predictions[{1, 0}] = 0.1f; predictions[{1, 1}] = 0.8f;  predictions[{1, 2}] = 0.1f;
    predictions[{2, 0}] = 0.1f; predictions[{2, 1}] = 0.1f;  predictions[{2, 2}] = 0.8f;
    predictions[{3, 0}] = 0.7f; predictions[{3, 1}] = 0.2f;  predictions[{3, 2}] = 0.1f;
    
    std::vector<uint8_t> labels = {0, 1, 2, 1};
    
    float acc = compute_accuracy(predictions, labels);
    
    EXPECT_TRUE(approx_equal(acc, 0.75f, 1e-5f));
}

TEST_F(NNLayersTest, ComputeAccuracyWithOffset) {
    Matrixf predictions({3, 4});
    predictions[{0, 0}] = 0.1f; predictions[{0, 1}] = 0.6f; predictions[{0, 2}] = 0.2f; predictions[{0, 3}] = 0.1f;
    predictions[{1, 0}] = 0.7f; predictions[{1, 1}] = 0.1f; predictions[{1, 2}] = 0.1f; predictions[{1, 3}] = 0.1f;
    predictions[{2, 0}] = 0.1f; predictions[{2, 1}] = 0.1f; predictions[{2, 2}] = 0.1f; predictions[{2, 3}] = 0.7f;
    
    std::vector<uint8_t> labels = {5, 6, 7, 1, 0, 3};
    
    float acc = compute_accuracy(predictions, labels, 3);
    
    EXPECT_TRUE(approx_equal(acc, 1.0f, 1e-5f));
}

TEST_F(NNLayersTest, ComputeAccuracyAllWrong) {
    Matrixf predictions({3, 3});
    predictions[{0, 0}] = 0.1f; predictions[{0, 1}] = 0.8f; predictions[{0, 2}] = 0.1f;
    predictions[{1, 0}] = 0.1f; predictions[{1, 1}] = 0.1f; predictions[{1, 2}] = 0.8f;
    predictions[{2, 0}] = 0.8f; predictions[{2, 1}] = 0.1f; predictions[{2, 2}] = 0.1f;
    
    std::vector<uint8_t> labels = {0, 0, 1};
    
    float acc = compute_accuracy(predictions, labels);
    
    EXPECT_TRUE(approx_equal(acc, 0.0f, 1e-5f));
}
