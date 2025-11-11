/**
 * @file tensor_nn_enhancements_test.cc
 * @brief Google Test suite for new neural network enhancement operations in tensor.h
 * 
 * Tests the following new operations:
 * - softmax_rows(): Row-wise softmax activation
 * - argmax_rows(): Row-wise argmax for classification
 * - randn(): Random normal distribution initialization
 * - rand_uniform(): Random uniform distribution initialization
 * - fused_scalar_mul_sub(): Fused SGD update operation
 * - fill_rows(): Batch data loading
 */

#include <gtest/gtest.h>
#include "tensor.h"
#include "nn_layers.h"
#include <cmath>
#include <vector>

using namespace tensor;

// ============================================
// Test Fixture for NN Enhancements
// ============================================

class TensorNNEnhancementsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// ============================================
// Test: softmax_rows()
// ============================================

TEST_F(TensorNNEnhancementsTest, SoftmaxRows_BasicFunctionality) {
    // Create a 2x4 tensor with known values
    Tensor<float, 2> logits({2, 4});
    logits[{0, 0}] = 2.0f; 
    logits[{0, 1}] = 1.0f; 
    logits[{0, 2}] = 0.1f; 
    logits[{0, 3}] = 0.5f;
    
    logits[{1, 0}] = 0.5f; 
    logits[{1, 1}] = 2.5f; 
    logits[{1, 2}] = 1.2f; 
    logits[{1, 3}] = 0.3f;
    
    auto probs = logits.softmax_rows();
    
    // Check dimensions
    auto shape = probs.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);
    
    // Check that each row sums to 1.0
    for (size_t i = 0; i < 2; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 4; ++j) {
            sum += probs[{i, j}];
            float val = probs[{i, j}];
            // Check that all values are positive
            EXPECT_GT(val, 0.0f);
            EXPECT_LT(val, 1.0f);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-6f);
    }
}

TEST_F(TensorNNEnhancementsTest, SoftmaxRows_NumericalStability) {
    // Test with large values to ensure numerical stability
    Tensor<float, 2> logits({2, 3});
    logits[{0, 0}] = 1000.0f;
    logits[{0, 1}] = 1000.0f;
    logits[{0, 2}] = 1000.0f;
    
    logits[{1, 0}] = 100.0f;
    logits[{1, 1}] = 200.0f;
    logits[{1, 2}] = 300.0f;
    
    auto probs = logits.softmax_rows();
    
    // Check for NaN or Inf
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FALSE(std::isnan(probs[{i, j}]));
            EXPECT_FALSE(std::isinf(probs[{i, j}]));
        }
    }
    
    // First row should be uniform (all same values)
    float prob00 = probs[{0, 0}];
    float prob01 = probs[{0, 1}];
    float prob02 = probs[{0, 2}];
    EXPECT_NEAR(prob00, 1.0f/3.0f, 1e-5f);
    EXPECT_NEAR(prob01, 1.0f/3.0f, 1e-5f);
    EXPECT_NEAR(prob02, 1.0f/3.0f, 1e-5f);
}

// ============================================
// Test: argmax_rows()
// ============================================

TEST_F(TensorNNEnhancementsTest, ArgmaxRows_BasicFunctionality) {
    Tensor<float, 2> predictions({3, 5});
    
    // Row 0: max at index 2
    predictions[{0, 0}] = 0.1f; 
    predictions[{0, 1}] = 0.2f; 
    predictions[{0, 2}] = 0.5f; 
    predictions[{0, 3}] = 0.1f; 
    predictions[{0, 4}] = 0.1f;
    
    // Row 1: max at index 4
    predictions[{1, 0}] = 0.05f; 
    predictions[{1, 1}] = 0.1f; 
    predictions[{1, 2}] = 0.2f; 
    predictions[{1, 3}] = 0.15f; 
    predictions[{1, 4}] = 0.5f;
    
    // Row 2: max at index 0
    predictions[{2, 0}] = 0.6f; 
    predictions[{2, 1}] = 0.1f; 
    predictions[{2, 2}] = 0.1f; 
    predictions[{2, 3}] = 0.1f; 
    predictions[{2, 4}] = 0.1f;
    
    auto pred_classes = predictions.argmax_rows();
    
    // Check dimensions
    auto shape = pred_classes.shape();
    EXPECT_EQ(shape[0], 3);
    
    // Check argmax results
    EXPECT_EQ(pred_classes[{0}], 2);
    EXPECT_EQ(pred_classes[{1}], 4);
    EXPECT_EQ(pred_classes[{2}], 0);
}

TEST_F(TensorNNEnhancementsTest, ArgmaxRows_TieBreaking) {
    // Test behavior when multiple values are equal (should return first occurrence)
    Tensor<float, 2> predictions({2, 4});
    
    // Row 0: multiple maxima (both 0.5)
    predictions[{0, 0}] = 0.1f;
    predictions[{0, 1}] = 0.5f;  // First max
    predictions[{0, 2}] = 0.5f;  // Second max
    predictions[{0, 3}] = 0.1f;
    
    // Row 1: single clear max
    predictions[{1, 0}] = 0.2f;
    predictions[{1, 1}] = 0.1f;
    predictions[{1, 2}] = 0.8f;
    predictions[{1, 3}] = 0.3f;
    
    auto pred_classes = predictions.argmax_rows();
    
    // Should return first occurrence of max
    EXPECT_EQ(pred_classes[{0}], 1);
    EXPECT_EQ(pred_classes[{1}], 2);
}

// ============================================
// Test: randn()
// ============================================

TEST_F(TensorNNEnhancementsTest, Randn_DistributionProperties) {
    const size_t n_samples = 10000;
    Tensor<float, 2> samples({1, n_samples});
    
    float mean_target = 0.0f;
    float stddev_target = 1.0f;
    
    samples.randn(mean_target, stddev_target);
    
    // Compute sample mean and variance
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (size_t i = 0; i < n_samples; ++i) {
        float val = samples[{0, i}];
        sum += val;
        sum_sq += val * val;
    }
    
    float sample_mean = sum / n_samples;
    float sample_variance = (sum_sq / n_samples) - (sample_mean * sample_mean);
    float sample_stddev = std::sqrt(sample_variance);
    
    // Check mean is close to 0 (within 3 standard errors)
    float stderr_mean = stddev_target / std::sqrt(n_samples);
    EXPECT_NEAR(sample_mean, mean_target, 3.0f * stderr_mean);
    
    // Check stddev is close to 1 (within reasonable tolerance)
    EXPECT_NEAR(sample_stddev, stddev_target, 0.05f);
}

TEST_F(TensorNNEnhancementsTest, Randn_CustomParameters) {
    const size_t n_samples = 1000;
    Tensor<float, 2> samples({1, n_samples});
    
    float mean_target = 5.0f;
    float stddev_target = 2.0f;
    
    samples.randn(mean_target, stddev_target);
    
    float sum = 0.0f;
    for (size_t i = 0; i < n_samples; ++i) {
        sum += samples[{0, i}];
    }
    
    float sample_mean = sum / n_samples;
    
    // Mean should be approximately 5.0
    EXPECT_NEAR(sample_mean, mean_target, 0.5f);
}

// ============================================
// Test: rand_uniform()
// ============================================

TEST_F(TensorNNEnhancementsTest, RandUniform_Range) {
    const size_t n_samples = 1000;
    Tensor<float, 2> samples({1, n_samples});
    
    float min_val = 0.0f;
    float max_val = 1.0f;
    
    samples.rand_uniform(min_val, max_val);
    
    // Check all values are within range
    for (size_t i = 0; i < n_samples; ++i) {
        float val = samples[{0, i}];
        EXPECT_GE(val, min_val);
        EXPECT_LE(val, max_val);
    }
}

TEST_F(TensorNNEnhancementsTest, RandUniform_CustomRange) {
    const size_t n_samples = 1000;
    Tensor<float, 2> samples({1, n_samples});
    
    float min_val = -5.0f;
    float max_val = 10.0f;
    
    samples.rand_uniform(min_val, max_val);
    
    float sum = 0.0f;
    float observed_min = max_val;
    float observed_max = min_val;
    
    for (size_t i = 0; i < n_samples; ++i) {
        float val = samples[{0, i}];
        sum += val;
        observed_min = std::min(observed_min, val);
        observed_max = std::max(observed_max, val);
        
        EXPECT_GE(val, min_val);
        EXPECT_LE(val, max_val);
    }
    
    // Check mean is approximately at midpoint
    float expected_mean = (min_val + max_val) / 2.0f;
    float sample_mean = sum / n_samples;
    EXPECT_NEAR(sample_mean, expected_mean, 1.0f);
}

// ============================================
// Test: fused_scalar_mul_sub()
// ============================================

TEST_F(TensorNNEnhancementsTest, FusedScalarMulSub_BasicOperation) {
    Tensor<float, 2> params({2, 3});
    Tensor<float, 2> grads({2, 3});
    
    params.fill(1.0f);
    grads.fill(0.1f);
    
    float learning_rate = 0.5f;
    
    // params -= lr * grads
    params.fused_scalar_mul_sub(learning_rate, grads);
    
    // Expected: 1.0 - 0.5 * 0.1 = 0.95
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float val = params[{i, j}];
            EXPECT_NEAR(val, 0.95f, 1e-6f);
        }
    }
}

TEST_F(TensorNNEnhancementsTest, FusedScalarMulSub_VaryingValues) {
    Tensor<float, 2> params({2, 2});
    Tensor<float, 2> grads({2, 2});
    
    params[{0, 0}] = 1.0f; params[{0, 1}] = 2.0f;
    params[{1, 0}] = 3.0f; params[{1, 1}] = 4.0f;
    
    grads[{0, 0}] = 0.1f; grads[{0, 1}] = 0.2f;
    grads[{1, 0}] = 0.3f; grads[{1, 1}] = 0.4f;
    
    float lr = 1.0f;
    
    params.fused_scalar_mul_sub(lr, grads);
    
    float val00 = params[{0, 0}];
    float val01 = params[{0, 1}];
    float val10 = params[{1, 0}];
    float val11 = params[{1, 1}];
    
    EXPECT_NEAR(val00, 0.9f, 1e-6f);  // 1.0 - 1.0*0.1
    EXPECT_NEAR(val01, 1.8f, 1e-6f);  // 2.0 - 1.0*0.2
    EXPECT_NEAR(val10, 2.7f, 1e-6f);  // 3.0 - 1.0*0.3
    EXPECT_NEAR(val11, 3.6f, 1e-6f);  // 4.0 - 1.0*0.4
}

TEST_F(TensorNNEnhancementsTest, FusedScalarMulSub_DimensionMismatch) {
    Tensor<float, 2> params({2, 3});
    Tensor<float, 2> grads({3, 2});  // Different dimensions
    
    params.fill(1.0f);
    grads.fill(0.1f);
    
    // Should handle gracefully (no operation or no crash)
    params.fused_scalar_mul_sub(0.5f, grads);
    
    // Params should be unchanged
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float val = params[{i, j}];
            EXPECT_NEAR(val, 1.0f, 1e-6f);
        }
    }
}

// ============================================
// Test: fill_rows()
// ============================================

TEST_F(TensorNNEnhancementsTest, FillRows_BasicFunctionality) {
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    
    Tensor<float, 2> tensor({2, 3});
    tensor.fill_rows(data, 0);
    
    // Check values
    float t00 = tensor[{0, 0}];
    float t01 = tensor[{0, 1}];
    float t02 = tensor[{0, 2}];
    float t10 = tensor[{1, 0}];
    float t11 = tensor[{1, 1}];
    float t12 = tensor[{1, 2}];
    
    EXPECT_FLOAT_EQ(t00, 1.0f);
    EXPECT_FLOAT_EQ(t01, 2.0f);
    EXPECT_FLOAT_EQ(t02, 3.0f);
    EXPECT_FLOAT_EQ(t10, 4.0f);
    EXPECT_FLOAT_EQ(t11, 5.0f);
    EXPECT_FLOAT_EQ(t12, 6.0f);
}

TEST_F(TensorNNEnhancementsTest, FillRows_WithOffset) {
    std::vector<std::vector<float>> data = {
        {7.0f, 8.0f, 9.0f}
    };
    
    Tensor<float, 2> tensor({3, 3});
    tensor.fill(0.0f);
    
    // Fill starting at row 1
    tensor.fill_rows(data, 1);
    
    // Row 0 should be zeros
    float r00 = tensor[{0, 0}];
    float r01 = tensor[{0, 1}];
    float r02 = tensor[{0, 2}];
    EXPECT_FLOAT_EQ(r00, 0.0f);
    EXPECT_FLOAT_EQ(r01, 0.0f);
    EXPECT_FLOAT_EQ(r02, 0.0f);
    
    // Row 1 should have data
    float r10 = tensor[{1, 0}];
    float r11 = tensor[{1, 1}];
    float r12 = tensor[{1, 2}];
    EXPECT_FLOAT_EQ(r10, 7.0f);
    EXPECT_FLOAT_EQ(r11, 8.0f);
    EXPECT_FLOAT_EQ(r12, 9.0f);
    
    // Row 2 should be zeros
    float r20 = tensor[{2, 0}];
    float r21 = tensor[{2, 1}];
    float r22 = tensor[{2, 2}];
    EXPECT_FLOAT_EQ(r20, 0.0f);
    EXPECT_FLOAT_EQ(r21, 0.0f);
    EXPECT_FLOAT_EQ(r22, 0.0f);
}

// ============================================
// Test: Optimized Softmax Layer
// ============================================

TEST_F(TensorNNEnhancementsTest, OptimizedSoftmaxLayer_Forward) {
    Softmax<float> softmax;
    
    Tensor<float, 2> input({2, 3});
    input[{0, 0}] = 1.0f; input[{0, 1}] = 2.0f; input[{0, 2}] = 3.0f;
    input[{1, 0}] = 0.5f; input[{1, 1}] = 1.5f; input[{1, 2}] = 2.5f;
    
    auto output = softmax.forward(input);
    
    // Check dimensions
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    
    // Check sum to 1.0 per row
    for (size_t i = 0; i < 2; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            sum += output[{i, j}];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-6f);
    }
}

// ============================================
// Integration Test: SGD Update with Fused Op
// ============================================

TEST_F(TensorNNEnhancementsTest, Integration_SGDUpdate) {
    // Simulate a simple SGD update scenario
    const size_t num_params = 3;
    std::vector<Tensor<float, 2>*> params;
    std::vector<Tensor<float, 2>> grads;
    
    for (size_t i = 0; i < num_params; ++i) {
        params.push_back(new Tensor<float, 2>({2, 2}));
        params[i]->fill(1.0f);
        
        grads.emplace_back(Tensor<float, 2>({2, 2}));
        grads[i].fill(0.1f);
    }
    
    float learning_rate = 0.01f;
    
    // Perform SGD update
    for (size_t i = 0; i < num_params; ++i) {
        params[i]->fused_scalar_mul_sub(learning_rate, grads[i]);
    }
    
    // Check all parameters updated correctly
    float expected = 1.0f - 0.01f * 0.1f;  // 0.999
    for (size_t i = 0; i < num_params; ++i) {
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 2; ++c) {
                float val = (*params[i])[{r, c}];
                EXPECT_NEAR(val, expected, 1e-6f);
            }
        }
    }
    
    // Cleanup
    for (auto p : params) {
        delete p;
    }
}
