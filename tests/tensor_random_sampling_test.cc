#include <gtest/gtest.h>
#include "../include/tensor.h"
#include <set>
#include <cmath>

using namespace tensor;

class TensorRandomSamplingTest : public ::testing::Test {
protected:
    void SetUp() override {
        TensorRandom<float>::seed(42);  // Reproducible tests
    }
};

TEST_F(TensorRandomSamplingTest, UniformDistribution) {
    auto tensor = TensorRandom<float>::uniform<2>({{100, 100}}, 0.0f, 1.0f);
    
    ASSERT_EQ(tensor.dims()[0], 100);
    ASSERT_EQ(tensor.dims()[1], 100);
    
    // Check all values are in range [0, 1)
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        ASSERT_GE(tensor.data()[i], 0.0f);
        ASSERT_LT(tensor.data()[i], 1.0f);
    }
    
    // Check mean is approximately 0.5
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        sum += tensor.data()[i];
    }
    float mean = sum / tensor.total_size();
    ASSERT_NEAR(mean, 0.5f, 0.1f);
}

TEST_F(TensorRandomSamplingTest, UniformCustomRange) {
    auto tensor = TensorRandom<float>::uniform<1>({{1000}}, -5.0f, 5.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        ASSERT_GE(tensor.data()[i], -5.0f);
        ASSERT_LT(tensor.data()[i], 5.0f);
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        sum += tensor.data()[i];
    }
    float mean = sum / tensor.total_size();
    ASSERT_NEAR(mean, 0.0f, 0.5f);
}

TEST_F(TensorRandomSamplingTest, NormalDistribution) {
    auto tensor = TensorRandom<float>::normal<2>({{50, 50}}, 0.0f, 1.0f);
    
    ASSERT_EQ(tensor.dims()[0], 50);
    ASSERT_EQ(tensor.dims()[1], 50);
    
    // Compute mean
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        sum += tensor.data()[i];
    }
    float mean = sum / tensor.total_size();
    
    // Compute std dev
    float var_sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        float diff = tensor.data()[i] - mean;
        var_sum += diff * diff;
    }
    float std_dev = std::sqrt(var_sum / tensor.total_size());
    
    // Check mean and std dev are close to expected
    ASSERT_NEAR(mean, 0.0f, 0.2f);
    ASSERT_NEAR(std_dev, 1.0f, 0.2f);
}

TEST_F(TensorRandomSamplingTest, NormalCustomParameters) {
    auto tensor = TensorRandom<float>::normal<1>({{10000}}, 10.0f, 2.0f);
    
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        sum += tensor.data()[i];
    }
    float mean = sum / tensor.total_size();
    
    float var_sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        float diff = tensor.data()[i] - mean;
        var_sum += diff * diff;
    }
    float std_dev = std::sqrt(var_sum / tensor.total_size());
    
    ASSERT_NEAR(mean, 10.0f, 0.2f);
    ASSERT_NEAR(std_dev, 2.0f, 0.2f);
}

TEST_F(TensorRandomSamplingTest, ExponentialDistribution) {
    auto tensor = TensorRandom<float>::exponential<1>({{5000}}, 1.0f);
    
    // Mean should be 1/lambda = 1
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        sum += tensor.data()[i];
        ASSERT_GE(tensor.data()[i], 0.0f);  // All values should be non-negative
    }
    float mean = sum / tensor.total_size();
    
    ASSERT_NEAR(mean, 1.0f, 0.1f);
}

TEST_F(TensorRandomSamplingTest, Permutation) {
    auto perm = TensorRandom<float>::permutation(10);
    
    ASSERT_EQ(perm.size(), 10);
    
    // Check all elements 0-9 are present
    std::set<size_t> elements(perm.begin(), perm.end());
    ASSERT_EQ(elements.size(), 10);
    
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(elements.count(i) > 0);
    }
}

TEST_F(TensorRandomSamplingTest, Randperm) {
    auto tensor = TensorRandom<float>::randperm(20);
    
    ASSERT_EQ(tensor.dims()[0], 20);
    
    // Check all elements 0-19 are present
    std::set<float> elements;
    for (size_t i = 0; i < 20; ++i) {
        elements.insert(tensor.data()[i]);
    }
    ASSERT_EQ(elements.size(), 20);
    
    for (size_t i = 0; i < 20; ++i) {
        ASSERT_TRUE(elements.count(static_cast<float>(i)) > 0);
    }
}

TEST_F(TensorRandomSamplingTest, ChoiceWithoutReplacement) {
    auto choices = TensorRandom<float>::choice(100, 10);
    
    ASSERT_EQ(choices.size(), 10);
    
    // Check no duplicates
    std::set<size_t> unique_choices(choices.begin(), choices.end());
    ASSERT_EQ(unique_choices.size(), 10);
    
    // Check all in range [0, 100)
    for (auto c : choices) {
        ASSERT_GE(c, 0);
        ASSERT_LT(c, 100);
    }
}

TEST_F(TensorRandomSamplingTest, ChoiceWithReplacement) {
    auto choices = TensorRandom<float>::choice_with_replacement(10, 100);
    
    ASSERT_EQ(choices.size(), 100);
    
    // Check all in range [0, 10)
    for (auto c : choices) {
        ASSERT_GE(c, 0);
        ASSERT_LT(c, 10);
    }
    
    // With 100 samples from 10 elements, we should likely have duplicates
    std::set<size_t> unique_choices(choices.begin(), choices.end());
    ASSERT_LT(unique_choices.size(), choices.size());
}

TEST_F(TensorRandomSamplingTest, SeedReproducibility) {
    TensorRandom<float>::seed(123);
    auto tensor1 = TensorRandom<float>::uniform<1>({{100}});
    
    TensorRandom<float>::seed(123);
    auto tensor2 = TensorRandom<float>::uniform<1>({{100}});
    
    // With same seed, should get same values
    for (size_t i = 0; i < 100; ++i) {
        ASSERT_FLOAT_EQ(tensor1.data()[i], tensor2.data()[i]);
    }
}

TEST_F(TensorRandomSamplingTest, DifferentSeeds) {
    TensorRandom<float>::seed(123);
    auto tensor1 = TensorRandom<float>::uniform<1>({{100}});
    
    TensorRandom<float>::seed(456);
    auto tensor2 = TensorRandom<float>::uniform<1>({{100}});
    
    // With different seeds, should get different values
    int differences = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (tensor1.data()[i] != tensor2.data()[i]) {
            differences++;
        }
    }
    ASSERT_GT(differences, 90);  // Most should be different
}

TEST_F(TensorRandomSamplingTest, GammaDistribution) {
    auto tensor = TensorRandom<float>::gamma<1>({{5000}}, 2.0f, 1.0f);
    
    // Mean should be alpha * beta = 2 * 1 = 2
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        sum += tensor.data()[i];
        ASSERT_GE(tensor.data()[i], 0.0f);  // All values should be non-negative
    }
    float mean = sum / tensor.total_size();
    
    ASSERT_NEAR(mean, 2.0f, 0.2f);
}

TEST_F(TensorRandomSamplingTest, BetaDistribution) {
    auto tensor = TensorRandom<float>::beta<1>({{5000}}, 2.0f, 5.0f);
    
    // Mean should be alpha / (alpha + beta) = 2 / 7 ≈ 0.286
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        sum += tensor.data()[i];
        ASSERT_GE(tensor.data()[i], 0.0f);  // All values should be in [0,1]
        ASSERT_LE(tensor.data()[i], 1.0f);
    }
    float mean = sum / tensor.total_size();
    
    ASSERT_NEAR(mean, 2.0f / 7.0f, 0.1f);
}

TEST_F(TensorRandomSamplingTest, ChiSquaredDistribution) {
    auto tensor = TensorRandom<float>::chi_squared<1>({{5000}}, 3.0f);
    
    // Mean should be k (degrees of freedom) = 3
    float sum = 0.0f;
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        sum += tensor.data()[i];
        ASSERT_GE(tensor.data()[i], 0.0f);  // All values should be non-negative
    }
    float mean = sum / tensor.total_size();
    
    ASSERT_NEAR(mean, 3.0f, 0.3f);
}

TEST_F(TensorRandomSamplingTest, CauchyDistribution) {
    auto tensor = TensorRandom<float>::cauchy<1>({{1000}}, 0.0f, 1.0f);
    
    // Cauchy has no defined mean, but median should be at location parameter (0)
    // Just check that values are generated
    ASSERT_EQ(tensor.dims()[0], 1000);
    
    // Values can be anywhere, but shouldn't all be the same
    float first_val = tensor.data()[0];
    int differences = 0;
    for (size_t i = 1; i < tensor.total_size(); ++i) {
        if (tensor.data()[i] != first_val) {
            differences++;
        }
    }
    ASSERT_GT(differences, 900);  // Most should be different
}

TEST_F(TensorRandomSamplingTest, MultinomialDistribution) {
    std::vector<float> probs = {0.2f, 0.3f, 0.5f};
    size_t n_trials = 100;
    size_t n_samples = 1000;
    
    auto result = TensorRandom<float>::multinomial(n_trials, probs, n_samples);
    
    ASSERT_EQ(result.dims()[0], n_samples);
    ASSERT_EQ(result.dims()[1], 3);
    
    // Check that each sample sums to n_trials
    for (size_t i = 0; i < n_samples; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            float val = result[{i, j}];
            sum += val;
            ASSERT_GE(val, 0.0f);
        }
        ASSERT_NEAR(sum, static_cast<float>(n_trials), 0.01f);
    }
    
    // Check average proportions are close to probabilities
    float avg_counts[3] = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            avg_counts[j] += result[{i, j}];
        }
    }
    
    for (size_t j = 0; j < 3; ++j) {
        avg_counts[j] /= n_samples;
        float expected = probs[j] * n_trials;
        ASSERT_NEAR(avg_counts[j], expected, 3.0f);  // Allow some variance
    }
}

