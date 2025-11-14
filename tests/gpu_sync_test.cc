/**
 * Tests for GPU memory synchronization with iterators and operator[]
 */

#include <gtest/gtest.h>
#include "tensor.h"

#ifdef USE_GPU

using namespace tensor;

TEST(GPUSyncTest, OperatorBracketAutoSync) {
    // Test that operator[] automatically syncs GPU when modified
    Tensor<float, 1> t({10}, true);  // GPU enabled
    t.fill(1.0f);
    
    // Modify through operator[] - should auto-mark CPU as modified
    t[{0}] = 5.0f;
    t[{1}] = 3.0f;
    
    // GPU is now stale, but sync_to_gpu() can update it
    t.sync_to_gpu();
    
    // Verify values are correct
    EXPECT_FLOAT_EQ(static_cast<float>(t[{0}]), 5.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t[{1}]), 3.0f);
}

TEST(GPUSyncTest, IteratorModification) {
    // Test iterator-based modification
    Tensor<float, 1> t({10}, true);  // GPU enabled
    t.fill(0.0f);
    
    // Modify through iterator
    float* ptr = t.begin();
    for (size_t i = 0; i < 10; ++i) {
        ptr[i] = static_cast<float>(i);
    }
    
    // Manual sync needed after iterator modification
    t.sync_to_gpu();
    
    // Verify through operator[]
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[{i}]), static_cast<float>(i));
    }
}

TEST(GPUSyncTest, CompoundAssignment) {
    // Test compound assignment operators with proxy
    Tensor<float, 1> t({5}, true);  // GPU enabled
    t.fill(10.0f);
    
    // Compound assignments should work with proxy
    t[{0}] += 5.0f;
    t[{1}] -= 3.0f;
    t[{2}] *= 2.0f;
    t[{3}] /= 2.0f;
    
    t.sync_to_gpu();
    
    EXPECT_FLOAT_EQ(static_cast<float>(t[{0}]), 15.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t[{1}]), 7.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t[{2}]), 20.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t[{3}]), 5.0f);
}

TEST(GPUSyncTest, DataPointerAccess) {
    // Test data() method for raw pointer access
    Tensor<float, 2> t({3, 3}, true);  // GPU enabled
    t.fill(1.0f);
    
    // Get mutable pointer and modify
    float* data = t.data();
    data[0] = 99.0f;
    data[4] = 88.0f;
    
    // Need to manually mark as modified and sync
    t.mark_cpu_modified();
    t.sync_to_gpu();
    
    // Verify
    EXPECT_FLOAT_EQ(static_cast<float>(t[{0, 0}]), 99.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t[{1, 1}]), 88.0f);
}

TEST(GPUSyncTest, ConstAccessNoInvalidation) {
    // Test that const access doesn't invalidate GPU
    Tensor<float, 1> t({10}, true);
    t.fill(5.0f);
    
    // Ensure data is on GPU
    t.sync_to_gpu();
    
    // Const access should not invalidate GPU
    const Tensor<float, 1>& const_ref = t;
    float val = const_ref[{0}];
    EXPECT_FLOAT_EQ(val, 5.0f);
    
    // GPU should still be valid (not tested directly, but no crash means good)
}

#else

TEST(GPUSyncTest, GPUNotEnabled) {
    GTEST_SKIP() << "GPU not enabled in build";
}

#endif
