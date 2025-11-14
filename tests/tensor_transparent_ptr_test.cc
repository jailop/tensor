#include <gtest/gtest.h>
#include "tensor_base.h"

#ifdef USE_GPU

TEST(TensorTransparentPtrTest, DataPtrReturnsGPUPointerAfterGPUOperation) {
    tensor::Tensor<float, 1> a({10}, true);
    tensor::Tensor<float, 1> b({10}, true);
    a.fill(2.0f);
    b.fill(3.0f);
    
    auto result = a + b;
    EXPECT_TRUE((std::holds_alternative<tensor::Tensor<float, 1>>(result)));
    auto& c = std::get<tensor::Tensor<float, 1>>(result);
    
    EXPECT_TRUE(c.uses_gpu());
    EXPECT_TRUE(c.is_data_on_gpu());
    
    const float* ptr = c.begin();
    
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(c.is_data_on_gpu());
}

TEST(TensorTransparentPtrTest, NonConstDataPtrMarksGPUModifiedAfterGPUOp) {
    tensor::Tensor<float, 1> a({10}, true);
    tensor::Tensor<float, 1> b({10}, true);
    a.fill(2.0f);
    b.fill(3.0f);
    
    auto result = a + b;
    EXPECT_TRUE((std::holds_alternative<tensor::Tensor<float, 1>>(result)));
    auto& c = std::get<tensor::Tensor<float, 1>>(result);
    
    EXPECT_TRUE(c.is_data_on_gpu());
    EXPECT_FALSE(c.gpu_needs_sync());
    
    // Getting mutable pointer doesn't automatically invalidate GPU
    float* ptr = c.begin();
    
    EXPECT_NE(ptr, nullptr);
    // GPU state unchanged - user controls when to mark as modified
    EXPECT_TRUE(c.is_data_on_gpu());
    EXPECT_FALSE(c.gpu_needs_sync());
    
    // User modifies data through pointer
    ptr[0] = 99.0f;
    
    // User must explicitly mark as modified
    c.mark_cpu_modified();
    
    // Now GPU is stale
    EXPECT_FALSE(c.is_data_on_gpu());
}

TEST(TensorTransparentPtrTest, DataPtrReturnsCPUWhenNotOnGPU) {
    tensor::Tensor<float, 1> t({10}, false);
    t.fill(1.5f);
    
    EXPECT_FALSE(t.is_data_on_gpu());
    
    const float* ptr = t.begin();
    EXPECT_NE(ptr, nullptr);
}

#endif
