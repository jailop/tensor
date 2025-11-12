#include "../include/tensor.h"
#include <gtest/gtest.h>
#include <cmath>

#ifdef USE_GPU
#include "../include/tensor_gpu.cuh"
#endif

using namespace tensor;

class TensorGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef USE_GPU
        if (!is_gpu_available()) {
            GTEST_SKIP() << "GPU not available, skipping GPU tests";
        }
#else
        GTEST_SKIP() << "Tests compiled without GPU support";
#endif
    }
    
    void TearDown() override {}
};

#ifdef USE_GPU

// GPU Availability Tests

TEST_F(TensorGPUTest, GPUAvailability) {
    EXPECT_TRUE(is_gpu_available());
}

// Note: TensorGPU doesn't expose get_device_count, testing availability is sufficient
// TEST_F(TensorGPUTest, GPUDeviceCount) {
//     int device_count = get_device_count();
//     EXPECT_GT(device_count, 0);
// }

// =============================================================================
// GPU Memory Management Tests
// =============================================================================

// Note: These tests use CUDA API directly, which are implementation details
// The high-level tensor operations are tested in the integration tests below
/*
TEST_F(TensorGPUTest, GPUMemoryAllocation) {
    const size_t size = 1000;
    float* d_ptr = nullptr;
    
    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&d_ptr, size * sizeof(float));
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(d_ptr, nullptr);
    
    // Free GPU memory
    err = cudaFree(d_ptr);
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(TensorGPUTest, GPUMemoryCopyHostToDevice) {
    const size_t size = 100;
    std::vector<float> h_data(size);
    
    // Fill host data
    for (size_t i = 0; i < size; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float* d_data = nullptr;
    cudaMalloc(&d_data, size * sizeof(float));
    
    // Copy to device
    cudaError_t err = cudaMemcpy(d_data, h_data.data(), 
                                  size * sizeof(float), 
                                  cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);
    
    // Copy back to verify
    std::vector<float> h_result(size);
    err = cudaMemcpy(h_result.data(), d_data, 
                     size * sizeof(float), 
                     cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);
    
    // Verify data
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
    }
    
    cudaFree(d_data);
}
*/

// =============================================================================
// Tensor GPU Backend Tests
// =============================================================================

TEST_F(TensorGPUTest, TensorUsesGPUByDefault) {
    Tensor<float, 1> tensor({100}, true);  // use_gpu=true
    EXPECT_TRUE(tensor.uses_gpu());
    EXPECT_EQ(tensor.backend(), Backend::GPU);
}

TEST_F(TensorGPUTest, TensorGPUCreationAndAccess) {
    Tensor<float, 2> tensor({3, 4}, true);
    tensor.fill(5.0f);
    
    // Verify backend
    EXPECT_EQ(tensor.backend(), Backend::GPU);
    
    // Access elements (should work via CPU-GPU transfers)
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((tensor[{i, j}]), 5.0f);
        }
    }
}

// =============================================================================
// GPU Arithmetic Operations Tests
// =============================================================================

TEST_F(TensorGPUTest, GPUAddition) {
    Tensor<float, 1> a({1000}, true);
    Tensor<float, 1> b({1000}, true);
    
    a.fill(2.0f);
    b.fill(3.0f);
    
    auto result_var = a + b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result_var)));
    
    auto& result = std::get<Tensor<float, 1>>(result_var);
    
    // Verify some elements
    EXPECT_FLOAT_EQ((result[{0}]), 5.0f);
    EXPECT_FLOAT_EQ((result[{500}]), 5.0f);
    EXPECT_FLOAT_EQ((result[{999}]), 5.0f);
}

TEST_F(TensorGPUTest, GPUSubtraction) {
    Tensor<float, 1> a({1000}, true);
    Tensor<float, 1> b({1000}, true);
    
    a.fill(10.0f);
    b.fill(3.0f);
    
    auto result_var = a - b;
    auto& result = std::get<Tensor<float, 1>>(result_var);
    
    EXPECT_FLOAT_EQ((result[{0}]), 7.0f);
    EXPECT_FLOAT_EQ((result[{999}]), 7.0f);
}

TEST_F(TensorGPUTest, GPUMultiplication) {
    Tensor<float, 1> a({1000}, true);
    Tensor<float, 1> b({1000}, true);
    
    a.fill(4.0f);
    b.fill(2.5f);
    
    auto result_var = a * b;
    auto& result = std::get<Tensor<float, 1>>(result_var);
    
    EXPECT_FLOAT_EQ((result[{0}]), 10.0f);
    EXPECT_FLOAT_EQ((result[{999}]), 10.0f);
}

TEST_F(TensorGPUTest, GPUDivision) {
    Tensor<float, 1> a({1000}, true);
    Tensor<float, 1> b({1000}, true);
    
    a.fill(10.0f);
    b.fill(2.0f);
    
    auto result_var = a / b;
    auto& result = std::get<Tensor<float, 1>>(result_var);
    
    EXPECT_FLOAT_EQ((result[{0}]), 5.0f);
    EXPECT_FLOAT_EQ((result[{999}]), 5.0f);
}

TEST_F(TensorGPUTest, GPUScalarOperations) {
    Tensor<float, 1> a({1000}, true);
    a.fill(5.0f);
    
    auto result = a * 2.0f;
    
    EXPECT_FLOAT_EQ((result[{0}]), 10.0f);
    EXPECT_FLOAT_EQ((result[{500}]), 10.0f);
}

// =============================================================================
// GPU Mathematical Functions Tests
// =============================================================================

TEST_F(TensorGPUTest, GPUExp) {
    Tensor<float, 1> a({100}, true);
    a.fill(1.0f);
    
    auto result = a.exp();
    
    EXPECT_NEAR((result[{0}]), std::exp(1.0f), 0.0001f);
    EXPECT_NEAR((result[{50}]), std::exp(1.0f), 0.0001f);
}

TEST_F(TensorGPUTest, GPULog) {
    Tensor<float, 1> a({100}, true);
    a.fill(2.718281828f);  // e
    
    auto result = a.log();
    
    EXPECT_NEAR((result[{0}]), 1.0f, 0.0001f);
}

TEST_F(TensorGPUTest, GPUSqrt) {
    Tensor<float, 1> a({100}, true);
    a.fill(4.0f);
    
    auto result = a.sqrt();
    
    EXPECT_FLOAT_EQ((result[{0}]), 2.0f);
}

TEST_F(TensorGPUTest, GPUPow) {
    Tensor<float, 1> a({100}, true);
    a.fill(2.0f);
    
    auto result = a.pow(3.0f);
    
    EXPECT_FLOAT_EQ((result[{0}]), 8.0f);
}

TEST_F(TensorGPUTest, GPUSin) {
    Tensor<float, 1> a({100}, true);
    a.fill(0.0f);
    
    auto result = a.sin();
    
    EXPECT_NEAR((result[{0}]), 0.0f, 0.0001f);
}

TEST_F(TensorGPUTest, GPUCos) {
    Tensor<float, 1> a({100}, true);
    a.fill(0.0f);
    
    auto result = a.cos();
    
    EXPECT_NEAR((result[{0}]), 1.0f, 0.0001f);
}

TEST_F(TensorGPUTest, GPUTanh) {
    Tensor<float, 1> a({100}, true);
    a.fill(0.0f);
    
    auto result = a.tanh();
    
    EXPECT_NEAR((result[{0}]), 0.0f, 0.0001f);
}

TEST_F(TensorGPUTest, GPUSigmoid) {
    Tensor<float, 1> a({100}, true);
    a.fill(0.0f);
    
    auto result = a.sigmoid();
    
    EXPECT_NEAR((result[{0}]), 0.5f, 0.0001f);
}

TEST_F(TensorGPUTest, GPUReLU) {
    Tensor<float, 1> a({100}, true);
    
    // Fill with mix of positive and negative
    for (size_t i = 0; i < 100; ++i) {
        a[{i}] = static_cast<float>(i) - 50.0f;
    }
    
    auto result = a.relu();
    
    EXPECT_FLOAT_EQ((result[{0}]), 0.0f);   // -50 -> 0
    EXPECT_FLOAT_EQ((result[{49}]), 0.0f);  // -1 -> 0
    EXPECT_FLOAT_EQ((result[{50}]), 0.0f);  // 0 -> 0
    EXPECT_FLOAT_EQ((result[{51}]), 1.0f);  // 1 -> 1
    EXPECT_FLOAT_EQ((result[{99}]), 49.0f); // 49 -> 49
}

// =============================================================================
// GPU vs CPU Consistency Tests
// =============================================================================

TEST_F(TensorGPUTest, GPUvsCPU_Addition) {
    const size_t size = 1000;
    
    // GPU version
    Tensor<float, 1> gpu_a({size}, true);
    Tensor<float, 1> gpu_b({size}, true);
    gpu_a.fill(2.5f);
    gpu_b.fill(3.5f);
    auto gpu_result_var = gpu_a + gpu_b;
    auto& gpu_result = std::get<Tensor<float, 1>>(gpu_result_var);
    
    // CPU version
    Tensor<float, 1> cpu_a({size}, false);
    Tensor<float, 1> cpu_b({size}, false);
    cpu_a.fill(2.5f);
    cpu_b.fill(3.5f);
    auto cpu_result_var = cpu_a + cpu_b;
    auto& cpu_result = std::get<Tensor<float, 1>>(cpu_result_var);
    
    // Compare results
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ((gpu_result[{i}]), (cpu_result[{i}]));
    }
}

TEST_F(TensorGPUTest, GPUvsCPU_Exp) {
    const size_t size = 100;
    
    // GPU version
    Tensor<float, 1> gpu_a({size}, true);
    for (size_t i = 0; i < size; ++i) {
        gpu_a[{i}] = static_cast<float>(i) * 0.1f;
    }
    auto gpu_result = gpu_a.exp();
    
    // CPU version
    Tensor<float, 1> cpu_a({size}, false);
    for (size_t i = 0; i < size; ++i) {
        cpu_a[{i}] = static_cast<float>(i) * 0.1f;
    }
    auto cpu_result = cpu_a.exp();
    
    // Compare results (allow small difference due to GPU precision)
    for (size_t i = 0; i < size; ++i) {
        float expected = cpu_result[{i}];
        float actual = gpu_result[{i}];
        // Use relative tolerance for large values
        float tolerance = std::max(0.001f, std::abs(expected) * 0.0001f);
        EXPECT_NEAR(actual, expected, tolerance);
    }
}

TEST_F(TensorGPUTest, GPUvsCPU_MatrixMultiplication) {
    // GPU version
    Tensor<float, 2> gpu_a({3, 4}, true);
    Tensor<float, 2> gpu_b({4, 5}, true);
    gpu_a.fill(2.0f);
    gpu_b.fill(3.0f);
    
    auto gpu_result_var = gpu_a.matmul(gpu_b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(gpu_result_var)));
    auto& gpu_result = std::get<Tensor<float, 2>>(gpu_result_var);
    
    // CPU version
    Tensor<float, 2> cpu_a({3, 4}, false);
    Tensor<float, 2> cpu_b({4, 5}, false);
    cpu_a.fill(2.0f);
    cpu_b.fill(3.0f);
    
    auto cpu_result_var = cpu_a.matmul(cpu_b);
    auto& cpu_result = std::get<Tensor<float, 2>>(cpu_result_var);
    
    // Compare results
    ASSERT_EQ(gpu_result.dims()[0], cpu_result.dims()[0]);
    ASSERT_EQ(gpu_result.dims()[1], cpu_result.dims()[1]);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            EXPECT_NEAR((gpu_result[{i, j}]), (cpu_result[{i, j}]), 0.001f);
        }
    }
}

// =============================================================================
// GPU Performance Tests (Optional)
// =============================================================================

TEST_F(TensorGPUTest, DISABLED_GPUPerformance_LargeAddition) {
    const size_t size = 10000000;  // 10M elements
    
    Tensor<float, 1> a({size}, true);
    Tensor<float, 1> b({size}, true);
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result_var = a + b;
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "GPU addition of " << size << " elements took: " 
              << duration.count() << "ms" << std::endl;
    
    // Just verify it completed
    EXPECT_TRUE((std::holds_alternative<Tensor<float, 1>>(result_var)));
}

TEST_F(TensorGPUTest, DISABLED_GPUvsCPU_PerformanceComparison) {
    const size_t size = 1000000;
    
    // GPU timing
    Tensor<float, 1> gpu_a({size}, true);
    Tensor<float, 1> gpu_b({size}, true);
    gpu_a.fill(1.0f);
    gpu_b.fill(2.0f);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    auto gpu_result_var = gpu_a * gpu_b;
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    
    // CPU timing
    Tensor<float, 1> cpu_a({size}, false);
    Tensor<float, 1> cpu_b({size}, false);
    cpu_a.fill(1.0f);
    cpu_b.fill(2.0f);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_result_var = cpu_a * cpu_b;
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    std::cout << "GPU time: " << gpu_duration.count() << "μs" << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << "μs" << std::endl;
    std::cout << "Speedup: " << (double)cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    
    // GPU should generally be faster for large operations
    // (This might not always be true for small sizes due to transfer overhead)
    EXPECT_GT(size, 0);  // Just a placeholder assertion
}

#endif // USE_GPU
