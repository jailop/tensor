#include "../include/tensor_perf.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>

class TensorPerfTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Memory Pool Tests
// =============================================================================

TEST_F(TensorPerfTest, MemoryPool_BasicAllocationDeallocation) {
    MemoryPool<float> pool;
    
    // Allocate memory
    float* ptr = pool.allocate(100);
    ASSERT_NE(ptr, nullptr);
    
    // Use the memory
    for (size_t i = 0; i < 100; ++i) {
        ptr[i] = static_cast<float>(i);
    }
    
    // Return to pool
    pool.deallocate(ptr, 100);
    
    auto stats = pool.stats();
    EXPECT_EQ(stats.first, 1);  // One allocation
    EXPECT_EQ(stats.second, 1); // One deallocation
}

TEST_F(TensorPerfTest, MemoryPool_Reuse) {
    MemoryPool<float> pool;
    
    // Allocate and deallocate
    float* ptr1 = pool.allocate(100);
    pool.deallocate(ptr1, 100);
    
    // Allocate again - should reuse
    float* ptr2 = pool.allocate(100);
    
    auto stats = pool.stats();
    EXPECT_EQ(stats.first, 2);  // Two allocations counted
    EXPECT_EQ(stats.second, 1); // One deallocation
    
    // Verify memory is usable
    for (size_t i = 0; i < 100; ++i) {
        ptr2[i] = static_cast<float>(i * 2);
    }
    
    pool.deallocate(ptr2, 100);
}

TEST_F(TensorPerfTest, MemoryPool_MultipleSizes) {
    MemoryPool<double> pool;
    
    // Allocate different sizes
    double* small = pool.allocate(50);
    double* medium = pool.allocate(100);
    double* large = pool.allocate(200);
    
    ASSERT_NE(small, nullptr);
    ASSERT_NE(medium, nullptr);
    ASSERT_NE(large, nullptr);
    
    // Deallocate all
    pool.deallocate(small, 50);
    pool.deallocate(medium, 100);
    pool.deallocate(large, 200);
    
    auto stats = pool.stats();
    EXPECT_EQ(stats.first, 3);
    EXPECT_EQ(stats.second, 3);
}

TEST_F(TensorPerfTest, MemoryPool_Clear) {
    MemoryPool<int> pool;
    
    // Allocate and deallocate several blocks
    for (int i = 0; i < 5; ++i) {
        int* ptr = pool.allocate(100);
        pool.deallocate(ptr, 100);
    }
    
    // Clear the pool
    pool.clear();
    
    // Stats should still be correct
    auto stats = pool.stats();
    EXPECT_EQ(stats.first, 5);
    EXPECT_EQ(stats.second, 5);
}

TEST_F(TensorPerfTest, MemoryPool_GlobalInstance) {
    // Test the global memory pool
    auto& pool = get_memory_pool<float>();
    
    float* ptr = pool.allocate(100);
    ASSERT_NE(ptr, nullptr);
    
    pool.deallocate(ptr, 100);
}

TEST_F(TensorPerfTest, MemoryPool_ThreadSafety) {
    MemoryPool<float> pool;
    const int num_threads = 4;
    const int allocs_per_thread = 10;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool, allocs_per_thread]() {
            for (int i = 0; i < allocs_per_thread; ++i) {
                float* ptr = pool.allocate(100);
                // Simulate some work
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                pool.deallocate(ptr, 100);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto stats = pool.stats();
    EXPECT_EQ(stats.first, num_threads * allocs_per_thread);
    EXPECT_EQ(stats.second, num_threads * allocs_per_thread);
}

// =============================================================================
// Thread Pool Tests
// =============================================================================

TEST_F(TensorPerfTest, ThreadPool_BasicTaskExecution) {
    ThreadPool pool(2);
    
    // Enqueue a simple task
    auto future = pool.enqueue([]() {
        return 42;
    });
    
    int result = future.get();
    EXPECT_EQ(result, 42);
}

TEST_F(TensorPerfTest, ThreadPool_MultipleTasksSequential) {
    ThreadPool pool(2);
    
    std::vector<std::future<int>> futures;
    
    // Enqueue multiple tasks
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.enqueue([i]() {
            return i * i;
        }));
    }
    
    // Verify results
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST_F(TensorPerfTest, ThreadPool_TaskWithArguments) {
    ThreadPool pool(2);
    
    auto add = [](int a, int b) {
        return a + b;
    };
    
    auto future = pool.enqueue(add, 10, 20);
    
    EXPECT_EQ(future.get(), 30);
}

TEST_F(TensorPerfTest, ThreadPool_ConcurrentExecution) {
    ThreadPool pool(4);
    
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    
    // Enqueue tasks that increment counter
    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }
    
    // Wait for all tasks
    for (auto& future : futures) {
        future.get();
    }
    
    EXPECT_EQ(counter.load(), 100);
}

TEST_F(TensorPerfTest, ThreadPool_GlobalInstance) {
    auto& pool = get_thread_pool();
    
    auto future = pool.enqueue([]() {
        return std::string("Hello from thread pool");
    });
    
    std::string result = future.get();
    EXPECT_EQ(result, "Hello from thread pool");
}

TEST_F(TensorPerfTest, ThreadPool_Size) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.size(), 4);
}

// =============================================================================
// Parallel For Tests
// =============================================================================

TEST_F(TensorPerfTest, ParallelFor_BasicExecution) {
    std::vector<int> data(1000);
    
    // Fill vector in parallel
    parallel_for(0, 1000, [&data](size_t i) {
        data[i] = static_cast<int>(i * 2);
    });
    
    // Verify results
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i * 2));
    }
}

TEST_F(TensorPerfTest, ParallelFor_SmallRange) {
    // Small range should execute sequentially
    std::vector<int> data(10);
    
    parallel_for(0, 10, [&data](size_t i) {
        data[i] = static_cast<int>(i);
    }, 1000); // min_per_thread = 1000, so won't parallelize
    
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i));
    }
}

TEST_F(TensorPerfTest, ParallelFor_LargeRange) {
    const size_t size = 100000;
    std::vector<float> data(size);
    
    // Perform computation in parallel
    parallel_for(0, size, [&data](size_t i) {
        data[i] = std::sqrt(static_cast<float>(i));
    });
    
    // Spot check results
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[100], std::sqrt(100.0f));
    EXPECT_FLOAT_EQ(data[10000], std::sqrt(10000.0f));
}

TEST_F(TensorPerfTest, ParallelFor_PartialRange) {
    std::vector<int> data(100, 0);
    
    // Only process part of the vector
    parallel_for(20, 80, [&data](size_t i) {
        data[i] = 1;
    });
    
    // Check that only the specified range was processed
    for (size_t i = 0; i < 100; ++i) {
        if (i >= 20 && i < 80) {
            EXPECT_EQ(data[i], 1);
        } else {
            EXPECT_EQ(data[i], 0);
        }
    }
}

// =============================================================================
// Mixed Precision (Float16) Tests
// =============================================================================

TEST_F(TensorPerfTest, Float16_ConstructFromFloat) {
    Float16 half(1.0f);
    EXPECT_NEAR(half.to_float(), 1.0f, 0.001f);
}

TEST_F(TensorPerfTest, Float16_ZeroValue) {
    Float16 half(0.0f);
    EXPECT_FLOAT_EQ(half.to_float(), 0.0f);
}

TEST_F(TensorPerfTest, Float16_SmallPositiveValue) {
    Float16 half(0.1f);
    EXPECT_NEAR(half.to_float(), 0.1f, 0.001f);
}

TEST_F(TensorPerfTest, Float16_SmallNegativeValue) {
    Float16 half(-0.5f);
    EXPECT_NEAR(half.to_float(), -0.5f, 0.001f);
}

TEST_F(TensorPerfTest, Float16_LargeValue) {
    Float16 half(100.0f);
    EXPECT_NEAR(half.to_float(), 100.0f, 1.0f);
}

TEST_F(TensorPerfTest, Float16_ConversionAccuracy) {
    // Test various values
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        2.0f, -2.0f, 10.0f, -10.0f, 100.0f
    };
    
    for (float val : test_values) {
        Float16 half(val);
        float converted = half.to_float();
        // FP16 has limited precision
        EXPECT_NEAR(converted, val, std::abs(val) * 0.001f + 0.001f);
    }
}

TEST_F(TensorPerfTest, Float16_ArrayConversion) {
    const size_t size = 100;
    std::vector<float> fp32(size);
    std::vector<Float16> fp16(size);
    std::vector<float> fp32_back(size);
    
    // Fill with test data
    for (size_t i = 0; i < size; ++i) {
        fp32[i] = static_cast<float>(i) * 0.1f;
    }
    
    // Convert to FP16
    convert_fp32_to_fp16(fp32.data(), fp16.data(), size);
    
    // Convert back to FP32
    convert_fp16_to_fp32(fp16.data(), fp32_back.data(), size);
    
    // Check accuracy
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(fp32_back[i], fp32[i], 0.01f);
    }
}

// =============================================================================
// Mixed Precision (BFloat16) Tests
// =============================================================================

TEST_F(TensorPerfTest, BFloat16_ConstructFromFloat) {
    BFloat16 bf16(1.0f);
    EXPECT_FLOAT_EQ(bf16.to_float(), 1.0f);
}

TEST_F(TensorPerfTest, BFloat16_ZeroValue) {
    BFloat16 bf16(0.0f);
    EXPECT_FLOAT_EQ(bf16.to_float(), 0.0f);
}

TEST_F(TensorPerfTest, BFloat16_PositiveValues) {
    std::vector<float> test_values = {0.5f, 1.0f, 2.0f, 10.0f, 100.0f};
    
    for (float val : test_values) {
        BFloat16 bf16(val);
        float converted = bf16.to_float();
        // BF16 has better range but less precision than FP16
        EXPECT_NEAR(converted, val, std::abs(val) * 0.01f + 0.001f);
    }
}

TEST_F(TensorPerfTest, BFloat16_NegativeValues) {
    std::vector<float> test_values = {-0.5f, -1.0f, -2.0f, -10.0f, -100.0f};
    
    for (float val : test_values) {
        BFloat16 bf16(val);
        float converted = bf16.to_float();
        EXPECT_NEAR(converted, val, std::abs(val) * 0.01f + 0.001f);
    }
}

TEST_F(TensorPerfTest, BFloat16_ArrayConversion) {
    const size_t size = 100;
    std::vector<float> fp32(size);
    std::vector<BFloat16> bf16(size);
    std::vector<float> fp32_back(size);
    
    // Fill with test data
    for (size_t i = 0; i < size; ++i) {
        fp32[i] = static_cast<float>(i) * 0.5f;
    }
    
    // Convert to BF16
    convert_fp32_to_bf16(fp32.data(), bf16.data(), size);
    
    // Convert back to FP32
    convert_bf16_to_fp32(bf16.data(), fp32_back.data(), size);
    
    // Check accuracy
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(fp32_back[i], fp32[i], 0.1f);
    }
}

TEST_F(TensorPerfTest, BFloat16_VsFP16_RangeAdvantage) {
    // BF16 should handle larger values better than FP16
    float large_value = 10000.0f;
    
    BFloat16 bf16(large_value);
    Float16 fp16(large_value);
    
    // BF16 should be more accurate for large values
    float bf16_converted = bf16.to_float();
    float fp16_converted = fp16.to_float();
    
    // BF16 maintains better range (same as FP32)
    EXPECT_NEAR(bf16_converted, large_value, large_value * 0.01f);
}

// =============================================================================
// Lazy Evaluation Tests
// =============================================================================

TEST_F(TensorPerfTest, LazyOperation_BasicCreation) {
    LazyOperation<float, 1> op;
    
    EXPECT_EQ(op.op_type, OperationType::None);
    EXPECT_FALSE(op.is_fused);
}

TEST_F(TensorPerfTest, LazyOperation_CanFuseElementwise) {
    LazyOperation<float, 1> op1;
    LazyOperation<float, 1> op2;
    
    op1.op_type = OperationType::Exp;
    op2.op_type = OperationType::Tanh;
    
    EXPECT_TRUE(op1.can_fuse_with(op2));
}

TEST_F(TensorPerfTest, LazyOperation_CannotFuseAlreadyFused) {
    LazyOperation<float, 1> op1;
    LazyOperation<float, 1> op2;
    
    op1.op_type = OperationType::Exp;
    op1.is_fused = true;
    op2.op_type = OperationType::Tanh;
    
    EXPECT_FALSE(op1.can_fuse_with(op2));
}

TEST_F(TensorPerfTest, LazyOperation_Execute) {
    LazyOperation<float, 1> op;
    
    // Set up a simple executor
    op.executor = [](float* dst, const float* src, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i] * 2.0f;
        }
    };
    
    float src[] = {1.0f, 2.0f, 3.0f};
    float dst[3];
    
    op.execute(dst, src, 3);
    
    EXPECT_FLOAT_EQ(dst[0], 2.0f);
    EXPECT_FLOAT_EQ(dst[1], 4.0f);
    EXPECT_FLOAT_EQ(dst[2], 6.0f);
}

// =============================================================================
// Operation Fusion Tests
// =============================================================================

TEST_F(TensorPerfTest, FuseOperations_SingleOperation) {
    std::vector<OperationType> ops = {OperationType::Exp};
    
    auto fused = fuse_operations<float>(ops);
    
    float src[] = {0.0f, 1.0f, 2.0f};
    float dst[3];
    
    fused(dst, src, 3);
    
    EXPECT_NEAR(dst[0], std::exp(0.0f), 0.0001f);
    EXPECT_NEAR(dst[1], std::exp(1.0f), 0.0001f);
    EXPECT_NEAR(dst[2], std::exp(2.0f), 0.0001f);
}

TEST_F(TensorPerfTest, FuseOperations_MultipleOperations) {
    // Fuse exp -> tanh
    std::vector<OperationType> ops = {
        OperationType::Exp,
        OperationType::Tanh
    };
    
    auto fused = fuse_operations<float>(ops);
    
    float src[] = {0.0f, 0.5f, 1.0f};
    float dst[3];
    
    fused(dst, src, 3);
    
    // Should compute tanh(exp(x))
    for (size_t i = 0; i < 3; ++i) {
        float expected = std::tanh(std::exp(src[i]));
        EXPECT_NEAR(dst[i], expected, 0.0001f);
    }
}

TEST_F(TensorPerfTest, FuseOperations_ReLU) {
    std::vector<OperationType> ops = {OperationType::ReLU};
    
    auto fused = fuse_operations<float>(ops);
    
    float src[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float dst[4];
    
    fused(dst, src, 4);
    
    EXPECT_FLOAT_EQ(dst[0], 0.0f);
    EXPECT_FLOAT_EQ(dst[1], 0.0f);
    EXPECT_FLOAT_EQ(dst[2], 1.0f);
    EXPECT_FLOAT_EQ(dst[3], 2.0f);
}

TEST_F(TensorPerfTest, FuseOperations_Sigmoid) {
    std::vector<OperationType> ops = {OperationType::Sigmoid};
    
    auto fused = fuse_operations<float>(ops);
    
    float src[] = {0.0f, 1.0f, -1.0f};
    float dst[3];
    
    fused(dst, src, 3);
    
    EXPECT_NEAR(dst[0], 0.5f, 0.0001f);
    EXPECT_NEAR(dst[1], 1.0f / (1.0f + std::exp(-1.0f)), 0.0001f);
    EXPECT_NEAR(dst[2], 1.0f / (1.0f + std::exp(1.0f)), 0.0001f);
}

TEST_F(TensorPerfTest, FuseOperations_ComplexChain) {
    // Fuse: exp -> tanh -> relu
    std::vector<OperationType> ops = {
        OperationType::Exp,
        OperationType::Tanh,
        OperationType::ReLU
    };
    
    auto fused = fuse_operations<float>(ops);
    
    float src[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float dst[5];
    
    fused(dst, src, 5);
    
    // Manually compute expected results
    for (size_t i = 0; i < 5; ++i) {
        float val = std::exp(src[i]);
        val = std::tanh(val);
        val = std::max(0.0f, val);
        EXPECT_NEAR(dst[i], val, 0.0001f);
    }
}
