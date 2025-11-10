/**
 * @file tensor_perf_test.cc
 * @brief Tests for performance optimization features
 */

#include "tensor.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

// ============================================
// Memory Pool Tests
// ============================================

TEST(TensorPerfBasicTest, MemoryPoolBasic) {
    MemoryPool<float> pool;
    
    // Allocate memory
    float* ptr1 = pool.allocate(1000);
    ASSERT_NE(ptr1, nullptr);
    
    // Fill with data
    for (size_t i = 0; i < 1000; ++i) {
        ptr1[i] = static_cast<float>(i);
    }
    
    // Deallocate
    pool.deallocate(ptr1, 1000);
    
    // Allocate again - should reuse the same block
    float* ptr2 = pool.allocate(1000);
    ASSERT_NE(ptr2, nullptr);
    
    pool.deallocate(ptr2, 1000);
}

TEST(TensorPerfBasicTest, MemoryPoolReuse) {
    MemoryPool<double> pool;
    
    std::vector<double*> ptrs;
    
    // Allocate multiple blocks
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(pool.allocate(100));
    }
    
    // Deallocate all
    for (auto ptr : ptrs) {
        pool.deallocate(ptr, 100);
    }
    
    auto stats = pool.stats();
    EXPECT_EQ(stats.first, 10);   // 10 allocations
    EXPECT_EQ(stats.second, 10);  // 10 deallocations
    
    // Allocate again - should reuse from pool
    double* new_ptr = pool.allocate(100);
    ASSERT_NE(new_ptr, nullptr);
    
    pool.deallocate(new_ptr, 100);
}

TEST(TensorPerfBasicTest, MemoryPoolClear) {
    MemoryPool<int> pool;
    
    int* ptr1 = pool.allocate(500);
    pool.deallocate(ptr1, 500);
    
    int* ptr2 = pool.allocate(300);
    pool.deallocate(ptr2, 300);
    
    // Clear the pool
    pool.clear();
    
    // Allocate after clear - will allocate new memory
    int* ptr3 = pool.allocate(200);
    ASSERT_NE(ptr3, nullptr);
    
    pool.deallocate(ptr3, 200);
}

TEST(TensorPerfBasicTest, GlobalMemoryPool) {
    auto& pool1 = get_memory_pool<float>();
    auto& pool2 = get_memory_pool<float>();
    
    // Should be the same instance
    EXPECT_EQ(&pool1, &pool2);
    
    float* ptr = pool1.allocate(100);
    ASSERT_NE(ptr, nullptr);
    
    pool2.deallocate(ptr, 100);
}

// ============================================
// Thread Pool Tests
// ============================================

TEST(TensorPerfBasicTest, ThreadPoolBasic) {
    ThreadPool pool(4);
    
    auto future = pool.enqueue([]() {
        return 42;
    });
    
    EXPECT_EQ(future.get(), 42);
}

TEST(TensorPerfBasicTest, ThreadPoolMultipleTasks) {
    ThreadPool pool(4);
    
    std::vector<std::future<int>> futures;
    
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.enqueue([i]() {
            return i * i;
        }));
    }
    
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST(TensorPerfBasicTest, ThreadPoolSize) {
    ThreadPool pool(8);
    EXPECT_EQ(pool.size(), 8);
}

TEST(TensorPerfBasicTest, GlobalThreadPool) {
    auto& pool1 = get_thread_pool();
    auto& pool2 = get_thread_pool();
    
    // Should be the same instance
    EXPECT_EQ(&pool1, &pool2);
}

TEST(TensorPerfBasicTest, ParallelFor) {
    const size_t size = 10000;
    std::vector<int> data(size, 0);
    
    parallel_for(0, size, [&](size_t i) {
        data[i] = static_cast<int>(i * 2);
    });
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i * 2));
    }
}

TEST(TensorPerfBasicTest, ParallelForSmallRange) {
    // Small range should execute serially
    std::vector<int> data(10, 0);
    
    parallel_for(0, 10, [&](size_t i) {
        data[i] = static_cast<int>(i);
    });
    
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i));
    }
}

// ============================================
// Mixed Precision Tests
// ============================================

TEST(TensorPerfBasicTest, Float16Construction) {
    Float16 half(3.14f);
    float result = half.to_float();
    
    // FP16 has limited precision
    EXPECT_NEAR(result, 3.14f, 0.01f);
}

TEST(TensorPerfBasicTest, Float16Conversion) {
    float original = 42.5f;
    Float16 half(original);
    float converted = static_cast<float>(half);
    
    EXPECT_NEAR(converted, original, 0.1f);
}

TEST(TensorPerfBasicTest, Float16Zero) {
    Float16 half(0.0f);
    EXPECT_EQ(half.to_float(), 0.0f);
}

TEST(TensorPerfBasicTest, Float16Negative) {
    Float16 half(-10.5f);
    EXPECT_NEAR(half.to_float(), -10.5f, 0.1f);
}

TEST(TensorPerfBasicTest, BFloat16Construction) {
    BFloat16 bf16(3.14f);
    float result = bf16.to_float();
    
    // BF16 has same range as FP32 but less precision
    EXPECT_NEAR(result, 3.14f, 0.02f);
}

TEST(TensorPerfBasicTest, BFloat16Conversion) {
    float original = 123.456f;
    BFloat16 bf16(original);
    float converted = static_cast<float>(bf16);
    
    EXPECT_NEAR(converted, original, 0.5f);
}

TEST(TensorPerfBasicTest, BFloat16Zero) {
    BFloat16 bf16(0.0f);
    EXPECT_EQ(bf16.to_float(), 0.0f);
}

TEST(TensorPerfBasicTest, BFloat16Negative) {
    BFloat16 bf16(-99.9f);
    EXPECT_NEAR(bf16.to_float(), -99.9f, 0.5f);
}

TEST(TensorPerfBasicTest, ArrayConversionFP16) {
    const size_t count = 100;
    float src[count];
    Float16 mid[count];
    float dst[count];
    
    // Initialize source
    for (size_t i = 0; i < count; ++i) {
        src[i] = static_cast<float>(i) * 0.1f;
    }
    
    // Convert to FP16
    convert_fp32_to_fp16(src, mid, count);
    
    // Convert back to FP32
    convert_fp16_to_fp32(mid, dst, count);
    
    // Check roundtrip
    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(dst[i], src[i], 0.05f);
    }
}

TEST(TensorPerfBasicTest, ArrayConversionBF16) {
    const size_t count = 100;
    float src[count];
    BFloat16 mid[count];
    float dst[count];
    
    // Initialize source
    for (size_t i = 0; i < count; ++i) {
        src[i] = static_cast<float>(i) * 10.0f;
    }
    
    // Convert to BF16
    convert_fp32_to_bf16(src, mid, count);
    
    // Convert back to FP32
    convert_bf16_to_fp32(mid, dst, count);
    
    // Check roundtrip - BF16 has less precision, especially at larger values
    for (size_t i = 0; i < count; ++i) {
        // BF16 has 7 bits of mantissa, roughly 1 part in 128 precision
        float tolerance = std::max(4.0f, src[i] * 0.02f);
        EXPECT_NEAR(dst[i], src[i], tolerance);
    }
}

// ============================================
// Lazy Evaluation Tests
// ============================================

TEST(TensorPerfBasicTest, LazyOperationBasic) {
    LazyOperation<float, 2> op;
    
    op.op_type = OperationType::Add;
    EXPECT_EQ(op.op_type, OperationType::Add);
    EXPECT_FALSE(op.is_fused);
}

TEST(TensorPerfBasicTest, LazyOperationCanFuse) {
    LazyOperation<float, 2> op1;
    op1.op_type = OperationType::Add;
    
    LazyOperation<float, 2> op2;
    op2.op_type = OperationType::Multiply;
    
    EXPECT_TRUE(op1.can_fuse_with(op2));
}

TEST(TensorPerfBasicTest, LazyOperationCannotFuseIfFused) {
    LazyOperation<float, 2> op1;
    op1.op_type = OperationType::Add;
    op1.is_fused = true;
    
    LazyOperation<float, 2> op2;
    op2.op_type = OperationType::Multiply;
    
    EXPECT_FALSE(op1.can_fuse_with(op2));
}

TEST(TensorPerfBasicTest, FuseOperationsExp) {
    std::vector<OperationType> ops = {OperationType::Exp};
    auto fused = fuse_operations<float>(ops);
    
    const size_t size = 5;
    float src[size] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    float dst[size];
    
    fused(dst, src, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(dst[i], std::exp(src[i]), 1e-5f);
    }
}

TEST(TensorPerfBasicTest, FuseOperationsTanh) {
    std::vector<OperationType> ops = {OperationType::Tanh};
    auto fused = fuse_operations<float>(ops);
    
    const size_t size = 5;
    float src[size] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float dst[size];
    
    fused(dst, src, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(dst[i], std::tanh(src[i]), 1e-5f);
    }
}

TEST(TensorPerfBasicTest, FuseOperationsSigmoid) {
    std::vector<OperationType> ops = {OperationType::Sigmoid};
    auto fused = fuse_operations<float>(ops);
    
    const size_t size = 5;
    float src[size] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float dst[size];
    
    fused(dst, src, size);
    
    for (size_t i = 0; i < size; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-src[i]));
        EXPECT_NEAR(dst[i], expected, 1e-5f);
    }
}

TEST(TensorPerfBasicTest, FuseOperationsReLU) {
    std::vector<OperationType> ops = {OperationType::ReLU};
    auto fused = fuse_operations<float>(ops);
    
    const size_t size = 5;
    float src[size] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float dst[size];
    
    fused(dst, src, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(dst[i], std::max(0.0f, src[i]));
    }
}

TEST(TensorPerfBasicTest, FuseMultipleOperations) {
    // Test exp followed by tanh
    std::vector<OperationType> ops = {OperationType::Exp, OperationType::Tanh};
    auto fused = fuse_operations<float>(ops);
    
    const size_t size = 3;
    float src[size] = {0.0f, 1.0f, 2.0f};
    float dst[size];
    
    fused(dst, src, size);
    
    for (size_t i = 0; i < size; ++i) {
        float expected = std::tanh(std::exp(src[i]));
        EXPECT_NEAR(dst[i], expected, 1e-4f);
    }
}

// ============================================
// Integration Tests
// ============================================

TEST(TensorPerfBasicTest, MemoryPoolThreadSafety) {
    MemoryPool<float> pool;
    const int num_threads = 8;
    const int allocations_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool, allocations_per_thread]() {
            std::vector<float*> ptrs;
            
            // Allocate
            for (int i = 0; i < allocations_per_thread; ++i) {
                ptrs.push_back(pool.allocate(100));
            }
            
            // Use memory
            for (auto ptr : ptrs) {
                for (int i = 0; i < 100; ++i) {
                    ptr[i] = static_cast<float>(i);
                }
            }
            
            // Deallocate
            for (auto ptr : ptrs) {
                pool.deallocate(ptr, 100);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto stats = pool.stats();
    EXPECT_EQ(stats.first, num_threads * allocations_per_thread);
    EXPECT_EQ(stats.second, num_threads * allocations_per_thread);
}

TEST(TensorPerfBasicTest, ParallelForPerformance) {
    const size_t size = 1000000;
    std::vector<double> data(size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    parallel_for(0, size, [&](size_t i) {
        data[i] = std::sqrt(static_cast<double>(i));
    });
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify results
    EXPECT_NEAR(data[0], 0.0, 1e-10);
    EXPECT_NEAR(data[100], 10.0, 1e-10);
    EXPECT_NEAR(data[10000], 100.0, 1e-10);
    
    // Just check that it completed
    EXPECT_LT(duration.count(), 5000); // Should complete in less than 5 seconds
}
