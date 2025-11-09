/**
 * @file tensor_perf.h
 * @brief Performance optimization features for tensor operations
 * 
 * This header provides advanced performance optimization features:
 * - Memory pooling for reduced allocation overhead
 * - Multi-threading utilities for CPU operations
 * - Mixed precision support (FP16, BF16)
 * - Lazy evaluation for operation fusion
 * 
 * @author Tensor Library Team
 * @version 1.0
 * @date 2024
 */

#ifndef _TENSOR_PERF_H
#define _TENSOR_PERF_H

#include <memory>
#include <vector>
#include <mutex>
#include <thread>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <queue>

// ============================================
// Memory Pool for Tensor Allocations
// ============================================

/**
 * @class MemoryPool
 * @brief Thread-safe memory pool for efficient tensor allocation
 * 
 * Reduces allocation overhead by reusing previously allocated memory blocks.
 * Useful for repeated tensor operations with similar sizes.
 * 
 * @tparam T Data type to allocate
 * 
 * @code
 * MemoryPool<float> pool;
 * auto ptr = pool.allocate(1000);  // Allocate 1000 floats
 * // ... use memory ...
 * pool.deallocate(ptr, 1000);      // Return to pool for reuse
 * @endcode
 */
template<typename T>
class MemoryPool {
private:
    struct Block {
        T* data;
        size_t size;
        
        Block(T* d, size_t s) : data(d), size(s) {}
    };
    
    std::vector<Block> free_blocks_;
    std::mutex mutex_;
    size_t total_allocated_ = 0;
    size_t total_freed_ = 0;
    
public:
    /**
     * @brief Construct a new Memory Pool
     */
    MemoryPool() = default;
    
    /**
     * @brief Destructor - frees all pooled memory
     */
    ~MemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& block : free_blocks_) {
            delete[] block.data;
        }
    }
    
    /**
     * @brief Allocate memory from pool
     * @param size Number of elements to allocate
     * @return Pointer to allocated memory
     * 
     * Attempts to reuse a pooled block. If no suitable block is found,
     * allocates new memory.
     */
    T* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try to find a suitable block from the pool
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->size >= size && it->size < size * 2) {
                T* data = it->data;
                free_blocks_.erase(it);
                total_allocated_++;
                return data;
            }
        }
        
        // No suitable block found, allocate new
        total_allocated_++;
        return new T[size];
    }
    
    /**
     * @brief Return memory to pool for reuse
     * @param ptr Pointer to memory to return
     * @param size Size of the memory block
     * 
     * Returned memory is kept in the pool for future reuse.
     */
    void deallocate(T* ptr, size_t size) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        total_freed_++;
        
        // Keep pool size reasonable - limit to 100 blocks
        if (free_blocks_.size() < 100) {
            free_blocks_.emplace_back(ptr, size);
        } else {
            delete[] ptr;
        }
    }
    
    /**
     * @brief Clear all pooled memory
     * 
     * Frees all blocks in the pool. Useful for memory management.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& block : free_blocks_) {
            delete[] block.data;
        }
        free_blocks_.clear();
    }
    
    /**
     * @brief Get statistics about memory pool usage
     * @return Pair of (total_allocated, total_freed)
     */
    std::pair<size_t, size_t> stats() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        return {total_allocated_, total_freed_};
    }
};

/**
 * @brief Get global memory pool for a specific type
 * @tparam T Data type
 * @return Reference to the global memory pool for type T
 * 
 * Provides a singleton memory pool instance per type.
 */
template<typename T>
MemoryPool<T>& get_memory_pool() {
    static MemoryPool<T> pool;
    return pool;
}

// ============================================
// Thread Pool for Parallel Operations
// ============================================

/**
 * @class ThreadPool
 * @brief Thread pool for parallel tensor operations
 * 
 * Manages a pool of worker threads for efficient parallel execution.
 * Useful for parallelizing CPU-based tensor operations.
 * 
 * @code
 * auto& pool = get_thread_pool();
 * auto future = pool.enqueue([](){ return 42; });
 * int result = future.get();
 * @endcode
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
    
public:
    /**
     * @brief Construct thread pool with specified number of threads
     * @param num_threads Number of worker threads (default: hardware concurrency)
     */
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    
    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    /**
     * @brief Enqueue a task for execution
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to function
     * @return Future for retrieving the result
     */
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("Enqueue on stopped ThreadPool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return res;
    }
    
    /**
     * @brief Get number of worker threads
     * @return Number of threads in the pool
     */
    size_t size() const {
        return workers_.size();
    }
};

/**
 * @brief Get global thread pool instance
 * @return Reference to the global thread pool
 */
inline ThreadPool& get_thread_pool() {
    static ThreadPool pool;
    return pool;
}

/**
 * @brief Parallel for loop using thread pool
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to execute for each index
 * @param min_per_thread Minimum iterations per thread (default: 1000)
 * 
 * Divides work among threads in the thread pool.
 * 
 * @code
 * parallel_for(0, 1000000, [&](size_t i) {
 *     data[i] = compute(i);
 * });
 * @endcode
 */
inline void parallel_for(size_t start, size_t end, 
                        std::function<void(size_t)> func,
                        size_t min_per_thread = 1000) {
    size_t range = end - start;
    if (range < min_per_thread) {
        // Not worth parallelizing
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    auto& pool = get_thread_pool();
    size_t num_threads = std::min(range / min_per_thread, pool.size());
    
    if (num_threads <= 1) {
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    size_t chunk_size = range / num_threads;
    std::vector<std::future<void>> futures;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t chunk_start = start + t * chunk_size;
        size_t chunk_end = (t == num_threads - 1) ? end : chunk_start + chunk_size;
        
        futures.push_back(pool.enqueue([chunk_start, chunk_end, &func]() {
            for (size_t i = chunk_start; i < chunk_end; ++i) {
                func(i);
            }
        }));
    }
    
    for (auto& f : futures) {
        f.get();
    }
}

// ============================================
// Mixed Precision Support (FP16, BF16)
// ============================================

/**
 * @struct Float16
 * @brief IEEE 754 half-precision floating-point (FP16)
 * 
 * 16-bit floating point: 1 sign bit, 5 exponent bits, 10 mantissa bits
 * Range: ~5.96e-8 to 65504
 * Useful for memory-efficient neural networks.
 */
struct Float16 {
    uint16_t data;
    
    Float16() : data(0) {}
    
    /**
     * @brief Construct from float
     * @param value Float value to convert
     */
    explicit Float16(float value) {
        uint32_t f;
        std::memcpy(&f, &value, sizeof(float));
        
        uint32_t sign = (f >> 16) & 0x8000;
        int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = f & 0x7FFFFF;
        
        if (exponent <= 0) {
            // Subnormal or zero
            data = sign;
        } else if (exponent >= 31) {
            // Infinity or overflow
            data = sign | 0x7C00;
        } else {
            // Normal number
            data = sign | (exponent << 10) | (mantissa >> 13);
        }
    }
    
    /**
     * @brief Convert to float
     * @return Float representation
     */
    float to_float() const {
        uint32_t sign = (data & 0x8000) << 16;
        uint32_t exponent = (data & 0x7C00) >> 10;
        uint32_t mantissa = (data & 0x3FF) << 13;
        
        uint32_t f;
        if (exponent == 0) {
            // Subnormal or zero
            f = sign;
        } else if (exponent == 31) {
            // Infinity
            f = sign | 0x7F800000 | mantissa;
        } else {
            // Normal number
            f = sign | ((exponent - 15 + 127) << 23) | mantissa;
        }
        
        float result;
        std::memcpy(&result, &f, sizeof(float));
        return result;
    }
    
    operator float() const { return to_float(); }
};

/**
 * @struct BFloat16
 * @brief Brain floating-point (BF16)
 * 
 * 16-bit floating point: 1 sign bit, 8 exponent bits, 7 mantissa bits
 * Range: same as FP32 (~1.4e-45 to 3.4e38)
 * Better for ML training than FP16 due to wider range.
 */
struct BFloat16 {
    uint16_t data;
    
    BFloat16() : data(0) {}
    
    /**
     * @brief Construct from float
     * @param value Float value to convert
     */
    explicit BFloat16(float value) {
        uint32_t f;
        std::memcpy(&f, &value, sizeof(float));
        
        // BF16 is simply the top 16 bits of FP32
        // with optional rounding
        uint32_t rounding = (f >> 16) & 1;  // Round to nearest even
        data = (f >> 16) + rounding;
    }
    
    /**
     * @brief Convert to float
     * @return Float representation
     */
    float to_float() const {
        uint32_t f = static_cast<uint32_t>(data) << 16;
        float result;
        std::memcpy(&result, &f, sizeof(float));
        return result;
    }
    
    operator float() const { return to_float(); }
};

/**
 * @brief Convert array from FP32 to FP16
 * @param src Source FP32 array
 * @param dst Destination FP16 array
 * @param count Number of elements
 */
inline void convert_fp32_to_fp16(const float* src, Float16* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = Float16(src[i]);
    }
}

/**
 * @brief Convert array from FP16 to FP32
 * @param src Source FP16 array
 * @param dst Destination FP32 array
 * @param count Number of elements
 */
inline void convert_fp16_to_fp32(const Float16* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = src[i].to_float();
    }
}

/**
 * @brief Convert array from FP32 to BF16
 * @param src Source FP32 array
 * @param dst Destination BF16 array
 * @param count Number of elements
 */
inline void convert_fp32_to_bf16(const float* src, BFloat16* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = BFloat16(src[i]);
    }
}

/**
 * @brief Convert array from BF16 to FP32
 * @param src Source BF16 array
 * @param dst Destination FP32 array
 * @param count Number of elements
 */
inline void convert_bf16_to_fp32(const BFloat16* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = src[i].to_float();
    }
}

// ============================================
// Lazy Evaluation Support
// ============================================

/**
 * @enum OperationType
 * @brief Types of deferred operations for lazy evaluation
 */
enum class OperationType {
    None,
    Add,
    Subtract,
    Multiply,
    Divide,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    ReLU
};

/**
 * @class LazyOperation
 * @brief Represents a deferred tensor operation
 * 
 * Stores operation metadata for later fusion and execution.
 * This enables operation fusion optimizations.
 * 
 * @tparam T Data type
 * @tparam N Number of dimensions
 */
template<typename T, size_t N>
class LazyOperation {
public:
    OperationType op_type;
    std::function<void(T*, const T*, size_t)> executor;
    bool is_fused;
    
    LazyOperation() : op_type(OperationType::None), is_fused(false) {}
    
    /**
     * @brief Check if operation can be fused with another
     * @param other The other operation
     * @return true if operations can be fused
     */
    bool can_fuse_with(const LazyOperation& other) const {
        // Simple element-wise operations can be fused
        return !is_fused && !other.is_fused &&
               (op_type == OperationType::Add || 
                op_type == OperationType::Multiply ||
                op_type == OperationType::Exp ||
                op_type == OperationType::Tanh);
    }
    
    /**
     * @brief Execute the lazy operation
     * @param dst Destination array
     * @param src Source array
     * @param size Number of elements
     */
    void execute(T* dst, const T* src, size_t size) {
        if (executor) {
            executor(dst, src, size);
        }
    }
};

/**
 * @brief Create fused element-wise operation
 * @tparam T Data type
 * @param ops Vector of operations to fuse
 * @return Fused operation function
 * 
 * Combines multiple element-wise operations into a single pass
 * to improve cache utilization and reduce memory bandwidth.
 */
template<typename T>
std::function<void(T*, const T*, size_t)> 
fuse_operations(const std::vector<OperationType>& ops) {
    return [ops](T* dst, const T* src, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            T val = src[i];
            for (auto op : ops) {
                switch (op) {
                    case OperationType::Exp:
                        val = std::exp(val);
                        break;
                    case OperationType::Tanh:
                        val = std::tanh(val);
                        break;
                    case OperationType::Sigmoid:
                        val = T(1) / (T(1) + std::exp(-val));
                        break;
                    case OperationType::ReLU:
                        val = std::max(T(0), val);
                        break;
                    default:
                        break;
                }
            }
            dst[i] = val;
        }
    };
}

#endif // _TENSOR_PERF_H
