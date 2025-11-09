#include "tensor.h"
#include "linalg.h"
#include "loss_functions.h"
#include "optimizers.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <functional>
#include <fstream>
#include <cmath>
#include <vector>

// ============================================
// Benchmark Configuration Parameters
// ============================================
#define ITERATIONS 20

// Vector dot product sizes
constexpr size_t VECTOR_SIZES[] = {100, 1000, 10000, 100000};
constexpr size_t NUM_VECTOR_SIZES = sizeof(VECTOR_SIZES) / sizeof(VECTOR_SIZES[0]);

// Matrix multiplication configurations
struct MatrixConfig {
    size_t m, n, p;
    const char* description;
};

constexpr MatrixConfig MATRIX_CONFIGS[] = {
    {50, 50, 50, "Small square (50x50 * 50x50)"},
    {100, 100, 100, "Medium square (100x100 * 100x100)"},
    {200, 200, 200, "Large square (200x200 * 200x200)"},
    {100, 1000, 100, "Tall rectangle (100x1000 * 1000x100)"},
    {1000, 100, 1000, "Wide rectangle (1000x100 * 100x1000)"},
};
constexpr size_t NUM_MATRIX_CONFIGS = sizeof(MATRIX_CONFIGS) / sizeof(MATRIX_CONFIGS[0]);

// Tensor construction sizes
constexpr size_t TENSOR_1D_SIZE = 1000000;
constexpr size_t TENSOR_2D_SIZE = 1000;
constexpr size_t TENSOR_3D_SIZE = 100;
constexpr size_t TENSOR_4D_SIZE_1 = 50;
constexpr size_t TENSOR_4D_SIZE_2 = 20;

// Number of iterations for different benchmark types
constexpr int ITERATIONS_FAST = ITERATIONS;        // For fast operations
constexpr int ITERATIONS_MEDIUM = ITERATIONS / 5;   // For medium operations (matrices)
constexpr int ITERATIONS_SLOW = ITERATIONS / 10;    // For slow operations (large tensors)

using namespace std::chrono;

// Structure to hold benchmark results
struct BenchmarkResult {
    std::string name;
    std::string category;
    double mean_ms;
    double std_dev_ms;
    double min_ms;
    double max_ms;
    int iterations;
    size_t total_operations;
};

// Global vector to store all results
std::vector<BenchmarkResult> all_results;

// Helper to generate random tensor
template<typename T, size_t N>
void fill_random(Tensor<T, N>& tensor, T min_val = 0.0, T max_val = 1.0) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min_val, max_val);
    
    auto dims = tensor.dims();
    std::function<void(size_t, TensorIndices<N>&)> fill_recursive;
    
    fill_recursive = [&](size_t depth, TensorIndices<N>& indices) {
        if (depth == N) {
            tensor[indices] = dis(gen);
            return;
        }
        for (size_t i = 0; i < dims[depth]; ++i) {
            indices[depth] = i;
            fill_recursive(depth + 1, indices);
        }
    };
    
    TensorIndices<N> indices;
    fill_recursive(0, indices);
}

// Template specialization for integers
template<>
void fill_random<int, 1>(Tensor<int, 1>& tensor, int min_val, int max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min_val, max_val);
    
    auto dims = tensor.dims();
    for (size_t i = 0; i < dims[0]; ++i) {
        tensor[{i}] = dis(gen);
    }
}

template<>
void fill_random<int, 2>(Tensor<int, 2>& tensor, int min_val, int max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min_val, max_val);
    
    auto dims = tensor.dims();
    for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
            tensor[{i, j}] = dis(gen);
        }
    }
}

// Specializations for float tensors of specific dimensions
template<>
void fill_random<float, 1>(Tensor<float, 1>& tensor, float min_val, float max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    auto dims = tensor.dims();
    for (size_t i = 0; i < dims[0]; ++i) {
        tensor[{i}] = dis(gen);
    }
}

template<>
void fill_random<float, 2>(Tensor<float, 2>& tensor, float min_val, float max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    auto dims = tensor.dims();
    for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
            tensor[{i, j}] = dis(gen);
        }
    }
}

template<>
void fill_random<float, 3>(Tensor<float, 3>& tensor, float min_val, float max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    auto dims = tensor.dims();
    for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
            for (size_t k = 0; k < dims[2]; ++k) {
                tensor[{i, j, k}] = dis(gen);
            }
        }
    }
}

template<>
void fill_random<float, 4>(Tensor<float, 4>& tensor, float min_val, float max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    auto dims = tensor.dims();
    for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
            for (size_t k = 0; k < dims[2]; ++k) {
                for (size_t l = 0; l < dims[3]; ++l) {
                    tensor[{i, j, k, l}] = dis(gen);
                }
            }
        }
    }
}

template<>
void fill_random<float, 5>(Tensor<float, 5>& tensor, float min_val, float max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    auto dims = tensor.dims();
    for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
            for (size_t k = 0; k < dims[2]; ++k) {
                for (size_t l = 0; l < dims[3]; ++l) {
                    for (size_t m = 0; m < dims[4]; ++m) {
                        tensor[{i, j, k, l, m}] = dis(gen);
                    }
                }
            }
        }
    }
}

// Benchmark helper
template<typename Func>
BenchmarkResult benchmark(const std::string& name, const std::string& category, 
                          Func&& func, int iterations = 5, size_t total_ops = 0) {
    std::cout << "Running: " << name << " (" << iterations << " iterations)" << std::endl;
    
    // Warmup
    func();
    
    // Collect timing data
    std::vector<double> times;
    times.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(time_ms);
    }
    
    // Calculate statistics
    double sum = 0.0;
    double min_time = times[0];
    double max_time = times[0];
    
    for (double t : times) {
        sum += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    
    double mean = sum / iterations;
    
    // Calculate standard deviation
    double variance_sum = 0.0;
    for (double t : times) {
        double diff = t - mean;
        variance_sum += diff * diff;
    }
    double std_dev = std::sqrt(variance_sum / iterations);
    
    // Display results
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms"
              << " ± " << std::setprecision(3) << std_dev << " ms" << std::endl;
    std::cout << "  Min: " << std::setprecision(3) << min_time << " ms"
              << ", Max: " << std::setprecision(3) << max_time << " ms" << std::endl;
    
    // Create result object
    BenchmarkResult result;
    result.name = name;
    result.category = category;
    result.mean_ms = mean;
    result.std_dev_ms = std_dev;
    result.min_ms = min_time;
    result.max_ms = max_time;
    result.iterations = iterations;
    result.total_operations = total_ops;
    
    all_results.push_back(result);
    
    return result;
}

void benchmark_vector_dot() {
    std::cout << "\n=== 1D Vector Dot Product Benchmark ===" << std::endl;
    
    for (size_t i = 0; i < NUM_VECTOR_SIZES; ++i) {
        size_t size = VECTOR_SIZES[i];
        Tensor<float, 1> a({size});
        Tensor<float, 1> b({size});
        fill_random(a);
        fill_random(b);
        
        std::string name = "Vector size " + std::to_string(size);
        benchmark(name, "1D_Dot_Product", [&]() {
            auto result = a.dot(b);
            // Force computation
            if (std::holds_alternative<float>(result)) {
                volatile float val = std::get<float>(result);
                (void)val;
            }
        }, ITERATIONS_FAST, size);
    }
}

void benchmark_matrix_multiplication() {
    std::cout << "\n=== 2D Matrix Multiplication Benchmark ===" << std::endl;
    
    for (size_t i = 0; i < NUM_MATRIX_CONFIGS; ++i) {
        const auto& config = MATRIX_CONFIGS[i];
        Tensor<float, 2> a({config.m, config.n});
        Tensor<float, 2> b({config.n, config.p});
        fill_random(a);
        fill_random(b);
        
        size_t total_ops = config.m * config.n * config.p;
        benchmark(config.description, "2D_Matrix_Multiplication", [&]() {
            auto result = a.dot(b);
            // Force computation
            if (std::holds_alternative<Tensor<float, 2>>(result)) {
                volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
                (void)val;
            }
        }, ITERATIONS_MEDIUM, total_ops);
    }
}

void benchmark_arithmetic_operations() {
    std::cout << "\n=== Arithmetic Operations Benchmark ===" << std::endl;
    
    constexpr size_t size = 500;
    Tensor<float, 2> a({size, size}, false);
    Tensor<float, 2> b({size, size}, false);
    fill_random(a);
    fill_random(b);
    
    benchmark("Element-wise addition (500x500)", "Arithmetic", [&]() {
        auto result_var = a + b;
        if (std::holds_alternative<Tensor<float, 2>>(result_var)) {
            auto& result = std::get<Tensor<float, 2>>(result_var);
            volatile float val = result[{0, 0}];
            (void)val;
        }
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Element-wise subtraction (500x500)", "Arithmetic", [&]() {
        auto result_var = a - b;
        if (std::holds_alternative<Tensor<float, 2>>(result_var)) {
            auto& result = std::get<Tensor<float, 2>>(result_var);
            volatile float val = result[{0, 0}];
            (void)val;
        }
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Element-wise multiplication (500x500)", "Arithmetic", [&]() {
        auto result_var = a * b;
        if (std::holds_alternative<Tensor<float, 2>>(result_var)) {
            auto& result = std::get<Tensor<float, 2>>(result_var);
            volatile float val = result[{0, 0}];
            (void)val;
        }
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Scalar addition (500x500)", "Arithmetic", [&]() {
        auto result = a + 5.0f;
        volatile float val = result[{0, 0}];
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Scalar multiplication (500x500)", "Arithmetic", [&]() {
        auto result = a * 2.5f;
        volatile float val = result[{0, 0}];
        (void)val;
    }, ITERATIONS_FAST, size * size);
}

void benchmark_math_functions() {
    std::cout << "\n=== Math Functions Benchmark ===" << std::endl;
    
    constexpr size_t size = 500;
    Tensor<float, 2> t({size, size}, false);
    fill_random(t, 0.1f, 2.0f);
    
    benchmark("Exponential (500x500)", "Math_Functions", [&]() {
        auto result = t.exp();
        volatile float val = result[{0, 0}];
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Natural log (500x500)", "Math_Functions", [&]() {
        auto result = t.log();
        volatile float val = result[{0, 0}];
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Sigmoid (500x500)", "Math_Functions", [&]() {
        auto result = t.sigmoid();
        volatile float val = result[{0, 0}];
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Tanh (500x500)", "Math_Functions", [&]() {
        auto result = t.tanh();
        volatile float val = result[{0, 0}];
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("ReLU (500x500)", "Math_Functions", [&]() {
        auto result = t.relu();
        volatile float val = result[{0, 0}];
        (void)val;
    }, ITERATIONS_FAST, size * size);
}

void benchmark_reduction_operations() {
    std::cout << "\n=== Reduction Operations Benchmark ===" << std::endl;
    
    constexpr size_t size = 500;
    Tensor<float, 2> t({size, size}, false);
    fill_random(t);
    
    benchmark("Sum all elements (500x500)", "Reductions", [&]() {
        float result = t.sum();
        volatile float val = result;
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Mean all elements (500x500)", "Reductions", [&]() {
        float result = t.mean();
        volatile float val = result;
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Min all elements (500x500)", "Reductions", [&]() {
        float result = t.min();
        volatile float val = result;
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Max all elements (500x500)", "Reductions", [&]() {
        float result = t.max();
        volatile float val = result;
        (void)val;
    }, ITERATIONS_FAST, size * size);
}

void benchmark_statistical_operations() {
    std::cout << "\n=== Statistical Operations Benchmark ===" << std::endl;
    
    constexpr size_t size = 500;
    Tensor<float, 2> t({size, size}, false);
    fill_random(t);
    
    benchmark("Variance (500x500)", "Statistics", [&]() {
        float result = t.variance();
        volatile float val = result;
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    benchmark("Standard deviation (500x500)", "Statistics", [&]() {
        float result = t.std();
        volatile float val = result;
        (void)val;
    }, ITERATIONS_FAST, size * size);
    
    // Correlation between two tensors
    Tensor<float, 1> v1({1000}, false);
    Tensor<float, 1> v2({1000}, false);
    fill_random(v1);
    fill_random(v2);
    
    benchmark("Correlation (1000 elements)", "Statistics", [&]() {
        auto result_var = v1.correlation(v2);
        if (std::holds_alternative<float>(result_var)) {
            volatile float val = std::get<float>(result_var);
            (void)val;
        }
    }, ITERATIONS_FAST, 1000);
    
    benchmark("Covariance (1000 elements)", "Statistics", [&]() {
        auto result_var = v1.covariance(v2);
        if (std::holds_alternative<float>(result_var)) {
            volatile float val = std::get<float>(result_var);
            (void)val;
        }
    }, ITERATIONS_FAST, 1000);
}

void benchmark_autograd_operations() {
    std::cout << "\n=== Autograd Operations Benchmark ===" << std::endl;
    
    // Simple scalar operations
    {
        Tensor<float, 1> x({1}, false, true);
        x[{0}] = 3.0f;
        
        benchmark("Autograd: Forward pass (scalar addition)", "Autograd", [&]() {
            auto y = x + 2.0f;
            volatile float val = y[{0}];
            (void)val;
        }, ITERATIONS_FAST, 1);
        
        benchmark("Autograd: Forward pass (scalar multiply)", "Autograd", [&]() {
            auto y = x * 2.0f;
            volatile float val = y[{0}];
            (void)val;
        }, ITERATIONS_FAST, 1);
        
        benchmark("Autograd: Forward + Backward (scalar)", "Autograd", [&]() {
            x.zero_grad();
            auto y_var = x * x;
            if (std::holds_alternative<Tensor<float, 1>>(y_var)) {
                auto y = std::get<Tensor<float, 1>>(y_var);
                y.backward();
            }
        }, ITERATIONS_FAST, 1);
    }
    
    // Matrix operations with autograd
    {
        constexpr size_t size = 100;
        Tensor<float, 2> a({size, size}, false, true);
        Tensor<float, 2> b({size, size}, false, true);
        fill_random(a);
        fill_random(b);
        
        benchmark("Autograd: Matrix addition forward (100x100)", "Autograd", [&]() {
            auto c_var = a + b;
            if (std::holds_alternative<Tensor<float, 2>>(c_var)) {
                auto& c = std::get<Tensor<float, 2>>(c_var);
                volatile float val = c[{0, 0}];
                (void)val;
            }
        }, ITERATIONS_FAST, size * size);
        
        benchmark("Autograd: Matrix multiply forward (100x100)", "Autograd", [&]() {
            auto c_var = a * b;
            if (std::holds_alternative<Tensor<float, 2>>(c_var)) {
                auto& c = std::get<Tensor<float, 2>>(c_var);
                volatile float val = c[{0, 0}];
                (void)val;
            }
        }, ITERATIONS_FAST, size * size);
        
        // Backward pass benchmarks
        benchmark("Autograd: Matrix addition forward+backward (100x100)", "Autograd", [&]() {
            a.zero_grad();
            b.zero_grad();
            auto c_var = a + b;
            if (std::holds_alternative<Tensor<float, 2>>(c_var)) {
                auto c = std::get<Tensor<float, 2>>(c_var);
                auto loss = c.sum();
                Tensor<float, 1> loss_tensor({1}, false, true);
                loss_tensor[{0}] = loss;
                loss_tensor.backward();
            }
        }, ITERATIONS_FAST / 2, size * size);
    }
    
    // Activation functions with autograd
    {
        constexpr size_t size = 100;
        Tensor<float, 2> x({size, size}, false, true);
        fill_random(x);
        
        benchmark("Autograd: Sigmoid forward (100x100)", "Autograd", [&]() {
            auto y = x.sigmoid();
            volatile float val = y[{0, 0}];
            (void)val;
        }, ITERATIONS_FAST, size * size);
        
        benchmark("Autograd: ReLU forward (100x100)", "Autograd", [&]() {
            auto y = x.relu();
            volatile float val = y[{0, 0}];
            (void)val;
        }, ITERATIONS_FAST, size * size);
        
        benchmark("Autograd: Tanh forward (100x100)", "Autograd", [&]() {
            auto y = x.tanh();
            volatile float val = y[{0, 0}];
            (void)val;
        }, ITERATIONS_FAST, size * size);
        
        benchmark("Autograd: Sigmoid forward+backward (100x100)", "Autograd", [&]() {
            x.zero_grad();
            auto y = x.sigmoid();
            auto loss = y.sum();
            Tensor<float, 1> loss_tensor({1}, false, true);
            loss_tensor[{0}] = loss;
            loss_tensor.backward();
        }, ITERATIONS_FAST / 2, size * size);
        
        benchmark("Autograd: ReLU forward+backward (100x100)", "Autograd", [&]() {
            x.zero_grad();
            auto y = x.relu();
            auto loss = y.sum();
            Tensor<float, 1> loss_tensor({1}, false, true);
            loss_tensor[{0}] = loss;
            loss_tensor.backward();
        }, ITERATIONS_FAST / 2, size * size);
    }
}

void benchmark_loss_functions() {
    std::cout << "\n=== Loss Functions Benchmark ===" << std::endl;
    
    constexpr size_t batch_size = 64;
    constexpr size_t num_classes = 10;
    
    // MSE Loss
    {
        Tensor<float, 2> predictions({batch_size, num_classes}, false);
        Tensor<float, 2> targets({batch_size, num_classes}, false);
        fill_random(predictions);
        fill_random(targets);
        
        benchmark("MSE Loss (64x10)", "Loss_Functions", [&]() {
            auto loss = loss::mse_loss(predictions, targets, "mean");
            volatile float val = loss[{0, 0}];
            (void)val;
        }, ITERATIONS_FAST, batch_size * num_classes);
    }
    
    // Cross Entropy Loss
    {
        Tensor<float, 2> logits({batch_size, num_classes}, false);
        Tensor<float, 2> targets({batch_size, num_classes}, false);
        fill_random(logits);
        fill_random(targets, 0.0f, 1.0f);
        
        benchmark("Cross Entropy Loss (64x10)", "Loss_Functions", [&]() {
            auto loss = loss::cross_entropy_loss(logits, targets, "mean");
            volatile float val = loss[{0}];
            (void)val;
        }, ITERATIONS_FAST, batch_size * num_classes);
    }
    
    // Binary Cross Entropy
    {
        Tensor<float, 2> predictions({batch_size, 1}, false);
        Tensor<float, 2> targets({batch_size, 1}, false);
        fill_random(predictions, 0.1f, 0.9f);
        fill_random(targets, 0.0f, 1.0f);
        
        benchmark("Binary Cross Entropy (64x1)", "Loss_Functions", [&]() {
            auto loss = loss::binary_cross_entropy(predictions, targets, "mean");
            volatile float val = loss[{0}];
            (void)val;
        }, ITERATIONS_FAST, batch_size);
    }
    
    // L1 Loss
    {
        Tensor<float, 2> predictions({batch_size, num_classes}, false);
        Tensor<float, 2> targets({batch_size, num_classes}, false);
        fill_random(predictions);
        fill_random(targets);
        
        benchmark("L1 Loss (64x10)", "Loss_Functions", [&]() {
            auto loss = loss::l1_loss(predictions, targets, "mean");
            volatile float val = loss[{0}];
            (void)val;
        }, ITERATIONS_FAST, batch_size * num_classes);
    }
    
    // Smooth L1 Loss
    {
        Tensor<float, 2> predictions({batch_size, num_classes}, false);
        Tensor<float, 2> targets({batch_size, num_classes}, false);
        fill_random(predictions);
        fill_random(targets);
        
        benchmark("Smooth L1 Loss (64x10)", "Loss_Functions", [&]() {
            auto loss = loss::smooth_l1_loss(predictions, targets, 1.0f, "mean");
            volatile float val = loss[{0}];
            (void)val;
        }, ITERATIONS_FAST, batch_size * num_classes);
    }
}

void benchmark_optimizers() {
    std::cout << "\n=== Optimizer Benchmark ===" << std::endl;
    
    constexpr size_t param_size = 1000;
    
    // SGD Optimizer
    {
        Tensor<float, 1> param({param_size}, false, true);
        fill_random(param);
        std::vector<Tensor<float, 1>*> params = {&param};
        SGD<float, 1> optimizer(params, 0.01f);
        
        benchmark("SGD step (1000 parameters)", "Optimizers", [&]() {
            // Simulate gradient
            if (param.grad()) {
                param.grad()->fill(0.1f);
            }
            optimizer.step();
        }, ITERATIONS_FAST, param_size);
        
        benchmark("SGD zero_grad (1000 parameters)", "Optimizers", [&]() {
            optimizer.zero_grad();
        }, ITERATIONS_FAST, param_size);
    }
    
    // SGD with Momentum
    {
        Tensor<float, 2> param({100, 100}, false, true);
        fill_random(param);
        std::vector<Tensor<float, 2>*> params = {&param};
        SGD<float, 2> optimizer(params, 0.01f, 0.9f);  // With momentum
        
        benchmark("SGD+Momentum step (100x100)", "Optimizers", [&]() {
            if (param.grad()) {
                param.grad()->fill(0.1f);
            }
            optimizer.step();
        }, ITERATIONS_FAST, 10000);
    }
    
    // Adam Optimizer
    {
        Tensor<float, 1> param({param_size}, false, true);
        fill_random(param);
        std::vector<Tensor<float, 1>*> params = {&param};
        Adam<float, 1> optimizer(params, 0.001f);
        
        benchmark("Adam step (1000 parameters)", "Optimizers", [&]() {
            if (param.grad()) {
                param.grad()->fill(0.1f);
            }
            optimizer.step();
        }, ITERATIONS_FAST, param_size);
    }
    
    // RMSprop Optimizer
    {
        Tensor<float, 1> param({param_size}, false, true);
        fill_random(param);
        std::vector<Tensor<float, 1>*> params = {&param};
        RMSprop<float, 1> optimizer(params, 0.01f);
        
        benchmark("RMSprop step (1000 parameters)", "Optimizers", [&]() {
            if (param.grad()) {
                param.grad()->fill(0.1f);
            }
            optimizer.step();
        }, ITERATIONS_FAST, param_size);
    }
    
    // Multiple parameters
    {
        Tensor<float, 2> w1({100, 100}, false, true);
        Tensor<float, 1> b1({100}, false, true);
        Tensor<float, 2> w2({100, 50}, false, true);
        Tensor<float, 1> b2({50}, false, true);
        fill_random(w1);
        fill_random(b1);
        fill_random(w2);
        fill_random(b2);
        
        std::vector<Tensor<float, 2>*> params2d = {&w1, &w2};
        std::vector<Tensor<float, 1>*> params1d = {&b1, &b2};
        
        Adam<float, 2> optimizer2d(params2d, 0.001f);
        Adam<float, 1> optimizer1d(params1d, 0.001f);
        
        benchmark("Adam multi-param step (15150 params)", "Optimizers", [&]() {
            if (w1.grad()) w1.grad()->fill(0.1f);
            if (w2.grad()) w2.grad()->fill(0.1f);
            if (b1.grad()) b1.grad()->fill(0.1f);
            if (b2.grad()) b2.grad()->fill(0.1f);
            optimizer2d.step();
            optimizer1d.step();
        }, ITERATIONS_FAST, 15150);
    }
}

void benchmark_advanced_operations() {
    std::cout << "\n=== Advanced Tensor Operations Benchmark ===" << std::endl;
    
    // Reshape
    {
        Tensor<float, 2> t({100, 100}, false);
        fill_random(t);
        TensorIndices<1> new_dims = {10000};
        
        benchmark("Reshape (100x100 -> 10000)", "Advanced", [&]() {
            auto result = t.reshape(new_dims);
            volatile float val = result[{0}];
            (void)val;
        }, ITERATIONS_MEDIUM, 10000);
    }
    
    // Transpose
    {
        Tensor<float, 2> t({200, 500}, false);
        fill_random(t);
        
        benchmark("Transpose (200x500)", "Advanced", [&]() {
            auto result = t.transpose();
            volatile float val = result[{0, 0}];
            (void)val;
        }, ITERATIONS_MEDIUM, 100000);
    }
    
    // Concatenate
    {
        Tensor<float, 1> a({1000}, false);
        Tensor<float, 1> b({1000}, false);
        fill_random(a);
        fill_random(b);
        
        benchmark("Concatenate 1D (1000 + 1000)", "Advanced", [&]() {
            auto result = a.concatenate(b, 0);
            volatile float val = result[{0}];
            (void)val;
        }, ITERATIONS_MEDIUM, 2000);
    }
    
    // Softmax
    {
        Tensor<float, 2> t({50, 500}, false);
        fill_random(t);
        
        benchmark("Softmax (50x500)", "Advanced", [&]() {
            auto result = t.softmax(1);
            volatile float val = result[{0, 0}];
            (void)val;
        }, ITERATIONS_SLOW, 25000);
    }
}

void benchmark_3d_tensor_operations() {
    std::cout << "\n=== 3D Tensor Dot Product Benchmark ===" << std::endl;
    
    struct TensorSize {
        TensorIndices<3> dims_a;
        TensorIndices<1> dims_b;
        std::string description;
    };
    
    std::vector<TensorSize> sizes = {
        {{10, 10, 10}, {10}, "Small 3D (10x10x10 * 10)"},
        {{20, 20, 20}, {20}, "Medium 3D (20x20x20 * 20)"},
        {{50, 50, 50}, {50}, "Large 3D (50x50x50 * 50)"},
    };
    
    for (const auto& size : sizes) {
        Tensor<float, 3> a(size.dims_a);
        Tensor<float, 1> b(size.dims_b);
        fill_random(a);
        fill_random(b);
        
        size_t total_ops = size.dims_a[0] * size.dims_a[1] * size.dims_a[2] * size.dims_b[0];
        benchmark(size.description, "3D_Dot_1D", [&]() {
            auto result = a.dot(b);
            // Force computation
            if (std::holds_alternative<Tensor<float, 2>>(result)) {
                volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
                (void)val;
            }
        }, ITERATIONS_SLOW, total_ops);
    }
}

void benchmark_3d_with_2d() {
    std::cout << "\n=== 3D Tensor * 2D Matrix Benchmark ===" << std::endl;
    
    struct TensorSize {
        TensorIndices<3> dims_a;
        TensorIndices<2> dims_b;
        std::string description;
    };
    
    std::vector<TensorSize> sizes = {
        {{10, 10, 10}, {10, 8}, "Small (10x10x10 * 10x8)"},
        {{20, 20, 15}, {15, 20}, "Medium (20x20x15 * 15x20)"},
        {{50, 30, 20}, {20, 25}, "Large (50x30x20 * 20x25)"},
    };
    
    for (const auto& size : sizes) {
        Tensor<float, 3> a(size.dims_a);
        Tensor<float, 2> b(size.dims_b);
        fill_random(a);
        fill_random(b);
        
        size_t total_ops = size.dims_a[0] * size.dims_a[1] * size.dims_a[2] * size.dims_b[0] * size.dims_b[1];
        benchmark(size.description, "3D_Dot_2D", [&]() {
            auto result = a.dot(b);
            if (std::holds_alternative<Tensor<float, 3>>(result)) {
                volatile float val = std::get<Tensor<float, 3>>(result)[{0, 0, 0}];
                (void)val;
            }
        }, ITERATIONS_SLOW, total_ops);
    }
}

void benchmark_tensor_construction() {
    std::cout << "\n=== Tensor Construction Benchmark ===" << std::endl;
    
    benchmark("1D Tensor (1M elements)", "Construction", [&]() {
        Tensor<float, 1> t({TENSOR_1D_SIZE});
        t.fill(1.0f);
    }, ITERATIONS_FAST, TENSOR_1D_SIZE);
    
    benchmark("2D Tensor (1000x1000)", "Construction", [&]() {
        Tensor<float, 2> t({TENSOR_2D_SIZE, TENSOR_2D_SIZE});
        t.fill(1.0f);
    }, ITERATIONS_FAST, TENSOR_2D_SIZE * TENSOR_2D_SIZE);
    
    benchmark("3D Tensor (100x100x100)", "Construction", [&]() {
        Tensor<float, 3> t({TENSOR_3D_SIZE, TENSOR_3D_SIZE, TENSOR_3D_SIZE});
        t.fill(1.0f);
    }, ITERATIONS_FAST, TENSOR_3D_SIZE * TENSOR_3D_SIZE * TENSOR_3D_SIZE);
    
    benchmark("4D Tensor (50x50x20x20)", "Construction", [&]() {
        Tensor<float, 4> t({TENSOR_4D_SIZE_1, TENSOR_4D_SIZE_1, TENSOR_4D_SIZE_2, TENSOR_4D_SIZE_2});
        t.fill(1.0f);
    }, ITERATIONS_FAST, TENSOR_4D_SIZE_1 * TENSOR_4D_SIZE_1 * TENSOR_4D_SIZE_2 * TENSOR_4D_SIZE_2);
}

void benchmark_copy_operations() {
    std::cout << "\n=== Copy Operations Benchmark ===" << std::endl;
    
    {
        Tensor<float, 2> a({TENSOR_2D_SIZE, TENSOR_2D_SIZE});
        fill_random(a);
        
        benchmark("Copy 2D Tensor (1000x1000)", "Copy", [&]() {
            Tensor<float, 2> copy(a);
            volatile float val = copy[{0, 0}];
            (void)val;
        }, ITERATIONS_FAST, TENSOR_2D_SIZE * TENSOR_2D_SIZE);
    }
    
    {
        Tensor<float, 3> a({TENSOR_3D_SIZE, TENSOR_3D_SIZE, TENSOR_3D_SIZE});
        fill_random(a);
        
        benchmark("Copy 3D Tensor (100x100x100)", "Copy", [&]() {
            Tensor<float, 3> copy(a);
            volatile float val = copy[{0, 0, 0}];
            (void)val;
        }, ITERATIONS_FAST, TENSOR_3D_SIZE * TENSOR_3D_SIZE * TENSOR_3D_SIZE);
    }
}

void benchmark_element_access() {
    std::cout << "\n=== Element Access Benchmark ===" << std::endl;
    
    Tensor<float, 2> matrix({TENSOR_2D_SIZE, TENSOR_2D_SIZE});
    fill_random(matrix);
    
    benchmark("Sequential access 2D (1M elements)", "Element_Access", [&]() {
        float sum = 0.0f;
        for (size_t i = 0; i < TENSOR_2D_SIZE; ++i) {
            for (size_t j = 0; j < TENSOR_2D_SIZE; ++j) {
                sum += matrix[{i, j}];
            }
        }
        volatile float val = sum;
        (void)val;
    }, ITERATIONS_FAST, TENSOR_2D_SIZE * TENSOR_2D_SIZE);
    
    Tensor<float, 3> tensor3d({TENSOR_3D_SIZE, TENSOR_3D_SIZE, TENSOR_3D_SIZE});
    fill_random(tensor3d);
    
    benchmark("Sequential access 3D (1M elements)", "Element_Access", [&]() {
        float sum = 0.0f;
        for (size_t i = 0; i < TENSOR_3D_SIZE; ++i) {
            for (size_t j = 0; j < TENSOR_3D_SIZE; ++j) {
                for (size_t k = 0; k < TENSOR_3D_SIZE; ++k) {
                    sum += tensor3d[{i, j, k}];
                }
            }
        }
        volatile float val = sum;
        (void)val;
    }, ITERATIONS_FAST, TENSOR_3D_SIZE * TENSOR_3D_SIZE * TENSOR_3D_SIZE);
}

void print_system_info() {
    std::cout << "\n=== System Information ===" << std::endl;
    std::cout << "C++ Standard: " << __cplusplus << std::endl;
    
#ifdef _OPENMP
    std::cout << "OpenMP: Enabled" << std::endl;
#else
    std::cout << "OpenMP: Disabled" << std::endl;
#endif
    
#ifdef __cpp_lib_parallel_algorithm
    std::cout << "Parallel STL: Available" << std::endl;
#else
    std::cout << "Parallel STL: Not available" << std::endl;
#endif

#ifdef USE_GPU
    std::cout << "GPU Support: Enabled" << std::endl;
#else
    std::cout << "GPU Support: Disabled" << std::endl;
#endif

#ifdef USE_BLAS
    std::cout << "BLAS Support: Enabled (Optimized CPU operations)" << std::endl;
#else
    std::cout << "BLAS Support: Disabled" << std::endl;
#endif
    
    std::cout << "Float size: " << sizeof(float) << " bytes" << std::endl;
    std::cout << "Double size: " << sizeof(double) << " bytes" << std::endl;
    std::cout << "Size_t size: " << sizeof(size_t) << " bytes" << std::endl;
}

#ifdef USE_GPU
void benchmark_gpu_intensive() {
    std::cout << "\n=====================================" << std::endl;
    std::cout << "  GPU INTENSIVE BENCHMARKS" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Very large vector dot products
    std::cout << "\n=== GPU: Large Vector Dot Products ===" << std::endl;
    {
        constexpr size_t sizes[] = {1000000, 10000000, 50000000};
        for (size_t size : sizes) {
            Tensor<float, 1> a({size});
            Tensor<float, 1> b({size});
            fill_random(a);
            fill_random(b);
            
            std::string desc = "GPU Vector " + std::to_string(size / 1000000) + "M elements";
            benchmark(desc.c_str(), "GPU_Vector_Dot", [&]() {
                auto result = a.dot(b);
                volatile float val = std::get<float>(result);
                (void)val;
            }, 10, size);
        }
    }
    
    // Very large matrix multiplications
    std::cout << "\n=== GPU: Large Matrix Multiplications ===" << std::endl;
    {
        struct GPUMatrixConfig {
            size_t size;
            const char* desc;
            int iterations;
        };
        
        GPUMatrixConfig configs[] = {
            {1000, "GPU Matrix 1000x1000", 10},
            {2000, "GPU Matrix 2000x2000", 5},
            {3000, "GPU Matrix 3000x3000", 3},
            {5000, "GPU Matrix 5000x5000", 2},
        };
        
        for (const auto& cfg : configs) {
            Tensor<float, 2> a({cfg.size, cfg.size});
            Tensor<float, 2> b({cfg.size, cfg.size});
            fill_random(a);
            fill_random(b);
            
            benchmark(cfg.desc, "GPU_Matrix_Mult", [&]() {
                auto result = a.dot(b);
                volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
                (void)val;
            }, cfg.iterations, cfg.size * cfg.size * cfg.size);
        }
    }
    
    // Large rectangular matrices (common in neural networks)
    std::cout << "\n=== GPU: Neural Network-like Matrices ===" << std::endl;
    {
        struct NNConfig {
            size_t m, n, p;
            const char* desc;
        };
        
        NNConfig configs[] = {
            {128, 1024, 512, "GPU Batch 128 x Hidden 1024->512"},
            {256, 2048, 1024, "GPU Batch 256 x Hidden 2048->1024"},
            {512, 4096, 2048, "GPU Batch 512 x Hidden 4096->2048"},
            {1024, 1024, 1024, "GPU Square 1024x1024"},
            {2048, 512, 2048, "GPU Wide 2048x512x2048"},
        };
        
        for (const auto& cfg : configs) {
            Tensor<float, 2> a({cfg.m, cfg.n});
            Tensor<float, 2> b({cfg.n, cfg.p});
            fill_random(a);
            fill_random(b);
            
            benchmark(cfg.desc, "GPU_NN_Mult", [&]() {
                auto result = a.dot(b);
                volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
                (void)val;
            }, 5, cfg.m * cfg.n * cfg.p);
        }
    }
    
    // Large 3D tensor operations
    std::cout << "\n=== GPU: Large 3D Tensor Operations ===" << std::endl;
    {
        {
            Tensor<float, 3> a({100, 100, 100});
            Tensor<float, 1> b({100});
            fill_random(a);
            fill_random(b);
            
            benchmark("GPU 3D Tensor (100x100x100) * Vector", "GPU_3D_Ops", [&]() {
                auto result = a.dot(b);
                volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
                (void)val;
            }, 5, 100 * 100 * 100 * 100);
        }
        
        {
            Tensor<float, 3> a({200, 150, 100});
            Tensor<float, 2> b({100, 128});
            fill_random(a);
            fill_random(b);
            
            benchmark("GPU 3D Tensor (200x150x100) * Matrix (100x128)", "GPU_3D_Ops", [&]() {
                auto result = a.dot(b);
                volatile float val = std::get<Tensor<float, 3>>(result)[{0, 0, 0}];
                (void)val;
            }, 3, 200 * 150 * 100 * 128);
        }
    }
    
    // Double precision benchmarks
    std::cout << "\n=== GPU: Double Precision ===" << std::endl;
    {
        {
            Tensor<double, 1> a({10000000});
            Tensor<double, 1> b({10000000});
            fill_random(a);
            fill_random(b);
            
            benchmark("GPU Double Vector 10M elements", "GPU_Double", [&]() {
                auto result = a.dot(b);
                volatile double val = std::get<double>(result);
                (void)val;
            }, 5, 10000000);
        }
        
        {
            Tensor<double, 2> a({2000, 2000});
            Tensor<double, 2> b({2000, 2000});
            fill_random(a);
            fill_random(b);
            
            benchmark("GPU Double Matrix 2000x2000", "GPU_Double", [&]() {
                auto result = a.dot(b);
                volatile double val = std::get<Tensor<double, 2>>(result)[{0, 0}];
                (void)val;
            }, 3, 2000ULL * 2000 * 2000);
        }
    }
    
    // CPU vs GPU comparison (small tensors to show overhead)
    std::cout << "\n=== GPU: CPU vs GPU Crossover Point ===" << std::endl;
    {
        size_t sizes[] = {10, 50, 100, 200, 500, 1000};
        for (size_t size : sizes) {
            Tensor<float, 2> a_gpu({size, size}, true);
            Tensor<float, 2> b_gpu({size, size}, true);
            fill_random(a_gpu);
            fill_random(b_gpu);
            
            std::string desc = "GPU Matrix " + std::to_string(size) + "x" + std::to_string(size) + " (GPU-enabled)";
            benchmark(desc.c_str(), "GPU_Crossover", [&]() {
                auto result = a_gpu.dot(b_gpu);
                volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
                (void)val;
            }, 20, size * size * size);
            
            Tensor<float, 2> a_cpu({size, size}, false);
            Tensor<float, 2> b_cpu({size, size}, false);
            fill_random(a_cpu);
            fill_random(b_cpu);
            
            desc = "CPU Matrix " + std::to_string(size) + "x" + std::to_string(size) + " (GPU-disabled)";
            benchmark(desc.c_str(), "GPU_Crossover", [&]() {
                auto result = a_cpu.dot(b_cpu);
                volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
                (void)val;
            }, 20, size * size * size);
        }
    }
    
    // Batch operations
    std::cout << "\n=== GPU: Batch Processing ===" << std::endl;
    {
        constexpr size_t batch_size = 64;
        constexpr size_t input_size = 512;
        constexpr size_t output_size = 256;
        
        Tensor<float, 2> batch({batch_size, input_size});
        Tensor<float, 2> weight({input_size, output_size});
        fill_random(batch);
        fill_random(weight);
        
        benchmark("GPU Batch processing (64x512 * 512x256)", "GPU_Batch", [&]() {
            auto result = batch.dot(weight);
            volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
            (void)val;
        }, 10, batch_size * input_size * output_size);
        
        // Larger batch
        constexpr size_t large_batch = 256;
        Tensor<float, 2> large_batch_data({large_batch, input_size});
        fill_random(large_batch_data);
        
        benchmark("GPU Large batch (256x512 * 512x256)", "GPU_Batch", [&]() {
            auto result = large_batch_data.dot(weight);
            volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
            (void)val;
        }, 10, large_batch * input_size * output_size);
    }
}
#endif

#ifdef USE_BLAS
void benchmark_blas_operations() {
    std::cout << "\n=====================================" << std::endl;
    std::cout << "  BLAS: Optimized CPU Operations" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Vector dot product benchmarks (CPU only, with BLAS)
    std::cout << "\n=== BLAS: Vector Dot Product ===" << std::endl;
    for (size_t size : VECTOR_SIZES) {
        Tensor<float, 1> v1({size}, false);  // CPU only
        Tensor<float, 1> v2({size}, false);  // CPU only
        fill_random(v1);
        fill_random(v2);
        
        std::string desc = "BLAS Float vector dot (" + std::to_string(size) + " elements)";
        benchmark(desc.c_str(), "BLAS_Vector", [&]() {
            auto result = v1.dot(v2);
            volatile float val = std::get<float>(result);
            (void)val;
        }, ITERATIONS_FAST, size);
        
        // Double precision
        Tensor<double, 1> d1({size}, false);
        Tensor<double, 1> d2({size}, false);
        fill_random(d1);
        fill_random(d2);
        
        desc = "BLAS Double vector dot (" + std::to_string(size) + " elements)";
        benchmark(desc.c_str(), "BLAS_Vector", [&]() {
            auto result = d1.dot(d2);
            volatile double val = std::get<double>(result);
            (void)val;
        }, ITERATIONS_FAST, size);
    }
    
    // Matrix multiplication benchmarks (CPU only, with BLAS)
    std::cout << "\n=== BLAS: Matrix Multiplication ===" << std::endl;
    for (const auto& config : MATRIX_CONFIGS) {
        Tensor<float, 2> a({config.m, config.n}, false);  // CPU only
        Tensor<float, 2> b({config.n, config.p}, false);  // CPU only
        fill_random(a);
        fill_random(b);
        
        std::string desc = "BLAS Float " + std::string(config.description);
        benchmark(desc.c_str(), "BLAS_Matrix", [&]() {
            auto result = a.dot(b);
            volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
            (void)val;
        }, ITERATIONS_MEDIUM, static_cast<size_t>(config.m) * config.n * config.p);
        
        // Double precision
        Tensor<double, 2> da({config.m, config.n}, false);
        Tensor<double, 2> db({config.n, config.p}, false);
        fill_random(da);
        fill_random(db);
        
        desc = "BLAS Double " + std::string(config.description);
        benchmark(desc.c_str(), "BLAS_Matrix", [&]() {
            auto result = da.dot(db);
            volatile double val = std::get<Tensor<double, 2>>(result)[{0, 0}];
            (void)val;
        }, ITERATIONS_MEDIUM, static_cast<size_t>(config.m) * config.n * config.p);
    }
    
    // BLAS vs Standard CPU comparison
    std::cout << "\n=== BLAS: Performance Comparison ===" << std::endl;
    {
        constexpr size_t test_size = 512;
        
        Tensor<float, 2> a({test_size, test_size}, false);
        Tensor<float, 2> b({test_size, test_size}, false);
        fill_random(a);
        fill_random(b);
        
        std::string desc = "BLAS-optimized matrix mult (512x512)";
        benchmark(desc.c_str(), "BLAS_Comparison", [&]() {
            auto result = a.dot(b);
            volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
            (void)val;
        }, 50, test_size * test_size * test_size);
        
        // Test with very large matrices
        constexpr size_t large_size = 1024;
        Tensor<double, 2> large_a({large_size, large_size}, false);
        Tensor<double, 2> large_b({large_size, large_size}, false);
        fill_random(large_a);
        fill_random(large_b);
        
        desc = "BLAS-optimized large matrix (1024x1024)";
        benchmark(desc.c_str(), "BLAS_Comparison", [&]() {
            auto result = large_a.dot(large_b);
            volatile double val = std::get<Tensor<double, 2>>(result)[{0, 0}];
            (void)val;
        }, 10, large_size * large_size * large_size);
    }
    
    // Batch processing with BLAS
    std::cout << "\n=== BLAS: Batch Processing ===" << std::endl;
    {
        constexpr size_t batch_size = 64;
        constexpr size_t input_size = 512;
        constexpr size_t output_size = 256;
        
        Tensor<float, 2> batch({batch_size, input_size}, false);
        Tensor<float, 2> weight({input_size, output_size}, false);
        fill_random(batch);
        fill_random(weight);
        
        benchmark("BLAS Batch processing (64x512 * 512x256)", "BLAS_Batch", [&]() {
            auto result = batch.dot(weight);
            volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
            (void)val;
        }, 30, batch_size * input_size * output_size);
        
        // Double precision batch
        Tensor<double, 2> dbatch({batch_size, input_size}, false);
        Tensor<double, 2> dweight({input_size, output_size}, false);
        fill_random(dbatch);
        fill_random(dweight);
        
        benchmark("BLAS Double batch (64x512 * 512x256)", "BLAS_Batch", [&]() {
            auto result = dbatch.dot(dweight);
            volatile double val = std::get<Tensor<double, 2>>(result)[{0, 0}];
            (void)val;
        }, 30, batch_size * input_size * output_size);
    }
    
    // Rectangular matrix tests (common in ML)
    std::cout << "\n=== BLAS: Rectangular Matrices ===" << std::endl;
    {
        struct RectConfig { size_t m, n, p; const char* desc; };
        RectConfig rect_configs[] = {
            {128, 1024, 128, "Narrow (128x1024 * 1024x128)"},
            {1024, 128, 1024, "Wide (1024x128 * 128x1024)"},
            {64, 4096, 64, "Very narrow (64x4096 * 4096x64)"},
            {4096, 64, 4096, "Very wide (4096x64 * 64x4096)"},
        };
        
        for (const auto& cfg : rect_configs) {
            Tensor<float, 2> a({cfg.m, cfg.n}, false);
            Tensor<float, 2> b({cfg.n, cfg.p}, false);
            fill_random(a);
            fill_random(b);
            
            benchmark(cfg.desc, "BLAS_Rectangular", [&]() {
                auto result = a.dot(b);
                volatile float val = std::get<Tensor<float, 2>>(result)[{0, 0}];
                (void)val;
            }, 20, cfg.m * cfg.n * cfg.p);
        }
    }
}
#endif

// ============================================
// Linear Algebra Benchmarks
// ============================================

void benchmark_linalg_vector_ops() {
    std::cout << "\n[Linear Algebra: Vector Operations]" << std::endl;
    
    for (size_t size : {100, 1000, 10000}) {
        // Vector norm
        benchmark(
            "LinAlg",
            "Vector norm (size=" + std::to_string(size) + ")",
            [&]() {
                Vector<float> v({size}, false);
                fill_random(v);
                volatile float result = linalg::norm(v);
                (void)result;
            }, ITERATIONS_FAST, size);
        
        // Vector dot product
        benchmark(
            "LinAlg",
            "Vector dot (size=" + std::to_string(size) + ")",
            [&]() {
                Vector<float> a({size}, false);
                Vector<float> b({size}, false);
                fill_random(a);
                fill_random(b);
                volatile float result = linalg::dot(a, b);
                (void)result;
            }, ITERATIONS_FAST, size);
        
        if (size <= 1000) {  // Outer product can be large
            // Vector outer product
            benchmark(
                "LinAlg",
                "Vector outer (size=" + std::to_string(size) + "x" + std::to_string(size) + ")",
                [&]() {
                    Vector<float> a({size}, false);
                    Vector<float> b({size}, false);
                    fill_random(a);
                    fill_random(b);
                    auto result = linalg::outer(a, b);
                }, ITERATIONS_FAST, size * size);
        }
    }
}

void benchmark_linalg_matrix_ops() {
    std::cout << "\n[Linear Algebra: Matrix Operations]" << std::endl;
    
    // Matrix-vector multiplication
    for (size_t size : {50, 100, 200}) {
        benchmark(
            "LinAlg",
            "Matrix-vector " + std::to_string(size) + "x" + std::to_string(size),
            [&]() {
                Matrix<float> mat({size, size}, false);
                Vector<float> vec({size}, false);
                fill_random(mat);
                fill_random(vec);
                auto result = linalg::matvec(mat, vec);
            }, ITERATIONS_MEDIUM, size * size);
    }
    
    // Matrix-matrix multiplication
    for (size_t size : {50, 100, 200}) {
        benchmark(
            "LinAlg",
            "Matrix mult " + std::to_string(size) + "x" + std::to_string(size),
            [&]() {
                Matrix<float> a({size, size}, false);
                Matrix<float> b({size, size}, false);
                fill_random(a);
                fill_random(b);
                auto result = linalg::matmul(a, b);
            }, ITERATIONS_MEDIUM, size * size * size);
    }
    
    // Matrix transpose
    for (size_t size : {100, 500, 1000}) {
        benchmark(
            "LinAlg",
            "Matrix transpose " + std::to_string(size) + "x" + std::to_string(size),
            [&]() {
                Matrix<float> mat({size, size}, false);
                fill_random(mat);
                auto result = linalg::transpose(mat);
            }, ITERATIONS_FAST, size * size);
    }
}

void benchmark_linalg_views() {
    std::cout << "\n[Linear Algebra: Tensor Views]" << std::endl;
    
    // 1D slice
    benchmark(
        "LinAlg",
        "1D tensor slice (10000 elements)",
        [&]() {
            Tensor<float, 1> tensor({10000}, false);
            fill_random(tensor);
            auto view = TensorSlice<float, 1>::slice(tensor, 0, 100, 5100);
            volatile float sum = 0;
            for (size_t i = 0; i < 1000; ++i) {
                sum += view[{i}];
            }
        }, ITERATIONS_FAST, 10000);
    
    // Matrix row access
    benchmark(
        "LinAlg",
        "Matrix row view (1000x1000)",
        [&]() {
            Matrix<float> mat({1000, 1000}, false);
            fill_random(mat);
            volatile float sum = 0;
            for (size_t i = 0; i < 100; ++i) {
                auto row = TensorSlice<float, 2>::row(mat, i);
                sum += row[{0}];
            }
        }, ITERATIONS_FAST, 1000000);
    
    // Matrix block access
    benchmark(
        "LinAlg",
        "Matrix block view (500x500 from 1000x1000)",
        [&]() {
            Matrix<float> mat({1000, 1000}, false);
            fill_random(mat);
            auto block = TensorSlice<float, 2>::block(mat, 0, 500, 0, 500);
            volatile float sum = 0;
            for (size_t i = 0; i < 100; ++i) {
                sum += block[{i, i}];
            }
        }, ITERATIONS_FAST, 250000);
}

#ifdef USE_GPU
void benchmark_linalg_gpu() {
    std::cout << "\n[Linear Algebra: GPU Acceleration]" << std::endl;
    
    for (size_t size : {100, 500}) {
        // GPU matrix multiplication
        benchmark(
            "LinAlg-GPU",
            "GPU Matrix mult " + std::to_string(size) + "x" + std::to_string(size),
            [&]() {
                Matrix<float> a({size, size}, true);  // GPU enabled
                Matrix<float> b({size, size}, true);
                fill_random(a);
                fill_random(b);
                auto result = linalg::matmul(a, b);
            }, ITERATIONS_MEDIUM, size * size * size);
        
        // GPU vector dot
        benchmark(
            "LinAlg-GPU",
            "GPU Vector dot (size=" + std::to_string(size * 10) + ")",
            [&]() {
                Vector<float> a({size * 10}, true);
                Vector<float> b({size * 10}, true);
                fill_random(a);
                fill_random(b);
                volatile float result = linalg::dot(a, b);
                (void)result;
            }, ITERATIONS_FAST, size * 10);
    }
}
#endif

#ifdef USE_BLAS
void benchmark_linalg_blas() {
    std::cout << "\n[Linear Algebra: BLAS Operations]" << std::endl;
    
    for (size_t size : {100, 500, 1000}) {
        // BLAS matrix multiplication
        benchmark(
            "LinAlg-BLAS",
            "BLAS Matrix mult " + std::to_string(size) + "x" + std::to_string(size),
            [&]() {
                Matrix<float> a({size, size}, false);  // CPU with BLAS
                Matrix<float> b({size, size}, false);
                fill_random(a);
                fill_random(b);
                auto result = linalg::matmul(a, b);
            }, ITERATIONS_MEDIUM, size * size * size);
        
        // BLAS vector dot
        benchmark(
            "LinAlg-BLAS",
            "BLAS Vector dot (size=" + std::to_string(size * 100) + ")",
            [&]() {
                Vector<float> a({size * 100}, false);
                Vector<float> b({size * 100}, false);
                fill_random(a);
                fill_random(b);
                volatile float result = linalg::dot(a, b);
                (void)result;
            }, ITERATIONS_FAST, size * 100);
    }
}
#endif

void write_results_to_csv(const std::string& filename) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write header
    csv << "Category,Benchmark,Mean(ms),StdDev(ms),Min(ms),Max(ms),Iterations,Total_Operations\n";
    
    // Write data
    for (const auto& result : all_results) {
        csv << result.category << ","
            << "\"" << result.name << "\","
            << std::fixed << std::setprecision(6) << result.mean_ms << ","
            << result.std_dev_ms << ","
            << result.min_ms << ","
            << result.max_ms << ","
            << result.iterations << ","
            << result.total_operations << "\n";
    }
    
    csv.close();
    std::cout << "\nResults written to: " << filename << std::endl;
}

// ============================================
// Performance Optimization Benchmarks
// ============================================

void benchmark_memory_pool() {
    std::cout << "\n=== Memory Pool Benchmark ===" << std::endl;
    
    // Benchmark standard allocation vs pool allocation
    const size_t alloc_size = 10000;
    const int num_allocs = 1000;
    
    benchmark(
        "MemoryPool",
        "Standard new/delete (" + std::to_string(num_allocs) + " allocations)",
        [&]() {
            std::vector<float*> ptrs;
            for (int i = 0; i < num_allocs; ++i) {
                ptrs.push_back(new float[alloc_size]);
            }
            for (auto ptr : ptrs) {
                delete[] ptr;
            }
        }, 5, num_allocs * alloc_size);
    
    benchmark(
        "MemoryPool",
        "Pool allocate/deallocate (" + std::to_string(num_allocs) + " allocations)",
        [&]() {
            auto& pool = get_memory_pool<float>();
            std::vector<float*> ptrs;
            for (int i = 0; i < num_allocs; ++i) {
                ptrs.push_back(pool.allocate(alloc_size));
            }
            for (auto ptr : ptrs) {
                pool.deallocate(ptr, alloc_size);
            }
        }, 5, num_allocs * alloc_size);
}

void benchmark_thread_pool() {
    std::cout << "\n=== Thread Pool Benchmark ===" << std::endl;
    
    const size_t num_tasks = 1000;
    
    benchmark(
        "ThreadPool",
        "Serial execution (" + std::to_string(num_tasks) + " tasks)",
        [&]() {
            std::vector<double> results;
            for (size_t i = 0; i < num_tasks; ++i) {
                results.push_back(std::sqrt(static_cast<double>(i)));
            }
            volatile double sum = 0;
            for (auto r : results) sum += r;
            (void)sum;
        }, ITERATIONS_FAST, num_tasks);
    
    benchmark(
        "ThreadPool",
        "ThreadPool execution (" + std::to_string(num_tasks) + " tasks)",
        [&]() {
            auto& pool = get_thread_pool();
            std::vector<std::future<double>> futures;
            for (size_t i = 0; i < num_tasks; ++i) {
                futures.push_back(pool.enqueue([i]() {
                    return std::sqrt(static_cast<double>(i));
                }));
            }
            volatile double sum = 0;
            for (auto& f : futures) sum += f.get();
            (void)sum;
        }, ITERATIONS_FAST, num_tasks);
}

void benchmark_parallel_for() {
    std::cout << "\n=== Parallel For Benchmark ===" << std::endl;
    
    const size_t data_size = 1000000;
    std::vector<double> data(data_size);
    
    benchmark(
        "ParallelFor",
        "Serial for-loop (1M elements)",
        [&]() {
            for (size_t i = 0; i < data_size; ++i) {
                data[i] = std::sqrt(static_cast<double>(i));
            }
        }, 5, data_size);
    
    benchmark(
        "ParallelFor",
        "parallel_for (1M elements)",
        [&]() {
            parallel_for(0, data_size, [&](size_t i) {
                data[i] = std::sqrt(static_cast<double>(i));
            });
        }, 5, data_size);
}

void benchmark_mixed_precision() {
    std::cout << "\n=== Mixed Precision Benchmark ===" << std::endl;
    
    const size_t array_size = 100000;
    std::vector<float> fp32_data(array_size);
    std::vector<Float16> fp16_data(array_size);
    std::vector<BFloat16> bf16_data(array_size);
    
    // Initialize
    for (size_t i = 0; i < array_size; ++i) {
        fp32_data[i] = static_cast<float>(i) * 0.01f;
    }
    
    benchmark(
        "MixedPrecision",
        "FP32 to FP16 conversion (100K elements)",
        [&]() {
            convert_fp32_to_fp16(fp32_data.data(), fp16_data.data(), array_size);
        }, ITERATIONS_FAST, array_size);
    
    benchmark(
        "MixedPrecision",
        "FP16 to FP32 conversion (100K elements)",
        [&]() {
            convert_fp16_to_fp32(fp16_data.data(), fp32_data.data(), array_size);
        }, ITERATIONS_FAST, array_size);
    
    benchmark(
        "MixedPrecision",
        "FP32 to BF16 conversion (100K elements)",
        [&]() {
            convert_fp32_to_bf16(fp32_data.data(), bf16_data.data(), array_size);
        }, ITERATIONS_FAST, array_size);
    
    benchmark(
        "MixedPrecision",
        "BF16 to FP32 conversion (100K elements)",
        [&]() {
            convert_bf16_to_fp32(bf16_data.data(), fp32_data.data(), array_size);
        }, ITERATIONS_FAST, array_size);
}

void benchmark_lazy_evaluation() {
    std::cout << "\n=== Lazy Evaluation Benchmark ===" << std::endl;
    
    const size_t array_size = 100000;
    std::vector<float> src(array_size);
    std::vector<float> dst(array_size);
    
    // Initialize
    for (size_t i = 0; i < array_size; ++i) {
        src[i] = static_cast<float>(i) * 0.001f;
    }
    
    benchmark(
        "LazyEval",
        "Separate exp + tanh (100K elements)",
        [&]() {
            std::vector<float> temp(array_size);
            for (size_t i = 0; i < array_size; ++i) {
                temp[i] = std::exp(src[i]);
            }
            for (size_t i = 0; i < array_size; ++i) {
                dst[i] = std::tanh(temp[i]);
            }
        }, ITERATIONS_FAST, array_size * 2);
    
    benchmark(
        "LazyEval",
        "Fused exp + tanh (100K elements)",
        [&]() {
            std::vector<OperationType> ops = {OperationType::Exp, OperationType::Tanh};
            auto fused = fuse_operations<float>(ops);
            fused(dst.data(), src.data(), array_size);
        }, ITERATIONS_FAST, array_size);
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "  Tensor Performance Benchmark Suite" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    print_system_info();
    
    benchmark_tensor_construction();
    benchmark_copy_operations();
    benchmark_element_access();
    benchmark_arithmetic_operations();
    benchmark_math_functions();
    benchmark_reduction_operations();
    benchmark_statistical_operations();
    benchmark_autograd_operations();
    benchmark_loss_functions();
    benchmark_optimizers();
    benchmark_advanced_operations();
    benchmark_vector_dot();
    benchmark_matrix_multiplication();
    benchmark_3d_tensor_operations();
    benchmark_3d_with_2d();
    
    // Linear algebra benchmarks
    benchmark_linalg_vector_ops();
    benchmark_linalg_matrix_ops();
    benchmark_linalg_views();
    
    // Performance optimization benchmarks
    benchmark_memory_pool();
    benchmark_thread_pool();
    benchmark_parallel_for();
    benchmark_mixed_precision();
    benchmark_lazy_evaluation();
    
#ifdef USE_GPU
    benchmark_gpu_intensive();
    benchmark_linalg_gpu();
#endif
    
#ifdef USE_BLAS
    benchmark_blas_operations();
    benchmark_linalg_blas();
#endif
    
    std::cout << "\n=====================================" << std::endl;
    std::cout << "  Benchmark Complete!" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Generate timestamp for filename
    auto now = system_clock::now();
    auto time = system_clock::to_time_t(now);
    std::string timestamp = std::to_string(time);
    std::string csv_filename = "tensor_benchmark_results_" + timestamp + ".csv";
    
    write_results_to_csv(csv_filename);
    
    // Also write a copy without timestamp for easy access
    write_results_to_csv("tensor_benchmark_results.csv");
    
    return 0;
}
