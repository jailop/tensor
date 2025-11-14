#include <benchmark/benchmark.h>
#include "tensor_normalize.h"
#include <random>

using namespace tensor;

// Benchmark L1 normalization on 1D tensor - small
static void BM_NormalizeL1_1D_Small(benchmark::State& state) {
    Tensor<float, 1> tensor({1000});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l1(tensor);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL1_1D_Small);

// Benchmark L1 normalization on 1D tensor - large
static void BM_NormalizeL1_1D_Large(benchmark::State& state) {
    Tensor<float, 1> tensor({1000000});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l1(tensor);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL1_1D_Large);

// Benchmark L1 normalization on 2D tensor - axis=-1
static void BM_NormalizeL1_2D_AllElements(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l1(tensor, -1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL1_2D_AllElements);

// Benchmark L1 normalization on 2D tensor - axis=0
static void BM_NormalizeL1_2D_Axis0(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l1(tensor, 0);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL1_2D_Axis0);

// Benchmark L1 normalization on 2D tensor - axis=1
static void BM_NormalizeL1_2D_Axis1(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l1(tensor, 1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL1_2D_Axis1);

// Benchmark L2 normalization on 1D tensor - small
static void BM_NormalizeL2_1D_Small(benchmark::State& state) {
    Tensor<float, 1> tensor({1000});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l2(tensor);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL2_1D_Small);

// Benchmark L2 normalization on 1D tensor - large
static void BM_NormalizeL2_1D_Large(benchmark::State& state) {
    Tensor<float, 1> tensor({1000000});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l2(tensor);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL2_1D_Large);

// Benchmark L2 normalization on 2D tensor - axis=-1
static void BM_NormalizeL2_2D_AllElements(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l2(tensor, -1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL2_2D_AllElements);

// Benchmark L2 normalization on 2D tensor - axis=0
static void BM_NormalizeL2_2D_Axis0(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l2(tensor, 0);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL2_2D_Axis0);

// Benchmark L2 normalization on 2D tensor - axis=1
static void BM_NormalizeL2_2D_Axis1(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l2(tensor, 1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL2_2D_Axis1);

// Benchmark Z-score normalization on 1D tensor - small
static void BM_NormalizeZScore_1D_Small(benchmark::State& state) {
    Tensor<float, 1> tensor({1000});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_zscore(tensor);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeZScore_1D_Small);

// Benchmark Z-score normalization on 1D tensor - large
static void BM_NormalizeZScore_1D_Large(benchmark::State& state) {
    Tensor<float, 1> tensor({1000000});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_zscore(tensor);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeZScore_1D_Large);

// Benchmark Z-score normalization on 2D tensor - axis=-1
static void BM_NormalizeZScore_2D_AllElements(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_zscore(tensor, -1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeZScore_2D_AllElements);

// Benchmark Z-score normalization on 2D tensor - axis=0
static void BM_NormalizeZScore_2D_Axis0(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_zscore(tensor, 0);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeZScore_2D_Axis0);

// Benchmark Z-score normalization on 2D tensor - axis=1
static void BM_NormalizeZScore_2D_Axis1(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_zscore(tensor, 1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeZScore_2D_Axis1);

// Benchmark Min-Max normalization on 1D tensor - small
static void BM_NormalizeMinMax_1D_Small(benchmark::State& state) {
    Tensor<float, 1> tensor({1000});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_minmax(tensor);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeMinMax_1D_Small);

// Benchmark Min-Max normalization on 1D tensor - large
static void BM_NormalizeMinMax_1D_Large(benchmark::State& state) {
    Tensor<float, 1> tensor({1000000});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_minmax(tensor);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeMinMax_1D_Large);

// Benchmark Min-Max normalization on 2D tensor - axis=-1
static void BM_NormalizeMinMax_2D_AllElements(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_minmax(tensor, -1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeMinMax_2D_AllElements);

// Benchmark Min-Max normalization on 2D tensor - axis=0
static void BM_NormalizeMinMax_2D_Axis0(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_minmax(tensor, 0);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeMinMax_2D_Axis0);

// Benchmark Min-Max normalization on 2D tensor - axis=1
static void BM_NormalizeMinMax_2D_Axis1(benchmark::State& state) {
    Tensor<float, 2> tensor({100, 100});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_minmax(tensor, 1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeMinMax_2D_Axis1);

// Benchmark 3D tensor normalization - L2 axis=2
static void BM_NormalizeL2_3D_Axis2(benchmark::State& state) {
    Tensor<float, 3> tensor({32, 32, 32});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l2(tensor, 2);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL2_3D_Axis2);

// Benchmark 4D tensor normalization - L2 axis=3 (batch normalization scenario)
static void BM_NormalizeL2_4D_Axis3(benchmark::State& state) {
    Tensor<float, 4> tensor({16, 16, 16, 16});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l2(tensor, 3);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_NormalizeL2_4D_Axis3);

// Benchmark comparing all normalization methods on same data
static void BM_CompareNormMethods_L1(benchmark::State& state) {
    Tensor<float, 2> tensor({256, 256});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l1(tensor, -1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_CompareNormMethods_L1);

static void BM_CompareNormMethods_L2(benchmark::State& state) {
    Tensor<float, 2> tensor({256, 256});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_l2(tensor, -1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_CompareNormMethods_L2);

static void BM_CompareNormMethods_ZScore(benchmark::State& state) {
    Tensor<float, 2> tensor({256, 256});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_zscore(tensor, -1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_CompareNormMethods_ZScore);

static void BM_CompareNormMethods_MinMax(benchmark::State& state) {
    Tensor<float, 2> tensor({256, 256});
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = dis(gen);
    }
    
    for (auto _ : state) {
        auto result = normalize_minmax(tensor, -1);
        benchmark::DoNotOptimize(result.begin());
    }
}
BENCHMARK(BM_CompareNormMethods_MinMax);

BENCHMARK_MAIN();
