# Tensor Normalization Optimization Procedure
## Complete Guide for Future Optimization Sessions

**Date:** 2025-11-12  
**Session:** Optimization of tensor_normalize.h  
**Result:** All 4 normalization functions optimized (L1, L2, Z-score, Min-Max)

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Optimization Phase](#pre-optimization-phase)
3. [Implementation Phase](#implementation-phase)
4. [Post-Optimization Phase](#post-optimization-phase)
5. [Lessons Learned](#lessons-learned)
6. [Templates and Code Patterns](#templates-and-code-patterns)

---

## Overview

### Optimization Strategy

The optimization follows a **three-tier backend system**:

1. **GPU (CUDA)** - Highest priority, 50-100x speedup for large tensors
2. **BLAS** - CPU optimization, 2-4x speedup using optimized libraries
3. **Parallel CPU (C++20)** - Fallback using `std::execution::par_unseq`, 2-8x speedup

### Key Principles

- **Use Tensor operations first**: Prefer high-level Tensor methods (sum(), abs(), operator/, etc.) over raw GPU/BLAS calls
- **Data locality**: GPU tensors stay on GPU, avoid CPU↔GPU transfers
- **Graceful degradation**: Falls back through backends if unavailable
- **Memory management**: Use Tensor objects instead of raw cudaMalloc/cudaFree
- **Compile-time selection**: Use `#ifdef` and `if constexpr` for backend dispatch
- **Two optimization approaches**:
  1. **High-level approach**: Use existing Tensor operations (preferred for axis=-1)
  2. **Low-level approach**: Direct GPU kernels (needed for axis-specific operations)

---

## Pre-Optimization Phase

### Step 1: Architecture Analysis

**Goal:** Understand the existing tensor infrastructure before making changes.

#### What to Review:

1. **Base Tensor Class** (`tensor_base.h`)
   - Available operations: sum(), mean(), std(), min(), max(), abs()
   - GPU support status of each operation
   - Operator overloads: +, -, *, / (tensor-tensor, tensor-scalar)
   - Memory management patterns (smart pointers, GPU flag)
   - Existing backend detection (`uses_gpu()`, `is_gpu_available()`)
   - Data access methods (`data_ptr()`)

2. **Existing GPU Operations** (`tensor_gpu.cuh`, `tensor_gpu.cu`)
   - Kernel patterns (reduction, element-wise, axis operations)
   - Function naming conventions (`_gpu` vs `_gpu_direct`)
   - Template instantiation patterns
   - How existing operations handle GPU memory

3. **BLAS Integration** (`tensor_blas.h`)
   - Available BLAS functions (for your operation type)
   - Calling conventions
   - Type support (float vs double)
   - Stride and layout considerations

#### Document Your Findings:

Create architecture summary documents:
- `ARCHITECTURE_SUMMARY.md` - Key findings about base classes
- `OPTIMIZATION_PLAN.md` - Strategy and expected improvements

**Example findings:**

```cpp
// Check what Tensor operations are available and GPU-enabled
tensor.sum()      // GPU-enabled ✓
tensor.abs()      // GPU-enabled ✓
tensor.min()      // GPU-enabled ✓
tensor.max()      // GPU-enabled ✓
tensor.mean()     // GPU-enabled ✓
tensor.std()      // GPU-enabled ✓
tensor / scalar   // GPU-enabled ✓
tensor * tensor   // GPU-enabled ✓

// GPU tensors have data already on GPU
if (tensor.uses_gpu()) {
    T* gpu_ptr = tensor.data_ptr();  // Already device pointer
    // Can call GPU kernel directly
}

// Two GPU operation patterns exist:
void operation_gpu(const T* cpu_a, T* cpu_result, size_t n);      // Copies data
void operation_gpu_direct(T* d_a, T* d_result, size_t n);         // Direct GPU
```

### Step 2: Establish Baseline

**Goal:** Create benchmarks to measure improvement.

#### Create Benchmark Infrastructure:

1. **Benchmark Test** (`benchs/your_operation_benchmark.cc`)
   - Use Google Benchmark framework
   - Test multiple tensor sizes (small, medium, large)
   - Test both axis=-1 and axis-specific operations (if applicable)
   - Cover all data types (float, double, int if applicable)
   - Test both CPU and GPU tensors

2. **Benchmark Runner** (`benchs/run_benchmark.sh`)
   - Automated execution
   - Saves timestamped results (JSON + text)
   - Creates symlinks to latest results

3. **Comparison Tool** (`benchs/compare_benchmarks.py`)
   - Automatically compares consecutive runs
   - Shows performance improvements/regressions
   - Generates summary statistics

#### Example Benchmark Structure:

```cpp
static void BM_YourOperation_1D_Small(benchmark::State& state) {
    Tensor<float, 1> tensor({1000});
    // Initialize with random data
    
    for (auto _ : state) {
        auto result = your_operation(tensor);
        benchmark::DoNotOptimize(result.data_ptr());
    }
}
BENCHMARK(BM_YourOperation_1D_Small);

// GPU benchmark
static void BM_YourOperation_GPU_1D_Large(benchmark::State& state) {
    Tensor<float, 1> tensor({1000000}, true);  // GPU tensor
    // Initialize with random data
    
    for (auto _ : state) {
        auto result = your_operation(tensor);
        benchmark::DoNotOptimize(result.data_ptr());
    }
}
BENCHMARK(BM_YourOperation_GPU_1D_Large);
```

### Step 3: Create Backup

**Always backup files before modification:**

```bash
mkdir -p /tmp/tensor_backup_$(date +%Y%m%d_%H%M%S)
cp include/your_file.h /tmp/tensor_backup_$(date +%Y%m%d_%H%M%S)/
```

---

## Implementation Phase

### Decision Tree: Which Approach to Use?

```
START: Does your operation work on ALL elements (axis=-1)?
  |
  ├─ YES: Can it be expressed using existing Tensor operations?
  |   |     (sum, mean, abs, min, max, operators +,-,*,/)
  |   |
  |   ├─ YES → Use HIGH-LEVEL APPROACH (Phase 2A)
  |   |        ✓ Simple, clean code
  |   |        ✓ Automatic GPU/BLAS/CPU support
  |   |        ✓ Memory managed automatically
  |   |
  |   └─ NO → Use LOW-LEVEL APPROACH (Phase 2B)
  |           Need to add new GPU kernels
  |
  └─ NO: Axis-specific operation?
      |
      ├─ For axis=-1 only → Use HIGH-LEVEL APPROACH
      |
      └─ For axis-specific → Use LOW-LEVEL APPROACH (Phase 2B)
              Must implement axis kernels
```

---

### Phase 2A: High-Level Tensor Operations Approach

**Use this when**: Operation can be expressed with existing Tensor methods

**Example: L1 Normalization (axis=-1)**

```cpp
template <typename T, size_t N>
Tensor<T, N> normalize_l1(const Tensor<T, N>& tensor, int axis = -1) {
    if (axis == -1) {
        // Use Tensor operations - automatically GPU/BLAS/CPU
        T sum = tensor.abs().sum();
        if (sum > T(0)) {
            return tensor / sum;  // Operator/ handles backend selection
        } else {
            return tensor;
        }
    }
    // ... axis-specific code
}
```

**Benefits:**
- ✅ Automatically GPU-accelerated (if Tensor operations support GPU)
- ✅ Automatically uses BLAS (if available)
- ✅ Falls back to parallel CPU
- ✅ Clean, readable code
- ✅ No manual memory management
- ✅ No raw GPU calls

**When to add GPU support to Tensor operations:**

If a Tensor method doesn't support GPU yet, add it once and reuse everywhere:

1. Add GPU kernel to `tensor_gpu.cu`
2. Add declaration to `tensor_gpu.cuh`
3. Update the Tensor method in `tensor_base.h` to call GPU kernel when `use_gpu_` is true
4. All future code using that operation gets GPU support automatically

**Example: Adding GPU support to abs()**

```cpp
// In tensor_base.h
Tensor<T, N> abs() const {
#ifdef USE_GPU
    if (use_gpu_) {
        Tensor<T, N> result(dims_, true);
        ensure_on_gpu();
        result.ensure_on_gpu();
        abs_gpu_direct(d_data_, result.d_data_, total_size());
        result.mark_gpu_modified();
        return result;
    }
#endif
    return map([](T x) { return std::abs(x); });
}
```

---

### Phase 2B: Low-Level GPU Kernel Approach

**Use this when**: 
- Operation needs custom GPU kernels
- Axis-specific operations
- Operation cannot be expressed with existing Tensor methods

### Phase 1: GPU Kernel Development

#### Step 1.1: Design Kernels

For each operation, identify the GPU operations needed:

**Example: Normalization**
- **Reduction**: Compute sum, norm, min, max
- **Element-wise**: Divide each element by scalar
- **Axis operation**: Per-slice reduction and normalization

**Example: Matrix operations**
- **GEMM**: Matrix multiplication
- **Transpose**: Reorder elements
- **Reduction**: Row/column sums

#### Step 1.2: Add Kernel Declarations

**File:** `include/tensor_gpu.cuh`

```cpp
// Add to existing function declarations
template<typename T>
void abs_sum_gpu_direct(const T* d_src, T* d_result, size_t n);

template<typename T>
void abs_sum_axis_gpu_direct(const T* d_src, T* d_sums, 
                              size_t outer, size_t axis_size, size_t inner);
```

**Naming Convention:**
- Use descriptive names: `abs_sum`, `l2_norm`, `zscore_normalize`
- Add `_gpu_direct` suffix for functions expecting GPU pointers
- Add `_axis` suffix for axis-specific operations

#### Step 1.3: Implement Kernels

**File:** `src/tensor_gpu.cu`

**Pattern for Reduction Kernels:**

```cpp
template<typename T>
__global__ void reduction_kernel(const T* src, T* partial_results, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and compute
    T value = (i < n) ? compute_value(src[i]) : T(0);
    sdata[tid] = value;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = combine(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_results[blockIdx.x] = sdata[0];
    }
}
```

**Pattern for Axis Operations:**

```cpp
template<typename T>
__global__ void axis_operation_kernel(const T* src, T* dst,
                                      size_t outer, size_t axis_size, size_t inner) {
    size_t o = blockIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (o < outer && i < inner) {
        // Process one slice
        for (size_t a = 0; a < axis_size; ++a) {
            size_t idx = o * axis_size * inner + a * inner + i;
            // Perform operation
        }
    }
}
```

**Pattern for Wrapper Functions:**

```cpp
template<typename T>
void operation_gpu_direct(const T* d_src, T* d_result, size_t n) {
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Allocate temporary storage if needed
    T* d_temp;
    cudaMalloc(reinterpret_cast<void**>(&d_temp), blocks * sizeof(T));
    
    // Launch kernel
    operation_kernel<<<blocks, threads_per_block, 
                       threads_per_block * sizeof(T)>>>(d_src, d_temp, n);
    
    // Final reduction or processing
    // ...
    
    // Cleanup
    cudaFree(d_temp);
    cudaDeviceSynchronize();
}
```

#### Step 1.4: Add Template Instantiations

**File:** `src/tensor_gpu.cu` (at the end, before `} // namespace tensor`)

```cpp
// Function name instantiations
template void abs_sum_gpu_direct<float>(const float*, float*, size_t);
template void abs_sum_gpu_direct<double>(const double*, double*, size_t);

template void abs_sum_axis_gpu_direct<float>(const float*, float*, 
                                              size_t, size_t, size_t);
template void abs_sum_axis_gpu_direct<double>(const double*, double*, 
                                               size_t, size_t, size_t);
```

**Important:** Instantiate for both `float` and `double` types.

#### Step 1.5: Handle CUDA Type Casting

**Common Error:**
```cpp
cudaMalloc(&d_ptr, size);  // ERROR: T** to void** conversion
```

**Solution:**
```cpp
cudaMalloc(reinterpret_cast<void**>(&d_ptr), size);  // CORRECT
```

### Phase 2: BLAS Integration

#### Step 2.1: Identify Available BLAS Functions

**File:** `include/tensor_normalize.h`

Add extern declarations:

```cpp
#ifdef USE_BLAS
extern "C" {
    // L1 norm (sum of absolute values)
    float cblas_sasum(const int N, const float *X, const int incX);
    double cblas_dasum(const int N, const double *X, const int incX);
    
    // L2 norm (Euclidean norm)
    float cblas_snrm2(const int N, const float *X, const int incX);
    double cblas_dnrm2(const int N, const double *X, const int incX);
}
#endif
```

#### Step 2.2: Use BLAS in Optimization

**Pattern:**

```cpp
#ifdef USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        T norm = cblas_snrm2(static_cast<int>(total), src, 1);
        // Use the result...
        return result;
    } else if constexpr (std::is_same_v<T, double>) {
        T norm = cblas_dnrm2(static_cast<int>(total), src, 1);
        // Use the result...
        return result;
    }
#endif
```

**Note:** BLAS only works on CPU data, not GPU.

### Phase 3: Parallel CPU Implementation

#### Step 3.1: Use C++20 Parallel Algorithms

**Available Algorithms:**

1. **Reduction:**
```cpp
T sum = std::reduce(std::execution::par_unseq, src, src + n, T(0));
```

2. **Map-Reduce:**
```cpp
T result = std::transform_reduce(std::execution::par_unseq,
                                 src, src + n, T(0), std::plus<>(),
                                 [](T x) { return transform(x); });
```

3. **Transform:**
```cpp
std::transform(std::execution::par_unseq, src, src + n, dst,
              [](T x) { return f(x); });
```

4. **Min/Max:**
```cpp
auto [min_it, max_it] = std::minmax_element(std::execution::par_unseq, 
                                             src, src + n);
```

5. **Parallel Loop:**
```cpp
std::for_each(std::execution::par_unseq,
             std::views::iota(size_t(0), count).begin(),
             std::views::iota(size_t(0), count).end(),
             [&](size_t i) { /* parallel body */ });
```

**Execution Policies:**
- `std::execution::seq` - Sequential
- `std::execution::par` - Parallel
- `std::execution::par_unseq` - Parallel + vectorized (best performance)

### Phase 4: Function Optimization Pattern

#### Complete Optimization Template:

```cpp
template <typename T, size_t N>
Tensor<T, N> normalize_function(const Tensor<T, N>& tensor, int axis = -1, 
                                 /* other params */) {
    Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
    const T* src = tensor.data_ptr();
    T* dst = result.data_ptr();
    size_t total = tensor.total_size();
    
    if (axis == -1) {
        // ========== GPU PATH ==========
#ifdef USE_GPU
        if (tensor.uses_gpu()) {
            // Allocate temporary GPU storage
            T* d_temp;
            cudaMalloc(reinterpret_cast<void**>(&d_temp), sizeof(T));
            
            // Call GPU kernel
            operation_gpu_direct(src, d_temp, dst, total);
            
            // Cleanup
            cudaFree(d_temp);
            return result;
        }
#endif

        // ========== BLAS PATH ==========
#ifdef USE_BLAS
        if constexpr (std::is_same_v<T, float>) {
            T value = cblas_sfunc(static_cast<int>(total), src, 1);
            // Process with parallel algorithms...
            return result;
        } else if constexpr (std::is_same_v<T, double>) {
            T value = cblas_dfunc(static_cast<int>(total), src, 1);
            // Process with parallel algorithms...
            return result;
        }
#endif

        // ========== PARALLEL CPU PATH ==========
        // Compute using std::execution::par_unseq
        T result_value = std::transform_reduce(std::execution::par_unseq,
                                               src, src + total, /* ... */);
        
        std::transform(std::execution::par_unseq, src, src + total, dst,
                      [result_value](T x) { return process(x); });
        
    } else if constexpr (N >= 2) {
        // Axis-specific normalization
        auto dims = tensor.dims();
        size_t axis_size = dims[axis];
        size_t outer_size = 1;
        size_t inner_size = 1;
        
        for (size_t i = 0; i < axis; ++i) {
            outer_size *= dims[i];
        }
        for (size_t i = axis + 1; i < N; ++i) {
            inner_size *= dims[i];
        }
        
        // ========== GPU PATH (AXIS) ==========
#ifdef USE_GPU
        if (tensor.uses_gpu()) {
            axis_operation_gpu_direct(src, dst, outer_size, axis_size, inner_size);
            return result;
        }
#endif

        // ========== PARALLEL CPU PATH (AXIS) ==========
        std::for_each(std::execution::par_unseq,
                     std::views::iota(size_t(0), outer_size).begin(),
                     std::views::iota(size_t(0), outer_size).end(),
                     [&](size_t outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                // Sequential inner loop (better cache locality)
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + 
                                ax * inner_size + inner;
                    // Process element
                }
            }
        });
    } else {
        // Fallback: just copy
        std::copy(std::execution::par_unseq, src, src + total, dst);
    }
    
    return result;
}
```

---

## Post-Optimization Phase

### Step 1: Compilation and Testing

#### Build the Project:

```bash
cd build
cmake ..
make
```

#### Common Compilation Issues:

1. **CUDA type casting errors:**
   ```
   error: invalid conversion from 'T**' to 'void**'
   ```
   **Fix:** Use `reinterpret_cast<void**>(&ptr)`

2. **Missing template instantiations:**
   ```
   undefined reference to `function<float>(...)`
   ```
   **Fix:** Add template instantiation in `.cu` file

3. **Parallel algorithm errors:**
   ```
   error: 'execution' is not a member of 'std'
   ```
   **Fix:** Ensure C++20 is enabled, link TBB if needed

#### Run Tests:

```bash
# Unit tests
./tensor_test

# Specific normalization tests
./tensor_test --gtest_filter="*Normalization*"
```

### Step 2: Benchmark Comparison

#### Run Baseline (before optimization):

```bash
cd benchs
./run_benchmark.sh
# Results saved as benchs/results/benchmark_TIMESTAMP.json
```

#### Run After Optimization:

```bash
./run_benchmark.sh
# Automatically compares with previous run
# Generates comparison report
```

#### Analyze Results:

Look for in the comparison report:
- Functions marked with ↑ (faster - improvement)
- Functions marked with ↓ (slower - regression)
- Overall summary statistics

**Expected improvements:**
- GPU: 50-100x for large tensors
- BLAS: 2-4x improvement
- Parallel CPU: 2-8x depending on cores

### Step 3: Documentation

Create comprehensive documentation:

1. **Implementation Summary** (`NORMALIZE_L1_OPTIMIZATION.md`)
   - What was changed
   - New functions added
   - Performance expectations

2. **Update Main Documentation**
   - Add "Optimized for GPU/BLAS/Parallel CPU" to function docstrings
   - Update any performance claims

3. **Create This Guide** (`OPTIMIZATION_PROCEDURE.md`)
   - Document the process for future sessions

---

## Lessons Learned

### What Worked Well

1. **Incremental Optimization**
   - Optimize one function at a time
   - Test each function before moving to next
   - Easier to debug issues

2. **Reusing Existing Infrastructure**
   - Existing GPU kernels (`sum_kernel`, `sqrt_gpu_direct`)
   - Existing BLAS integration patterns
   - Reduced development time significantly

3. **Three-Tier Backend System**
   - Clear priority: GPU > BLAS > Parallel CPU
   - Graceful fallback ensures compatibility
   - Users get best performance available

4. **Architecture Analysis First**
   - Understanding existing patterns prevented mistakes
   - Avoided duplicate code
   - Followed project conventions

### Challenges Faced

1. **CUDA Type Casting**
   - `cudaMalloc` requires `void**` but templates use `T**`
   - **Solution:** Always use `reinterpret_cast<void**>(&ptr)`

2. **Template Instantiations**
   - Easy to forget instantiations for new functions
   - **Solution:** Add instantiations immediately after implementation
   - Create checklist: float, double, int (where applicable)

3. **Parallel Algorithm Performance**
   - Not all algorithms benefit equally from parallelization
   - Small tensors may be slower with parallel execution
   - **Solution:** Could add size threshold in future
   ```cpp
   if (total > PARALLEL_THRESHOLD) {
       // Use parallel
   } else {
       // Use sequential
   }
   ```

4. **GPU Memory Management**
   - Must track temporary allocations
   - **Solution:** Always pair `cudaMalloc` with `cudaFree`
   - Consider RAII wrappers for complex functions

### Best Practices Discovered

1. **Consistent Naming:**
   ```
   function_name_gpu_direct()      # Direct GPU operation
   function_name_axis_gpu_direct() # Axis-specific GPU
   ```

2. **Code Organization:**
   - GPU kernels: `src/tensor_gpu.cu`
   - GPU declarations: `include/tensor_gpu.cuh`
   - Optimized functions: `include/tensor_normalize.h`
   - Keep GPU and CPU code separate

3. **Error Handling:**
   - Add `cudaDeviceSynchronize()` at end of GPU operations
   - Check for errors in development builds
   - Document assumptions (e.g., "data must be on GPU")

4. **Testing Strategy:**
   - Test correctness first (unit tests)
   - Then test performance (benchmarks)
   - Compare results between backends
   - Verify GPU memory doesn't leak

---

## Templates and Code Patterns

### GPU Kernel Template

```cpp
// 1. Kernel declaration
template<typename T>
__global__ void my_kernel(const T* input, T* output, size_t n, params...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = process(input[idx], params...);
    }
}

// 2. Wrapper function
template<typename T>
void my_operation_gpu_direct(const T* d_input, T* d_output, size_t n, params...) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    my_kernel<<<blocks, threads>>>(d_input, d_output, n, params...);
    cudaDeviceSynchronize();
}

// 3. Template instantiation
template void my_operation_gpu_direct<float>(const float*, float*, size_t, ...);
template void my_operation_gpu_direct<double>(const double*, double*, size_t, ...);
```

### BLAS Integration Template

```cpp
#ifdef USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        float result = cblas_sfunc(N, x, incx, ...);
        // Use result
        return tensor_result;
    } else if constexpr (std::is_same_v<T, double>) {
        double result = cblas_dfunc(N, x, incx, ...);
        // Use result
        return tensor_result;
    }
#endif
```

### Parallel CPU Template

```cpp
// Reduction
T result = std::reduce(
    std::execution::par_unseq, 
    begin, end, 
    initial_value
);

// Transform-Reduce
T result = std::transform_reduce(
    std::execution::par_unseq,
    begin, end,
    initial_value,
    combine_op,          // e.g., std::plus<>()
    transform_op         // e.g., [](T x) { return x*x; }
);

// Transform
std::transform(
    std::execution::par_unseq,
    src_begin, src_end,
    dst_begin,
    [](T x) { return f(x); }
);

// Parallel loop over indices
std::for_each(
    std::execution::par_unseq,
    std::views::iota(size_t(0), count).begin(),
    std::views::iota(size_t(0), count).end(),
    [&](size_t i) {
        // Process index i
    }
);
```

### Benchmark Template

```cpp
static void BM_FunctionName_Variant(benchmark::State& state) {
    // Setup
    Tensor<float, N> input(dims);
    initialize_tensor(input);
    
    // Benchmark loop
    for (auto _ : state) {
        auto result = function_to_test(input, params...);
        benchmark::DoNotOptimize(result.data_ptr());
    }
    
    // Optional: report custom metrics
    state.SetItemsProcessed(state.iterations() * input.total_size());
}
BENCHMARK(BM_FunctionName_Variant);
```

---

## Checklist for Future Optimizations

### Pre-Implementation

- [ ] Read and understand target function's current implementation
- [ ] Review `tensor_base.h` for relevant methods
- [ ] Check `tensor_gpu.cuh` for reusable GPU operations
- [ ] Check `tensor_blas.h` for available BLAS functions
- [ ] Create architecture summary document
- [ ] Create optimization plan document
- [ ] Set up benchmark infrastructure
- [ ] Run baseline benchmarks
- [ ] Create backup of files to be modified

### GPU Implementation

- [ ] Design kernel operations (reduction, element-wise, etc.)
- [ ] Add function declarations to `tensor_gpu.cuh`
- [ ] Implement kernels in `tensor_gpu.cu`
- [ ] Add wrapper functions with proper memory management
- [ ] Use `reinterpret_cast<void**>` for cudaMalloc
- [ ] Add `cudaDeviceSynchronize()` at end of operations
- [ ] Add template instantiations for float and double
- [ ] Compile and fix any CUDA errors

### BLAS Integration

- [ ] Identify applicable BLAS functions
- [ ] Add extern "C" declarations
- [ ] Use `if constexpr` for type dispatch
- [ ] Handle both float (`cblas_s*`) and double (`cblas_d*`)
- [ ] Combine with parallel algorithms for remaining operations

### Parallel CPU

- [ ] Identify opportunities for parallel algorithms
- [ ] Use `std::execution::par_unseq` for maximum performance
- [ ] Choose appropriate algorithm (reduce, transform, for_each)
- [ ] Test with different tensor sizes

### Function Integration Checklist

#### For High-Level Approach (Tensor operations):
- [ ] Express operation using existing Tensor methods
- [ ] Verify all used methods support GPU
- [ ] If not, add GPU support to those methods first
- [ ] Test with both CPU and GPU tensors
- [ ] Verify no raw pointers or manual memory management

#### For Low-Level Approach (Custom kernels):
- [ ] Follow the three-tier pattern: GPU → BLAS → Parallel CPU
- [ ] Use `#ifdef USE_GPU` and `#ifdef USE_BLAS` guards
- [ ] Ensure early returns prevent fallthrough  
- [ ] Use Tensor objects for temporary buffers (not raw cudaMalloc)
- [ ] Handle axis=-1 and axis-specific cases separately
- [ ] For axis=-1: prefer Tensor operations if possible
- [ ] For axis-specific: implement GPU kernels with BLAS/CPU fallback

#### Memory Management:
- [ ] Use Tensor objects for all temporary buffers
- [ ] Never use raw `cudaMalloc` in operation functions
- [ ] Let Tensor class handle GPU memory lifecycle
- [ ] Example: `Tensor<T, 1> temp({size}, true);` instead of `cudaMalloc(&d_temp, ...)`

### Testing and Validation

- [ ] Compile successfully
- [ ] Run unit tests - verify correctness
- [ ] Run on CPU tensors
- [ ] Run on GPU tensors (if available)
- [ ] Check for memory leaks (valgrind, cuda-memcheck)
- [ ] Run benchmarks and compare with baseline
- [ ] Verify expected speedups achieved

### Documentation

- [ ] Create implementation summary document
- [ ] Update function docstrings
- [ ] Document any caveats or limitations
- [ ] Update this procedure with new learnings
- [ ] Commit changes with descriptive message

---

## Example Sessions

### Session 1: Normalization Functions (2025-11-12)

**Approach Used:** Mixed (High-level for axis=-1, Low-level for axis-specific)

**Functions Optimized:** 4  
- normalize_l1
- normalize_l2
- normalize_zscore
- normalize_minmax

**Implementation Details:**

1. **Added GPU support to Tensor class operations:**
   - `sum()` - uses `sum_gpu_direct()`
   - `abs()` - uses `abs_gpu_direct()`
   - `min()` - uses `min_gpu_direct()`
   - `max()` - uses `max_gpu_direct()`

2. **For axis=-1:** Used high-level Tensor operations
   ```cpp
   // L1: tensor / tensor.abs().sum()
   // L2: tensor / sqrt(squared.sum())
   // Z-score: (tensor - mean) / std
   // Min-Max: scaled by range
   ```

3. **For axis-specific:** Custom GPU kernels
   - L1: `abs_sum_axis_gpu_direct()` + `normalize_by_sums_gpu_direct()`
   - L2: `l2_norm_axis_gpu_direct()` + `normalize_by_sums_gpu_direct()`
   - Z-score: `zscore_normalize_axis_gpu_direct()`
   - Min-Max: `minmax_normalize_axis_gpu_direct()`
   - BLAS fallback for L1/L2 (cblas_sasum, cblas_snrm2)
   - Parallel CPU fallback for all

**GPU Kernels Added:** 12 functions in total

**Files Modified:**
- `include/tensor_gpu.cuh` - Added function declarations
- `src/tensor_gpu.cu` - Added kernels and _direct functions
- `include/tensor_base.h` - Added GPU support to sum(), abs(), min(), max()
- `include/tensor_normalize.h` - Rewrote using Tensor operations + axis kernels

**Performance Improvements (Expected):**
- GPU: 50-100x for large tensors
- BLAS: 2-4x for CPU tensors
- Parallel CPU: 2-8x baseline improvement

**Time Investment:**
- Analysis: ~30 minutes
- GPU support for Tensor operations: ~1 hour
- GPU kernels for axis operations: ~2 hours
- Function rewrite: ~1 hour
- Testing and debugging: ~30 minutes
- Documentation: ~30 minutes
- **Total: ~5.5 hours** for 4 functions

---

## Code Patterns and Templates

### Pattern 1: High-Level Tensor Operations (Preferred for axis=-1)

```cpp
template <typename T, size_t N>
Tensor<T, N> your_operation(const Tensor<T, N>& tensor) {
    // Express operation using Tensor methods
    // GPU/BLAS/CPU backend automatically selected
    
    T value = tensor.some_reduction();  // sum(), min(), max(), etc.
    return tensor / value;               // operator handles backend
}
```

**Advantages:**
- Clean, readable
- Automatically GPU-accelerated
- No memory management needed
- Reuses existing optimized code

### Pattern 2: Low-Level with Three-Tier Fallback

```cpp
template <typename T, size_t N>
Tensor<T, N> your_operation_axis(const Tensor<T, N>& tensor, int axis) {
    Tensor<T, N> result(tensor.dims(), tensor.uses_gpu());
    const T* src = tensor.data_ptr();
    T* dst = result.data_ptr();
    
    // Calculate axis parameters
    auto dims = tensor.dims();
    size_t axis_size = dims[axis];
    size_t outer_size = 1, inner_size = 1;
    for (size_t i = 0; i < axis; ++i) outer_size *= dims[i];
    for (size_t i = axis + 1; i < N; ++i) inner_size *= dims[i];
    
#ifdef USE_GPU
    if (tensor.uses_gpu()) {
        // GPU path: use Tensor objects for temps
        Tensor<T, 1> temp({outer_size * inner_size}, true);
        your_axis_gpu_direct(src, temp.data_ptr(), dst, 
                            outer_size, axis_size, inner_size);
        return result;
    }
#endif

#ifdef USE_BLAS
    // BLAS path (if applicable)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                // Use BLAS with strided access
                // cblas_saxpy, cblas_sdot, etc.
            }
        }
        return result;
    }
#endif

    // Parallel CPU fallback
    std::for_each(std::execution::par_unseq,
                 std::views::iota(size_t(0), outer_size).begin(),
                 std::views::iota(size_t(0), outer_size).end(),
                 [&](size_t outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            // CPU implementation
        }
    });
    return result;
}
```

### Pattern 3: Adding GPU Support to Tensor Methods

```cpp
// In tensor_base.h
T your_reduction() const {
    size_t total = total_size();
    
#ifdef USE_GPU
    if (use_gpu_) {
        ensure_on_gpu();
        Tensor<T, 1> d_result_tensor({1}, true);
        your_reduction_gpu_direct(d_data_, d_result_tensor.data_ptr(), total);
        
        T result;
        cudaMemcpy(&result, d_result_tensor.data_ptr(), 
                   sizeof(T), cudaMemcpyDeviceToHost);
        return result;
    }
#endif
    
    // CPU fallback
    T result = T(0);
    for (size_t i = 0; i < total; ++i) {
        result = /* combine with data_[i] */;
    }
    return result;
}
```

---

## Conclusion

This optimization procedure provides a systematic approach to improving any tensor operation performance. The key decision is choosing the right approach:

1. **High-Level Approach**: Use existing Tensor operations when possible
   - Simplest and cleanest
   - Automatically inherits all optimizations
   - Add GPU support to Tensor methods once, benefit everywhere

2. **Low-Level Approach**: Implement custom kernels when needed
   - Required for axis-specific operations
   - Required for operations that can't be expressed with existing methods
   - Follow three-tier fallback: GPU → BLAS → Parallel CPU

3. **Memory Management**: Always use Tensor objects, never raw cudaMalloc

**Process Summary:**
1. **Understand** the operation and available Tensor methods
2. **Choose** high-level or low-level approach
3. **Implement** with three-tier backend support
4. **Validate** with benchmarks and tests
5. **Document** for future maintainers

Future optimization sessions should follow this pattern, adapting the approach based on the specific operation characteristics.

---

## Quick Reference Card

### When to Use Each Backend

| Operation Type | GPU | BLAS | Parallel CPU |
|---------------|-----|------|--------------|
| Simple reduction | ✓✓✓ | ✓✓ | ✓ |
| Complex reduction | ✓✓✓ | ✗ | ✓✓ |
| Element-wise | ✓✓✓ | ✗ | ✓✓✓ |
| Matrix operations | ✓✓✓ | ✓✓✓ | ✓ |
| Small tensors (<1K) | ✗ | ✓ | ✓✓ |
| Large tensors (>100K) | ✓✓✓ | ✓✓ | ✓✓ |

### Memory Location Guide

| Tensor Type | data_ptr() Points To | Backend to Use |
|------------|---------------------|----------------|
| CPU tensor | CPU memory | BLAS or Parallel CPU |
| GPU tensor | GPU memory | GPU kernels only |

### Memory Management Best Practices

| Situation | ❌ Don't Do | ✅ Do Instead |
|-----------|------------|--------------|
| Temporary GPU buffer | `cudaMalloc(&d_temp, size)` | `Tensor<T, 1> temp({n}, true)` |
| GPU result storage | `cudaMalloc(&d_result, size)` | `Tensor<T, N> result(dims, true)` |
| Copying to/from GPU | Manual cudaMemcpy loops | Use Tensor operations |
| Memory cleanup | `cudaFree(d_ptr)` | Let Tensor destructor handle it |

### Operation Type Guide

| Operation | Best Approach | Backend Priority |
|-----------|--------------|------------------|
| Element-wise (all) | Tensor operators | GPU → CPU |
| Reduction (all) | Tensor sum/min/max | GPU → BLAS → CPU |
| Statistical (all) | Tensor mean/std | GPU → CPU |
| Axis-specific | Custom kernels | GPU → BLAS → CPU |
| Matrix ops | Check if exists | GPU → BLAS → CPU |

### Common Pitfalls

❌ **Don't:**
- Use raw `cudaMalloc`/`cudaFree` in operation functions
- Call BLAS functions on GPU data
- Forget template instantiations in tensor_gpu.cu
- Use sequential algorithms for large tensors
- Copy between CPU and GPU unnecessarily
- Implement custom kernel when Tensor operation exists

✅ **Do:**
- Use Tensor objects for all memory
- Check `tensor.uses_gpu()` before choosing backend
- Use existing Tensor operations when possible
- Add GPU support to Tensor methods for reuse
- Add `cudaDeviceSynchronize()` after kernel launches
- Test with both small and large tensors
- Test with both CPU and GPU tensors

---

**End of Tensor Operations Optimization Procedure**
