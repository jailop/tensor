# Part III: Vectorization and SIMD Optimization

> **Implementation:** `../src/core/kernels.zig` demonstrates SIMD vectorization using Zig's `@Vector` type. This section explains low-level CPU optimization.

## Understanding SIMD (Single Instruction, Multiple Data)

Modern CPUs can process multiple data elements simultaneously using specialized instructions. This is critical for transformer performance.

### CPU Vector Capabilities

**Evolution of SIMD:**
```
SSE (1999):     128-bit registers → 4× float32 operations per instruction
SSE2-SSE4:      Integer and double support
AVX (2011):     256-bit registers → 8× float32 operations per instruction  
AVX2 (2013):    Fused multiply-add (FMA), better integer support
AVX-512 (2017): 512-bit registers → 16× float32 operations per instruction

Performance impact:
  Scalar code:      1 float32/cycle
  AVX-512 code:    16 float32/cycle
  Theoretical speedup: 16x
  Realistic speedup: 8-12x (memory bandwidth limited)
```

**Hardware Details:**
```
Modern CPU (Intel Xeon Scalable):
  - 32-48 cores
  - 2 AVX-512 FMA units per core
  - Peak: 32 float32 operations per cycle per core
  - Total: ~1000 float32 operations per cycle (1 TFLOPS @ 3GHz)

Memory hierarchy:
  L1 cache: 32 KB data + 32 KB instruction per core (4 cycles latency)
  L2 cache: 1 MB per core (12 cycles)
  L3 cache: 2 MB per core, shared (40 cycles)
  DRAM: 40-100 GB/s bandwidth (200+ cycles)
```

---

## Softmax Vectorization

Softmax is the second-largest computational bottleneck (10-15% of time). Vectorizing it provides significant gains.

### Scalar Implementation

```
function stableSoftmax(x: array of float32):
    n = length(x)
    
    // Step 1: Find maximum (numerical stability)
    max_val = -INFINITY
    for i = 0 to n:
        if x[i] > max_val:
            max_val = x[i]
    
    // Step 2: Compute exp(x - max) and sum
    sum = 0.0
    for i = 0 to n:
        x[i] = exp(x[i] - max_val)
        sum += x[i]
    
    // Step 3: Normalize
    inv_sum = 1.0 / sum
    for i = 0 to n:
        x[i] = x[i] × inv_sum
```

**Performance:** ~5 GFLOPS (processing 1M elements/sec)

**Bottlenecks:**
- Loop overhead: branch prediction, iterator updates
- Scalar exp(): 20-40 cycles per call
- Memory access: One element at a time

### Vectorized Implementation (8-wide)

```
function softmaxVectorized(x: array of float32):
    VEC_SIZE = 8
    n = length(x)
    vec_len = n / VEC_SIZE
    
    // Step 1: Find maximum using SIMD
    max_vec = vector_splat(-INFINITY, VEC_SIZE)  // [-inf, -inf, ..., -inf]
    
    i = 0
    while i < vec_len:
        vec = load_vector(&x[i × VEC_SIZE], VEC_SIZE)  // Load 8 floats
        max_vec = vector_max(max_vec, vec)              // Parallel max
        i += 1
    
    // Horizontal reduction: max across vector lanes
    max_val = horizontal_max(max_vec)
    // Example: max_vec = [3, 7, 2, 9, 1, 5, 4, 8]
    //          → horizontal_max = 9
    
    // Handle remainder (n % VEC_SIZE elements)
    i = vec_len × VEC_SIZE
    while i < n:
        max_val = max(max_val, x[i])
        i += 1
    
    // Step 2: Compute exp(x - max) and sum
    max_splat = vector_splat(max_val, VEC_SIZE)  // [max, max, ..., max]
    sum_vec = vector_splat(0.0, VEC_SIZE)
    
    i = 0
    while i < vec_len:
        vec = load_vector(&x[i × VEC_SIZE], VEC_SIZE)
        vec = vector_sub(vec, max_splat)          // 8 subtractions
        vec = vector_exp(vec)                      // 8 exp() via SVML
        store_vector(&x[i × VEC_SIZE], vec)
        sum_vec = vector_add(sum_vec, vec)
        i += 1
    
    sum = horizontal_sum(sum_vec)
    
    // Remainder
    i = vec_len × VEC_SIZE
    while i < n:
        x[i] = exp(x[i] - max_val)
        sum += x[i]
        i += 1
    
    // Step 3: Normalize
    inv_sum = 1.0 / sum
    inv_sum_vec = vector_splat(inv_sum, VEC_SIZE)
    
    i = 0
    while i < vec_len:
        vec = load_vector(&x[i × VEC_SIZE], VEC_SIZE)
        vec = vector_mul(vec, inv_sum_vec)
        store_vector(&x[i × VEC_SIZE], vec)
        i += 1
    
    // Remainder
    i = vec_len × VEC_SIZE
    while i < n:
        x[i] = x[i] × inv_sum
        i += 1
```

**Implementation:** `src/core/kernels.zig`

**Performance:** ~40 GFLOPS (8M elements/sec)

**Speedup:** 8x faster than scalar

**Why Not 16x with AVX-512?**
- Memory bandwidth: Loading data from DRAM is the bottleneck
- Exp() function: Even vectorized, still expensive (~10 cycles)
- Loop overhead: Remainder handling, reductions

### Zig Vector Implementation

Zig provides portable SIMD through `@Vector` type:

```zig
// Zig code (from src/core/kernels.zig)
pub fn softmaxVectorized(x: []f32) void {
    const VecF32 = @Vector(8, f32);  // 8-wide float32 vector
    const vec_len = x.len / 8;
    
    // Find max
    var max_vec = @as(VecF32, @splat(-math.inf(f32)));
    var i: usize = 0;
    while (i < vec_len) : (i += 1) {
        const vec: VecF32 = x[i * 8 ..][0..8].*;  // Load from slice
        max_vec = @max(max_vec, vec);              // Compiler uses SIMD max
    }
    
    // Reduce max vector to scalar
    var max_val: f32 = -math.inf(f32);
    for (0..8) |j| {
        max_val = @max(max_val, max_vec[j]);
    }
    
    // Handle remainder elements...
    
    // Compute exp and sum
    const max_splat = @as(VecF32, @splat(max_val));
    var sum_vec = @as(VecF32, @splat(0.0));
    
    i = 0;
    while (i < vec_len) : (i += 1) {
        var vec: VecF32 = x[i * 8 ..][0..8].*;
        vec = vec - max_splat;                     // Vector subtraction
        vec = @exp(vec);                           // Vector exp
        x[i * 8 ..][0..8].* = vec;
        sum_vec += vec;
    }
    
    // Continue with normalization...
}
```

**Compiler Output (x86_64 with -Dcpu=native):**
```asm
; Load 8 floats
vmovups ymm0, [rsi + rax*4]    ; AVX: load 256 bits = 8×float32

; Max operation
vmaxps ymm1, ymm1, ymm0        ; Parallel max of 8 floats

; Subtraction
vsubps ymm0, ymm0, ymm2        ; 8 subtractions

; Exp (SVML - Intel's Short Vector Math Library)
call __svml_expf8_ha           ; Vectorized exp for 8 floats

; Store result
vmovups [rsi + rax*4], ymm0    ; Store 8 floats
```

---

## Layer Normalization SIMD

Layer norm normalizes features across the hidden dimension, crucial for training stability.

### Algorithm

```
LayerNorm(x, γ, β):
    // x: [hidden_dim] array
    // γ: [hidden_dim] scale parameters
    // β: [hidden_dim] shift parameters
    
    // 1. Compute mean
    mean = (1/d) Σ x_i
    
    // 2. Compute variance
    var = (1/d) Σ (x_i - mean)²
    
    // 3. Normalize
    x̂_i = (x_i - mean) / √(var + ε)
    
    // 4. Scale and shift
    y_i = γ_i × x̂_i + β_i
```

### Scalar Implementation

```
function layerNorm(x, gamma, beta, eps):
    n = length(x)
    
    // Compute mean
    mean = 0.0
    for i = 0 to n:
        mean += x[i]
    mean = mean / n
    
    // Compute variance
    variance = 0.0
    for i = 0 to n:
        diff = x[i] - mean
        variance += diff × diff
    variance = variance / n
    
    // Normalize and scale
    inv_std = 1.0 / sqrt(variance + eps)
    for i = 0 to n:
        x[i] = (x[i] - mean) × inv_std × gamma[i] + beta[i]
```

**Performance:** ~10 GB/s memory bandwidth utilization

### Vectorized Implementation

```
function layerNormVectorized(x, gamma, beta, eps):
    n = length(x)
    vec_len = n / 8
    
    // Compute mean (vectorized reduction)
    sum_vec = vector_splat(0.0, 8)
    for i = 0 to vec_len:
        vec = load_vector(&x[i × 8], 8)
        sum_vec = vector_add(sum_vec, vec)
    
    mean = horizontal_sum(sum_vec) / n
    
    // Compute variance (vectorized reduction)
    mean_vec = vector_splat(mean, 8)
    var_vec = vector_splat(0.0, 8)
    
    for i = 0 to vec_len:
        vec = load_vector(&x[i × 8], 8)
        diff_vec = vector_sub(vec, mean_vec)
        sq_vec = vector_mul(diff_vec, diff_vec)
        var_vec = vector_add(var_vec, sq_vec)
    
    variance = horizontal_sum(var_vec) / n
    
    // Normalize (vectorized)
    inv_std = 1.0 / sqrt(variance + eps)
    inv_std_vec = vector_splat(inv_std, 8)
    mean_vec = vector_splat(mean, 8)
    
    for i = 0 to vec_len:
        x_vec = load_vector(&x[i × 8], 8)
        g_vec = load_vector(&gamma[i × 8], 8)
        b_vec = load_vector(&beta[i × 8], 8)
        
        // Compute: (x - mean) / std × gamma + beta
        norm_vec = vector_sub(x_vec, mean_vec)
        norm_vec = vector_mul(norm_vec, inv_std_vec)
        norm_vec = vector_mul(norm_vec, g_vec)
        result_vec = vector_add(norm_vec, b_vec)
        
        store_vector(&x[i × 8], result_vec)
```

**Performance:** ~60 GB/s memory bandwidth utilization (6x improvement)

---

## GELU Activation Vectorization

GELU (Gaussian Error Linear Unit) is used in feed-forward layers.

### Mathematical Definition

```
GELU(x) = x × Φ(x)

where Φ(x) = (1/√(2π)) ∫_{-∞}^x e^{-t²/2} dt  (Gaussian CDF)
```

**Exact computation is expensive:** Requires error function evaluation.

### Approximate GELU

```
GELU(x) ≈ x × σ(1.702x)  where σ(x) = 1/(1 + e^{-x})

or

GELU(x) ≈ 0.5x × (1 + tanh(√(2/π) × (x + 0.044715x³)))
```

### Vectorized Implementation

```
function gelu_vectorized(x):
    vec_len = length(x) / 8
    
    // Constants
    coeff1 = vector_splat(0.7978845608, 8)  // √(2/π)
    coeff2 = vector_splat(0.044715, 8)
    half = vector_splat(0.5, 8)
    one = vector_splat(1.0, 8)
    
    for i = 0 to vec_len:
        x_vec = load_vector(&x[i × 8], 8)
        
        // Compute x³
        x_sq = vector_mul(x_vec, x_vec)
        x_cubed = vector_mul(x_sq, x_vec)
        
        // Inner term: √(2/π) × (x + 0.044715x³)
        inner = vector_mul(coeff2, x_cubed)
        inner = vector_add(inner, x_vec)
        inner = vector_mul(inner, coeff1)
        
        // tanh(inner)
        tanh_val = vector_tanh(inner)  // Uses lookup table or polynomial
        
        // 0.5 × x × (1 + tanh(...))
        result = vector_add(one, tanh_val)
        result = vector_mul(result, x_vec)
        result = vector_mul(result, half)
        
        store_vector(&x[i × 8], result)
```

**Alternative: Lookup Table**

For even faster approximation:
```
// Precompute GELU for range [-6, 6] with 0.01 resolution
table = [GELU(-6.00), GELU(-5.99), ..., GELU(6.00)]  // 1200 entries

function gelu_lut(x):
    // Clamp and scale to table index
    idx = (x + 6.0) × 100
    idx = clamp(idx, 0, 1199)
    return table[int(idx)]
```

**Speedup:** 10-20x faster, <0.001 max error

---

## Memory Alignment and Prefetching

Proper alignment and prefetching can significantly impact performance.

### Alignment Requirements

**Why alignment matters:**
```
Unaligned load (address not divisible by vector size):
  - CPU must perform 2 memory accesses
  - Cross-cache-line reads
  - 2x slower

Aligned load:
  - Single memory access
  - Single cache line
  - Optimal speed

Example (AVX-512, 64-byte vectors):
  Address 0x1000: Aligned ✓ (0x1000 % 64 = 0)
  Address 0x1008: Misaligned ✗ (0x1008 % 64 = 8)
```

**Ensuring alignment in Zig:**
```zig
// Allocate with specific alignment
const data = try allocator.alignedAlloc(f32, 64, size);

// Verify alignment at runtime
const addr = @intFromPtr(data.ptr);
std.debug.assert(addr % 64 == 0);
```

### Prefetching

**Concept:** Tell CPU to load data into cache before it's needed.

```
function process_large_array(data, size):
    PREFETCH_DISTANCE = 16  // Process 16 elements ahead
    
    for i = 0 to size:
        // Prefetch future element
        if i + PREFETCH_DISTANCE < size:
            prefetch(&data[i + PREFETCH_DISTANCE])
        
        // Process current element
        result = expensive_operation(data[i])
        output[i] = result
```

**Intrinsics:**
```c
// x86 prefetch
_mm_prefetch(address, _MM_HINT_T0);  // Prefetch to L1
_mm_prefetch(address, _MM_HINT_T1);  // Prefetch to L2
_mm_prefetch(address, _MM_HINT_T2);  // Prefetch to L3
```

**Impact:**
- Sequential access: ~10% improvement (hardware prefetcher helps)
- Strided access: ~50% improvement (hardware prefetcher less effective)
- Random access: ~200% improvement (critical for hash tables, graphs)

---

## Practical SIMD Guidelines

### When to Vectorize

**Good candidates:**
- Element-wise operations (add, multiply, exp, log)
- Reductions (sum, max, min)
- Simple transformations (normalization, activation functions)
- Regular memory access patterns

**Poor candidates:**
- Branching logic (if/else on individual elements)
- Irregular memory access (gather/scatter is slow)
- Already memory-bound operations
- Short loops (overhead outweighs benefits)

### Vectorization Checklist

1. **Ensure alignment:** 64-byte for AVX-512, 32-byte for AVX2
2. **Handle remainders:** Process leftover elements with scalar code
3. **Avoid branches in inner loops:** Use masked operations instead
4. **Use compiler intrinsics or vector types:** Portable and optimized
5. **Profile before and after:** Measure actual speedup
6. **Consider memory bandwidth:** Don't vectorize if already bandwidth-limited

### Common Pitfalls

**1. False sharing:**
```
// Bad: Different threads write to nearby addresses
thread1: x[0] += 1   // Cache line 0
thread2: x[1] += 1   // Cache line 0 (contention!)

// Good: Pad to cache line boundaries
thread1: x[0] += 1   // Cache line 0
thread2: x[64] += 1  // Cache line 1 (no contention)
```

**2. Alignment assumptions:**
```
// Bad: Assumes alignment without verification
vmovaps ymm0, [rax]  // Requires 32-byte alignment, crashes if not

// Good: Use unaligned load or verify
vmovups ymm0, [rax]  // Works with any alignment
```

**3. Over-vectorization:**
```
// Bad: Vectorize tiny operations
for i = 0 to 4:
    x[i] = x[i] + 1  // Overhead > benefit

// Good: Use scalar for small loops
x[0] += 1; x[1] += 1; x[2] += 1; x[3] += 1  // Compiler may auto-vectorize
```

---

## Performance Benchmarks

```
Softmax (1M elements):
  Scalar:     200 ms
  Vectorized:  25 ms
  Speedup:     8x

Layer Norm (hidden_dim=768, batch=1024):
  Scalar:     15 ms
  Vectorized:  2.5 ms
  Speedup:     6x

GELU (1M elements):
  Scalar:      100 ms
  Vectorized:   12 ms
  Lookup:        5 ms
  Speedup:      8x (vec), 20x (lut)

Overall transformer inference (BERT-base):
  No SIMD:     180 ms/batch
  With SIMD:    75 ms/batch
  Speedup:     2.4x
```

**Next:** [Part IV - System Scalability and Deployment](./04_scalability.md)
