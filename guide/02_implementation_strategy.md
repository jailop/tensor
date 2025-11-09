# Part II: Low-Level Implementation Strategy

> **Implementation:** `../src/core/` contains tensor structures and computational kernels. This section covers data structures, algorithms, and low-level optimization.

## Data Structure Design

The foundation of high-performance transformer implementation lies in optimal data layout and memory access patterns.

### Tensor Representation

**Design Goals:**
1. **Cache-friendly access:** Sequential reads maximize prefetching
2. **SIMD compatibility:** Aligned memory enables vectorized operations
3. **Zero-copy indexing:** Direct pointer arithmetic without allocation overhead

**Implementation:**

```
Structure: Tensor
    data: array[float32] aligned to 64-byte boundary
    shape: [batch, seq_len, hidden_dim, heads]
    strides: [stride0, stride1, stride2, stride3]
    allocator: memory allocator handle
    
    method init(allocator, shape):
        total_size = shape[0] × shape[1] × shape[2] × shape[3]
        data = allocate_aligned(allocator, total_size, 64)
        
        // Row-major strides calculation
        strides[3] = 1
        strides[2] = shape[3]
        strides[1] = shape[2] × strides[2]
        strides[0] = shape[1] × strides[1]
        
        return Tensor{data, shape, strides, allocator}
    
    method at(indices: [4]int) -> pointer to float32:
        index = indices[0] × strides[0] + 
                indices[1] × strides[1] + 
                indices[2] × strides[2] + 
                indices[3] × strides[3]
        return &data[index]
```

**Implementation:** `src/core/tensor.zig`

### Why 64-Byte Alignment?

**Cache Line Boundaries:**
- Modern CPUs have 64-byte cache lines
- Misaligned access can span two cache lines → 2x memory traffic
- AVX-512 requires 64-byte alignment for optimal performance

**Example:**
```
Unaligned (addr=0x1008):
  Load 64 bytes: touches cache lines at 0x1000 and 0x1040
  Memory bus transfers: 2 cache lines = 128 bytes
  
Aligned (addr=0x1040):
  Load 64 bytes: touches only cache line at 0x1040
  Memory bus transfers: 1 cache line = 64 bytes
  
Speedup: 2x reduction in memory bandwidth
```

### Row-Major vs Column-Major Layout

**Row-Major (C-style):**
```
A[i][j] = A[i × cols + j]

Memory layout for 3×4 matrix:
  [ a00 a01 a02 a03 | a10 a11 a12 a13 | a20 a21 a22 a23 ]
    ← row 0 →         ← row 1 →         ← row 2 →

Access pattern for column iteration (j):
  a00, a01, a02, a03 → sequential (cache-friendly)
  
Access pattern for row iteration (i):
  a00, a10, a20 → strided (cache-unfriendly)
```

**Column-Major (Fortran-style):**
```
A[i][j] = A[i + j × rows]

Memory layout for 3×4 matrix:
  [ a00 a10 a20 | a01 a11 a21 | a02 a12 a22 | a03 a13 a23 ]
    ← col 0 →     ← col 1 →     ← col 2 →     ← col 3 →

Access pattern for column iteration (j):
  a00, a01, a02, a03 → strided (cache-unfriendly)
  
Access pattern for row iteration (i):
  a00, a10, a20 → sequential (cache-friendly)
```

**Transformer Choice: Row-Major**

Why? Matrix multiplication C = A @ B:
```
for i in rows(A):
    for j in cols(B):
        for k in cols(A):
            C[i,j] += A[i,k] × B[k,j]
```

With row-major:
- A accessed row-wise: sequential ✓
- B accessed column-wise: strided ✗
- Solution: Pre-transpose B or use optimized BLAS

### Batch Dimension Placement

**Bad: Batch Last [seq, batch, hidden]**
```
Memory layout (seq=3, batch=2, hidden=4):
  [s0b0h0 s0b0h1 s0b0h2 s0b0h3 | s0b1h0 s0b1h1 s0b1h2 s0b1h3 |
   s1b0h0 s1b0h1 s1b0h2 s1b0h3 | s1b1h0 s1b1h1 s1b1h2 s1b1h3 | ...]

Processing batch 0:
  Read s0b0h0, skip s0b0h1-s0b0h3, skip s0b1h0-s0b1h3
  Read s1b0h0, skip...
  → Strided access, poor cache utilization
```

**Good: Batch First [batch, seq, hidden]**
```
Memory layout (batch=2, seq=3, hidden=4):
  [b0s0h0 b0s0h1 b0s0h2 b0s0h3 | b0s1h0 b0s1h1 b0s1h2 b0s1h3 | 
   b0s2h0 b0s2h1 b0s2h2 b0s2h3 | b1s0h0 b1s0h1 b1s0h2 b1s0h3 | ...]

Processing batch 0:
  Read b0s0h0, b0s0h1, b0s0h2, b0s0h3 → sequential
  Read b0s1h0, b0s1h1, b0s1h2, b0s1h3 → sequential
  → Entire batch 0 is contiguous: 64KB+ blocks
```

**Cache Impact:**
```
L1 cache: 32-64 KB per core
L2 cache: 256-512 KB per core
L3 cache: 8-64 MB shared

Batch-first allows:
  - Single batch element fits in L2 (seq=128, hidden=768 = 384KB)
  - Entire minibatch (batch=4) fits in L3 (~1.5MB)
  - Prefetcher can load ahead effectively
```

### Attention Matrix Storage

**The Memory Challenge:**

Attention scores have shape [batch, heads, seq_len, seq_len]:
```
Storage requirements:
  batch=32, heads=16, seq=512
  Size = 32 × 16 × 512 × 512 × 4 bytes = 2.1 GB

Implications:
  - Doesn't fit in L3 cache (64 MB typical)
  - Multiple layers → 8-16 GB total
  - GPU memory becomes limiting factor
```

**Storage Strategies:**

**1. Full Materialization (Standard Attention):**
```
Allocate scores[batch, heads, seq, seq]

Pros:
  - Simple implementation
  - Easy to debug and understand
  - Flexible for different attention patterns

Cons:
  - O(N²) memory
  - Cache thrashing for large sequences
  - Memory bandwidth bottleneck
```

**2. Sparse Patterns:**
```
Local Window Attention:
  Only attend to nearby tokens (window_size = 256)
  Storage: batch × heads × seq × window_size
  Memory: 32 × 16 × 512 × 256 × 4 = 256 MB (8x reduction)

Strided Attention:
  Attend to every k-th token (stride = 8)
  Storage: batch × heads × seq × (seq/8)
  Memory: 32 × 16 × 512 × 64 × 4 = 64 MB (32x reduction)

Trade-off: Accuracy vs memory
  - Local attention: 0.5-2% accuracy loss (task-dependent)
  - Works well for long documents (>2048 tokens)
```

**3. Tiled Computation (Flash Attention):**
```
Process in blocks, never materialize full matrix:

BLOCK_SIZE = 64

for block_i in 0 to seq_len / BLOCK_SIZE:
    for block_j in 0 to seq_len / BLOCK_SIZE:
        scores_block = Q[block_i] @ K[block_j]^T  // 64×64 matrix
        // Process and discard
        
Memory: BLOCK_SIZE² × 4 = 64 × 64 × 4 = 16 KB (fits in L1!)

Benefits:
  - Constant memory regardless of sequence length
  - 2-4x faster due to better cache utilization
  - Enables training on 8K+ token sequences
```

### KV Cache for Autoregressive Generation

**The Problem:**

Autoregressive decoding generates one token at a time:
```
Step 1: Generate token 1 from prompt
Step 2: Generate token 2 from prompt + token 1
Step 3: Generate token 3 from prompt + token 1 + token 2
...

At step N, must recompute attention for all previous N-1 tokens!
Complexity: O(N²) for sequence of length N
```

**The Solution: Cache Key/Value States**

```
Structure: KVCache
    k_cache: array[num_layers] of Tensor[batch, heads, max_seq, head_dim]
    v_cache: array[num_layers] of Tensor[batch, heads, max_seq, head_dim]
    seq_len: current position in cache
    max_seq: maximum sequence length
    
    method update(layer_idx, k_new, v_new):
        // k_new, v_new shape: [batch, heads, new_tokens, head_dim]
        
        if seq_len + new_tokens > max_seq:
            return ERROR_CACHE_OVERFLOW
        
        // Copy new K/V to cache at current position
        k_cache[layer_idx][:, :, seq_len:seq_len+new_tokens, :] = k_new
        v_cache[layer_idx][:, :, seq_len:seq_len+new_tokens, :] = v_new
        
        seq_len += new_tokens
    
    method get_cached(layer_idx):
        // Return all cached K/V up to current position
        k = k_cache[layer_idx][:, :, :seq_len, :]
        v = v_cache[layer_idx][:, :, :seq_len, :]
        return k, v
```

**Implementation:** `src/core/tensor.zig`

**Memory Savings:**

```
Without cache (generation of 100 tokens):
  Step 1: Process 1 token
  Step 2: Process 2 tokens (recompute token 1)
  Step 3: Process 3 tokens (recompute tokens 1-2)
  ...
  Step 100: Process 100 tokens
  
  Total token processing: 1 + 2 + 3 + ... + 100 = 5,050 tokens

With cache:
  Each token processed exactly once: 100 tokens
  
  Speedup: 5,050 / 100 = 50.5x

For 1000 tokens: (1000 × 1001) / 2 / 1000 = 500x speedup!
```

**Memory Cost:**

```
Model: 12 layers, 12 heads, head_dim=64, max_seq=2048

K cache: 12 × 1 × 12 × 2048 × 64 × 4 bytes = 75 MB (per batch element)
V cache: 75 MB
Total: 150 MB per sequence

For batch=16: 2.4 GB

Trade-off: 2.4 GB memory for 500x speedup → usually worth it!
```

---

## Computational Kernels

Identifying and optimizing the computational hotspots is critical for performance.

### Profiling Results (BERT-base, batch=32, seq=128)

```
Total forward pass time: 100ms

Breakdown:
  Matrix Multiplication (GEMM):    68ms (68%)
    - Q, K, V projections:         24ms
    - Output projection:           18ms
    - Feed-forward layers:         26ms
  
  Softmax:                         12ms (12%)
  
  Layer Normalization:             8ms (8%)
  
  GELU Activation:                 6ms (6%)
  
  Element-wise ops:                4ms (4%)
  
  Memory transfers:                2ms (2%)
```

**Optimization Priority:**
1. **GEMM optimization:** 68% of time → use optimized BLAS
2. **Softmax:** 12% → vectorize and fuse with subsequent ops
3. **Layer norm:** 8% → SIMD implementation
4. **GELU:** 6% → lookup table or polynomial approximation

### Matrix Multiplication Optimization

**Naive Implementation (O(N³)):**
```
function matmul_naive(A, B, C, M, N, K):
    // C[M,N] = A[M,K] @ B[K,N]
    
    for i = 0 to M:
        for j = 0 to N:
            sum = 0.0
            for k = 0 to K:
                sum += A[i,k] × B[k,j]
            C[i,j] = sum
```

**Performance:** ~10 GFLOPS (0.5% of peak on modern CPU)

**Why So Slow?**
- Inner loop: K iterations, each does 2 memory reads + 1 multiply-add
- A[i,k]: Strided access (skip N-1 elements each iteration)
- B[k,j]: Strided access (skip M-1 elements each iteration)
- No cache reuse, constant cache misses

**Cache-Friendly Implementation (i-k-j loop ordering):**
```
function matmul_ikj(A, B, C, M, N, K):
    C = zeros(M, N)
    
    for i = 0 to M:
        for k = 0 to K:
            a_val = A[i,k]  // Load once, use N times
            for j = 0 to N:
                C[i,j] += a_val × B[k,j]
```

**Performance:** ~50 GFLOPS (2.5% of peak)

**Why Better?**
- A[i,k]: Sequential access in outer loop
- B[k,j]: Sequential access in inner loop
- C[i,j]: Sequential writes in inner loop
- a_val held in register for N iterations

**Blocked/Tiled Implementation:**
```
function matmul_blocked(A, B, C, M, N, K):
    BLOCK = 64  // Tuned to cache size
    
    C = zeros(M, N)
    
    for i_block = 0 to M step BLOCK:
        for j_block = 0 to N step BLOCK:
            for k_block = 0 to K step BLOCK:
                // Process BLOCK×BLOCK sub-matrices
                for i = i_block to min(i_block+BLOCK, M):
                    for k = k_block to min(k_block+BLOCK, K):
                        a_val = A[i,k]
                        for j = j_block to min(j_block+BLOCK, N):
                            C[i,j] += a_val × B[k,j]
```

**Performance:** ~200 GFLOPS (10% of peak)

**Why Better?**
- Blocks sized to fit in L1/L2 cache (64×64×4 = 16 KB)
- Each block reused multiple times before eviction
- Reduces DRAM accesses by ~8x

**Implementation:** `src/core/kernels.zig`

**Production Approach: Use Optimized BLAS**

```
Performance comparison (M=N=K=1024):

Naive implementation:     0.5 GFLOPS
Blocked implementation:   50 GFLOPS
Intel MKL:               800 GFLOPS (95% of peak)
cuBLAS (GPU):          5,000 GFLOPS

Speedup: MKL is 1600x faster than naive!
```

**Why BLAS is so fast:**
- Hand-tuned assembly for specific CPUs
- Uses all SIMD lanes (AVX-512: 16 floats/cycle)
- Multi-threaded across cores
- Prefetching and cache optimization
- Special instructions (FMA: fused multiply-add)

**Linking with BLAS (production):**
```
// Zig example
const cblas = @cImport(@cInclude("cblas.h"));

function matmul_blas(A, B, C, M, N, K):
    cblas.cblas_sgemm(
        cblas.CblasRowMajor,     // Row-major layout
        cblas.CblasNoTrans,      // Don't transpose A
        cblas.CblasNoTrans,      // Don't transpose B
        M, N, K,                 // Dimensions
        1.0,                     // Alpha (scaling factor)
        A.ptr, K,                // A matrix and leading dimension
        B.ptr, N,                // B matrix and leading dimension
        0.0,                     // Beta (C scaling, 0 = overwrite)
        C.ptr, N                 // C matrix and leading dimension
    )
```

### Attention Kernel Implementation

**Standard Attention (with optimizations):**

```
function scaledDotProductAttention(Q, K, V, scale):
    // Input shapes: [batch, heads, seq_len, head_dim]
    
    batch = Q.shape[0]
    heads = Q.shape[1]
    seq_len_q = Q.shape[2]
    seq_len_k = K.shape[2]
    head_dim = Q.shape[3]
    
    // Allocate scores matrix (the memory bottleneck)
    scores = allocate(batch, heads, seq_len_q, seq_len_k)
    
    for b = 0 to batch:
        for h = 0 to heads:
            // Step 1: Compute Q @ K^T (use BLAS)
            Q_slice = Q[b, h, :, :]  // [seq_len_q, head_dim]
            K_slice = K[b, h, :, :]  // [seq_len_k, head_dim]
            
            // scores[b,h] = Q_slice @ K_slice^T
            cblas_sgemm(
                ...,
                seq_len_q, seq_len_k, head_dim,
                scale,  // Apply scaling in GEMM for efficiency
                Q_slice, K_slice^T,
                scores[b, h]
            )
            
            // Step 2: Apply softmax row-wise
            for i = 0 to seq_len_q:
                stableSoftmax(scores[b, h, i, :])
            
            // Step 3: Compute scores @ V (use BLAS)
            V_slice = V[b, h, :, :]  // [seq_len_k, head_dim]
            output_slice = output[b, h, :, :]  // [seq_len_q, head_dim]
            
            cblas_sgemm(
                ...,
                seq_len_q, head_dim, seq_len_k,
                1.0,
                scores[b, h], V_slice,
                output_slice
            )
    
    free(scores)
    return output
```

**Implementation:** `src/core/kernels.zig`

**Performance Breakdown:**
```
Sequence length = 512, head_dim = 64

Q @ K^T:  512 × 512 × 64 = 16.7M FLOPS
Softmax:  512 × 512 exp operations = 262K ops
scores @ V: 512 × 64 × 512 = 16.7M FLOPS

Total: ~33.4M FLOPS per head
For 12 heads: 401M FLOPS
On 1 TFLOP GPU: ~0.4ms (compute-bound)

Actual time: ~2ms (5x slower than theoretical)
Bottleneck: Memory bandwidth (loading Q, K, V, writing output)
```

### Next Section

Continue with [Part III - Vectorization and SIMD](./03_vectorization.md)
