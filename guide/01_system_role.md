# Part I: The Transformer's System Role

> **Implementation Note:** Complete Zig implementation in `../src/`.
> This section establishes the context for systems developers.

## Algorithm's Core Function

Transformers revolutionized sequence processing by introducing the
**self-attention mechanism** as an alternative to recurrent
architectures. Unlike RNNs and LSTMs that process sequences sequentially
(introducing data dependencies that prevent parallelization),
transformers can attend to all positions simultaneously.

### What Problem Does It Solve?

**Sequential Data Bottleneck:** Traditional RNNs suffer from:

1. **Sequential Processing:** Each token must wait for the previous
   token's computation
2. **Gradient Vanishing:** Long sequences lose information through many
   recurrent steps
3. **Limited Parallelization:** Cannot leverage GPU parallel compute
   effectively

**Transformer Solution:**

- **Parallel Processing:** All tokens processed simultaneously
- **Direct Long-Range Dependencies:** Any token can directly attend to
  any other token
- **Hardware Efficiency:** Matrix operations map perfectly to GPU/TPU
  architectures

### Real-World Applications

**Production Use Cases:**

- **Machine Translation:** Google Translate, DeepL (100M+ daily
  translations)
- **Text Generation:** GPT models, code completion (GitHub Copilot)
- **Search Ranking:** BERT-based relevance scoring at scale
- **Speech Recognition:** Whisper, wav2vec for audio transcription
- **Computer Vision:** Vision Transformers (ViT) for image
  classification

---

## System Impact

Understanding where transformers fit in production pipelines is critical
for optimization decisions.

### Deployment Patterns

#### 1. **Low-Latency Inference APIs**

**Use Case:** Real-time autocomplete, chatbots, translation services

**Requirements:**
- **Latency:** P95 < 100ms (often <50ms for interactive)
- **Throughput:** 100-1000 QPS per instance
- **Cost:** Minimize compute per request

**System Characteristics:**
```
Request Flow:
  Client → Load Balancer → API Gateway → Inference Service → Model
  
Constraints:
  - Small batch sizes (1-4) to minimize queueing delay
  - KV caching essential (reduces computation by 50-70%)
  - Model quantization (INT8) to fit in GPU memory
  - Multiple model replicas for horizontal scaling
```

**Example Configuration:**

```
Model: BERT-base (110M parameters)
Hardware: NVIDIA T4 GPU
Batch Size: 4
Sequence Length: 128
Latency: P50=15ms, P95=30ms, P99=45ms
Throughput: ~250 QPS per GPU
Cost: $0.35/hour × 4 GPUs = $1.40/hour for 1000 QPS
```

#### 2. **Batch Processing Pipelines**

**Use Case:** Document classification, embedding generation, data labeling

**Requirements:**
- **Throughput:** Process millions of documents daily
- **Cost Efficiency:** Maximize GPU utilization
- **Fault Tolerance:** Checkpointing and retry logic

**System Characteristics:**

```
Pipeline:
  Data Lake → Batch Loader → Preprocessing → Model Inference → Results Store
  
Optimization:
  - Large batch sizes (32-128) to maximize GPU usage
  - Dynamic batching to reduce padding waste
  - Prefetching and pipelining to hide I/O latency
  - Spot instances for cost savings (70% cheaper)
```

**Example Configuration:**

```
Model: RoBERTa-large (355M parameters)
Hardware: NVIDIA A100 GPU (40GB)
Batch Size: 64
Sequence Length: 512
Throughput: 25M documents/day per GPU
Cost: $2.00/hour on spot = $48/day
Daily Volume: 25M docs × $48/day ÷ 1 GPU = $0.00192 per 1000 docs
```

#### 3. **Distributed Training**

**Use Case:** Pre-training large language models, fine-tuning on custom data

**Requirements:**
- **Scale:** Billion+ parameter models across hundreds of GPUs
- **Efficiency:** Minimize communication overhead
- **Reliability:** Handle hardware failures gracefully

**System Characteristics:**
```
Architecture:
  Parameter Server or All-Reduce (Ring-AllReduce, Tree-AllReduce)
  
Parallelism Strategies:
  - Data Parallelism: Different batches per GPU
  - Model Parallelism: Model split across GPUs (tensor/pipeline)
  - Mixed: Hybrid approaches for largest models
  
Communication:
  - InfiniBand/NVLink for low-latency GPU-to-GPU
  - Gradient compression to reduce bandwidth
  - Overlapping compute and communication
```

**Example Configuration:**
```
Model: GPT-3 (175B parameters)
Hardware: 1024× NVIDIA A100 GPUs (80GB)
Training Time: 34 days for 300B tokens
Cost: ~$4.6M USD (at $2/hr per GPU)
Communication: 600GB/s InfiniBand, 3TB/s NVLink aggregate
```

---

## Critical Non-Functional Requirements

### Latency Requirements

**Interactive Applications (<50ms):**
- Autocomplete, chatbots, real-time translation
- Every 100ms delay = 1% conversion drop (Amazon research)
- User perception threshold: 100ms feels "instant", 1s feels "slow"

**Breakdown Analysis:**
```
Total Budget: 50ms
  - Network RTT: 10ms (client ↔ server)
  - Load balancing: 2ms
  - Input preprocessing: 3ms (tokenization, padding)
  - Model inference: 25ms (CRITICAL PATH)
  - Output postprocessing: 5ms (decoding, formatting)
  - Response serialization: 5ms
  
Optimization Focus: Model inference (50% of budget)
```

**Techniques:**
- KV caching: 2-3x speedup for autoregressive generation
- Operator fusion: 20-30% reduction via custom kernels
- Mixed precision (FP16): 2x speedup on Tensor Cores
- Quantization (INT8): 3-4x speedup with <1% accuracy loss

### Throughput Requirements

**Batch Processing (1000+ sequences/sec):**
- Must process large datasets economically
- GPU utilization target: >80% (otherwise cost-inefficient)

**Scaling Math:**
```
Single GPU (A100):
  Latency per batch: 40ms (batch=32, seq=128)
  Batches per second: 1000ms / 40ms = 25
  Sequences per second: 25 × 32 = 800 seq/s
  
To achieve 10,000 seq/s:
  GPUs needed: 10,000 / 800 = 12.5 → 13 GPUs
  Cost: 13 × $2/hr = $26/hr
  Cost per million: $26 × (1M / 10K / 3600) = $0.72
```

**Optimization Techniques:**
- Dynamic batching: Group requests to minimize padding
- Multi-stream execution: Overlap independent operations
- Pipeline parallelism: Process different stages concurrently

### Memory Ceiling

**The Quadratic Problem:**

Attention memory grows as O(N²) where N = sequence length:
```
Memory = batch × heads × seq_len² × sizeof(float32)

Example (batch=32, heads=16, seq=2048):
  Attention scores: 32 × 16 × 2048² × 4 bytes = 8.6 GB
  
With 4 layers: 34.4 GB just for attention!
```

**Mitigation Strategies:**

1. **Sparse Attention Patterns:**
   - Local windows: Attend to nearby tokens only
   - Strided patterns: Skip tokens with regular intervals
   - Complexity: O(N²) → O(N log N) or O(N)

2. **Flash Attention:**
   - Tiled computation: Process in blocks
   - Recomputation: Trade compute for memory
   - Result: O(N²) → O(N) memory, 2-4x faster

3. **Gradient Checkpointing:**
   - Store only activations at layer boundaries
   - Recompute intermediate activations during backward
   - Memory: O(L × N²) → O(L) where L = layers

---

## Mathematical Concept

The core innovation is **scaled dot-product attention**. Understanding
this mathematically informs all implementation choices.

### The Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Component Breakdown:**

**Query (Q), Key (K), Value (V):**

- Linear projections of input embeddings
- Shape: [batch, seq_len, d_model] → [batch, seq_len, d_k]
- Purpose: Q asks "what am I looking for?", K answers "what do I
  offer?", V provides "here's the information"

**Dot Product (QK^T):**

- Computes similarity between all pairs of positions
- Shape: [batch, seq_len, d_k] @ [batch, d_k, seq_len] → [batch,
  seq_len, seq_len]
- Computational cost: O(N² × d_k) - this is the bottleneck!

**Scaling (1/√d_k):**

- Without scaling, dot products grow with dimension
- Large values → steep softmax gradients → training instability
- Example: d_k=64, typical dot product magnitude ~8; softmax(8)
  dominates softmax(7.9)

**Softmax:**

- Converts scores to probability distribution
- Numerical stability critical (subtract max before exp)
- Most "important" positions get highest weights

**Weighted Sum (× V):**

- Combines value vectors based on attention weights
- Shape: [batch, seq_len, seq_len] @ [batch, seq_len, d_v] → [batch,
  seq_len, d_v]

### Multi-Head Attention

Instead of single attention, run **h** parallel attention mechanisms:

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W_O

where head_i = Attention(QW_Qi, KW_Ki, VW_Vi)
```

**Why Multiple Heads?**

1. **Diverse Representations:** Each head learns different patterns
   - Head 1: Syntactic dependencies (subject-verb agreement)
   - Head 2: Semantic relationships (synonyms, antonyms)
   - Head 3: Positional patterns (adjacent words)

2. **Parallel Computation:** Heads are independent → SIMD-friendly

3. **Smaller Dimensions:** d_k = d_model / h (typically 64 with h=8-16)
   - Reduces per-head computation
   - Better cache locality

**Computational Complexity:**

```
Single-head attention: O(N² × d_model)
Multi-head attention: O(h × N² × d_k) = O(h × N² × d_model/h) = O(N² × d_model)

Same complexity, but better representation power!
```

### Implementation Impact

**The formula directly determines:**

1. **Memory Layout:**
   - Need contiguous [batch, seq, dim] tensors for efficient matmul
   - Attention scores [seq, seq] must fit in fast memory

2. **Kernel Bottlenecks:**
   - 60-70%: Matrix multiplications (BLAS/cuBLAS critical)
   - 10-15%: Softmax (numerical stability matters)
   - Remaining: Data movement, layer norm, activations

3. **Optimization Opportunities:**
   - Fuse Q, K, V projections into single GEMM
   - Fuse softmax with subsequent operations
   - Use Tensor Cores for mixed precision (FP16)

4. **Numerical Precision:**
   - FP32: Safe but slow
   - FP16: 2x faster, but softmax needs careful handling
   - INT8: 4x faster, requires calibration

### Concrete Example

```
Input: "The cat sat on the mat"
Tokens: [the, cat, sat, on, the, mat]
d_model = 512, h = 8, d_k = 64

Step 1: Embed and project to Q, K, V
  Input: [1, 6, 512]  (batch=1, seq=6, dim=512)
  Q = Input @ W_Q: [1, 6, 512] → [1, 8, 6, 64]  (8 heads, 64 dim each)
  K = Input @ W_K: [1, 6, 512] → [1, 8, 6, 64]
  V = Input @ W_V: [1, 6, 512] → [1, 8, 6, 64]

Step 2: Compute attention scores
  Scores = Q @ K^T / √64: [1, 8, 6, 64] @ [1, 8, 64, 6] → [1, 8, 6, 6]
  
  Example scores for token "cat" (position 1):
    [0.1, 0.7, 0.05, 0.05, 0.05, 0.05]  (cat attends strongly to itself)

Step 3: Softmax normalization
  Attention weights: [0.15, 0.52, 0.08, 0.08, 0.08, 0.09]  (sums to 1.0)

Step 4: Weighted combination
  Output = Attention @ V: [1, 8, 6, 6] @ [1, 8, 6, 64] → [1, 8, 6, 64]
  
  For "cat": 0.52×V_cat + 0.15×V_the + 0.09×V_mat + ...

Step 5: Concatenate heads and project
  Concat: [1, 8, 6, 64] → [1, 6, 512]
  Output = Concat @ W_O: [1, 6, 512] → [1, 6, 512]
```

**Memory Requirements:**
```
Weights: 4 × (512 × 512) × 4 bytes = 4 MB (Q, K, V, O projections)
Activations per layer:
  - Q, K, V: 3 × 6 × 512 × 4 = 36 KB
  - Attention scores: 8 × 6 × 6 × 4 = 1.15 KB
  - Output: 6 × 512 × 4 = 12 KB

Total per layer: ~4 MB (weights dominate)
For 12 layers: ~48 MB

Scaling to seq=2048:
  Attention scores: 8 × 2048 × 2048 × 4 = 128 MB (!)
  With 12 layers: 1.5 GB just for attention scores
```

---

## Key Takeaways for Systems Developers

1. **Attention is the bottleneck:** 60%+ of compute time
2. **Memory scales quadratically:** Limits maximum sequence length
3. **Batch size trades latency for throughput:** Optimize based on use case
4. **Numerical stability matters:** FP16 requires careful softmax implementation
5. **Hardware matters:** GPUs 10-100x faster than CPUs for transformers
6. **Caching is critical:** KV cache reduces inference cost by 2-3x

**Next:** [Part II - Low-Level Implementation Strategy](./02_implementation_strategy.md)
