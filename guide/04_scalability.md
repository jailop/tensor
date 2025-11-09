# Part IV: System Scalability and Deployment

> **Implementation:** `src/utils/memory_pool.zig`, build system configuration

## Scalability Considerations

Transformers face unique scaling challenges due to their quadratic attention complexity and large model sizes.

### Model Size Growth

**Trends in model scaling:**

```
Model          Parameters  Memory (FP32)  Memory (FP16)
GPT-2          1.5B        6 GB           3 GB
GPT-3 Small    125M        500 MB         250 MB
GPT-3          175B        700 GB         350 GB
GPT-4 (est)    1.7T        6.8 TB         3.4 TB
```

**Implications:**
- Single-GPU deployment only feasible for small models (<10B parameters)
- Model parallelism required for large models
- Inference optimization critical for production deployment

---

## Vertical Scaling (Single Machine)

### Memory Optimization

**1. Quantization**

Reduce precision while maintaining accuracy:

```pseudocode
// FP32: 4 bytes per parameter
weights_fp32 = load_model()

// FP16: 2 bytes per parameter (2x smaller)
weights_fp16 = quantize_fp16(weights_fp32)

// INT8: 1 byte per parameter (4x smaller)
weights_int8, scale = quantize_int8(weights_fp32)

// Mixed precision: critical layers in FP32, rest in INT8
for layer in model:
    if layer.is_critical:
        use_fp32(layer)
    else:
        use_int8(layer)
```

**Trade-offs:**
- FP16: ~1% accuracy loss, 2x memory reduction
- INT8: ~2-3% accuracy loss, 4x memory reduction
- Quantization-aware training can recover most accuracy

**Implementation:** Modern frameworks support mixed precision (PyTorch AMP, TensorFlow mixed precision)

**2. Gradient Checkpointing**

Trade compute for memory during training:

```pseudocode
// Standard forward pass: store all activations
function forward_standard(input, layers):
    activations = []
    x = input
    for layer in layers:
        x = layer.forward(x)
        activations.append(x)  // Memory grows with depth
    return x, activations

// Checkpointed forward: only store at checkpoints
function forward_checkpointed(input, layers, checkpoint_every=2):
    checkpoints = []
    x = input
    for i, layer in enumerate(layers):
        x = layer.forward(x)
        if i % checkpoint_every == 0:
            checkpoints.append(x)  // Only store every N layers
    return x, checkpoints

// Backward pass: recompute activations as needed
function backward_checkpointed(grad, checkpoints, layers):
    // Recompute forward from last checkpoint
    // Trade compute for memory
```

**Results:**
- Memory reduction: 50-80% for large models
- Compute overhead: 30-50% (one extra forward pass)
- Essential for training models that don't fit in GPU memory

**3. KV Cache Management**

For autoregressive generation:

```pseudocode
// Naive: recompute full sequence each step
function generate_naive(prompt, max_tokens):
    tokens = prompt
    for i in 0..max_tokens:
        // O(n²) complexity per token
        output = transformer(tokens)
        new_token = sample(output[-1])
        tokens.append(new_token)
    return tokens

// Optimized: cache key-value pairs
function generate_cached(prompt, max_tokens):
    tokens = prompt
    kv_cache = compute_kv(prompt)  // Only once
    
    for i in 0..max_tokens:
        // O(n) complexity per token
        output = transformer_incremental(tokens[-1], kv_cache)
        new_token = sample(output)
        tokens.append(new_token)
        update_kv_cache(kv_cache, new_token)
    
    return tokens
```

**Speedup:** 10-100x for long sequences

**Memory cost:**
```
KV cache size = 2 × num_layers × batch_size × seq_len × hidden_dim × bytes_per_param

Example (GPT-2):
  Layers: 12
  Batch: 8
  Seq length: 1024
  Hidden dim: 768
  Precision: FP16 (2 bytes)
  
  Size = 2 × 12 × 8 × 1024 × 768 × 2 = 301 MB per batch
```

**Implementation:** `src/utils/memory_pool.zig` provides tensor pooling for KV cache reuse

### CPU Optimization

**Multi-threading Strategies:**

```pseudocode
// 1. Layer-parallel (limited by depth)
function forward_layer_parallel(input, layers):
    // Can't parallelize sequential layers well
    // Only useful for batch processing
    for layer in layers:
        output = layer.forward(input)  // Each layer can use threads
        input = output
    return output

// 2. Attention head parallel (effective)
function multi_head_attention_parallel(Q, K, V, num_heads):
    head_dim = hidden_dim / num_heads
    results = parallel_for head in 0..num_heads:
        q = Q[:, head * head_dim:(head+1) * head_dim]
        k = K[:, head * head_dim:(head+1) * head_dim]
        v = V[:, head * head_dim:(head+1) * head_dim]
        attention_output = scaled_dot_product(q, k, v)
    concatenate(results)

// 3. Batch parallel (simple, effective)
function inference_batch_parallel(inputs, model):
    results = parallel_for input in inputs:
        model.forward(input)
    return results
```

**Thread scaling:**
```
Threads    Throughput    Efficiency
1          100 req/sec   100%
2          180 req/sec   90%
4          320 req/sec   80%
8          560 req/sec   70%
16         800 req/sec   50%
```

**Optimal:** 4-8 threads per model instance on modern CPUs

---

## Horizontal Scaling (Distributed)

### Data Parallelism

Replicate model across multiple GPUs/machines:

```pseudocode
// Each worker has full model copy
function data_parallel_training(dataset, model, num_workers):
    
    // Split data across workers
    worker_data = split(dataset, num_workers)
    
    for epoch in epochs:
        gradients = []
        
        // Each worker processes its batch
        parallel_for worker_id in 0..num_workers:
            batch = worker_data[worker_id].next_batch()
            local_grad = compute_gradients(batch, model)
            gradients.append(local_grad)
        
        // Synchronize: AllReduce gradients
        avg_gradient = allreduce(gradients) / num_workers
        
        // Update all model copies
        update_model(model, avg_gradient)
```

**Characteristics:**
- Effective batch size = batch_per_worker × num_workers
- Scales well to 8-16 GPUs
- Communication overhead grows with num_workers
- All workers need same model in memory

**Throughput:**
```
GPUs    Training throughput    Scaling efficiency
1       100 samples/sec        100%
2       190 samples/sec        95%
4       360 samples/sec        90%
8       640 samples/sec        80%
16      1100 samples/sec       69%
```

### Model Parallelism

Split model across multiple devices:

```pseudocode
// Vertical split: different layers on different devices
function pipeline_parallel(input, model, devices):
    // Device 0: layers 0-5
    // Device 1: layers 6-11
    // Device 2: layers 12-17
    // etc.
    
    x = input
    for device_id, layer_group in enumerate(model.layer_groups):
        x = transfer_to_device(x, devices[device_id])
        x = layer_group.forward(x)
    return x

// Horizontal split: split each layer across devices
function tensor_parallel(input, weight_matrix, devices):
    // Split weight matrix into columns
    weight_chunks = split_columns(weight_matrix, num_devices)
    
    results = []
    parallel_for device_id in 0..num_devices:
        chunk = weight_chunks[device_id]
        local_result = matmul(input, chunk)
        results.append(local_result)
    
    // Concatenate results
    return concatenate(results)
```

**Trade-offs:**

| Strategy | Memory/device | Communication | Scalability |
|----------|--------------|---------------|-------------|
| Pipeline | Model_size/N | Low (point-to-point) | Limited by depth |
| Tensor   | Model_size/N | High (all-to-all) | Good for wide layers |
| Hybrid   | Model_size/N² | Medium | Best for huge models |

### Inference Serving

**Pattern 1: Model Replication (small models)**

```
Load Balancer
     |
     +--> Server 1 (full model)
     +--> Server 2 (full model)
     +--> Server 3 (full model)
     +--> Server 4 (full model)
```

**Characteristics:**
- Simple deployment
- Linear scaling with servers
- Each server: 1-10ms latency
- Works for models <2GB

**Pattern 2: Model Sharding (large models)**

```
Request Router
     |
     +--> Shard 1 (layers 0-5)
          |
          +--> Shard 2 (layers 6-11)
               |
               +--> Shard 3 (layers 12-17)
```

**Characteristics:**
- Required for models >10GB
- Higher latency (network hops)
- Complex deployment
- Better resource utilization

**Pattern 3: Batch Aggregation**

```pseudocode
// Instead of processing requests immediately:
function batched_inference_server(request_queue, model, batch_size, max_wait_ms):
    
    while true:
        batch = []
        start_time = now()
        
        // Collect requests into batch
        while len(batch) < batch_size and (now() - start_time) < max_wait_ms:
            if request_queue.has_requests():
                batch.append(request_queue.pop())
        
        if len(batch) == 0:
            continue
        
        // Process batch efficiently
        inputs = [req.input for req in batch]
        outputs = model.forward_batch(inputs)
        
        // Return results
        for req, output in zip(batch, outputs):
            req.respond(output)
```

**Benefits:**
- 5-10x throughput improvement
- Better GPU utilization
- Trade latency for throughput

**Trade-offs:**
```
Batch size    Latency    Throughput
1             10 ms      100 req/s
4             15 ms      266 req/s
16            25 ms      640 req/s
64            80 ms      800 req/s
```

---

## Production Deployment Patterns

### Real-time API Serving

**Requirements:**
- Latency: <50ms p99
- Throughput: 100-1000 req/s
- Availability: 99.9%+

**Architecture:**

```
Client → [Load Balancer] → [Model Server Pool]
                          → [Model Server Pool]
                          → [Model Server Pool]
                            ↓
                        [KV Cache Store]
```

**Optimizations:**
- Quantization to INT8/FP16
- Small batch sizes (1-4)
- KV cache for multi-turn conversations
- Model replication for high availability

**Example setup (BERT-base classification):**
```
Model: BERT-base (110M params)
Quantization: INT8
Batch size: 1
Hardware: CPU (8 cores) or GPU (V100)

Latency: 15ms (CPU), 5ms (GPU)
Throughput: 65 req/s (CPU), 200 req/s (GPU)
Memory: 500MB (CPU), 2GB (GPU including cache)
Cost: $0.10/hr (CPU), $3.00/hr (GPU)
```

### Batch Processing

**Requirements:**
- Throughput: process millions of items
- Latency: hours to days acceptable
- Cost: minimize per-item cost

**Architecture:**

```
[Data Lake/Warehouse]
         ↓
[Batch Job Scheduler]
         ↓
[Worker Pool with large batches]
         ↓
[Results Storage]
```

**Optimizations:**
- Large batch sizes (64-256)
- Model parallelism for huge models
- Checkpointing for fault tolerance
- Spot instances for cost savings

**Example setup (document embedding):**
```
Model: BERT-base
Batch size: 128
Hardware: GPU (V100)

Throughput: 10,000 docs/min
Cost: $0.0003 per 1000 documents
Reliability: checkpoints every 10,000 docs
```

### Edge Deployment

**Requirements:**
- Model size: <100MB
- Latency: <100ms
- Power: battery-constrained
- Privacy: on-device inference

**Techniques:**
- Aggressive quantization (INT8, INT4)
- Model pruning (remove 30-50% of weights)
- Knowledge distillation (teacher → student)
- Mobile-optimized architectures (MobileBERT, DistilBERT)

**Example (mobile sentiment analysis):**
```
Original: BERT-base (110M params, 440MB)
Optimized: DistilBERT + INT8 (66M params, 33MB)

Accuracy: 94% vs 96% (original)
Latency: 80ms (iPhone 12)
Battery impact: 0.5% per 1000 inferences
```

---

## Monitoring and Optimization

### Key Metrics

**Latency metrics:**
```
- p50 (median): typical user experience
- p95: catch outliers
- p99: worst-case for most users
- p99.9: absolute worst-case
```

**Throughput metrics:**
```
- Requests per second
- Tokens processed per second
- GPU/CPU utilization
- Memory usage (peak, average)
```

**Quality metrics:**
```
- Model accuracy on production data
- Drift detection (input distribution changes)
- Calibration scores
- User feedback metrics
```

### Profiling Production Systems

```pseudocode
function profile_inference_request(input):
    timings = {}
    
    timings.total_start = now()
    
    timings.preprocessing_start = now()
    tokens = tokenize(input)
    timings.preprocessing = now() - timings.preprocessing_start
    
    timings.model_start = now()
    output = model.forward(tokens)
    timings.model = now() - timings.model_start
    
    timings.postprocessing_start = now()
    result = decode(output)
    timings.postprocessing = now() - timings.postprocessing_start
    
    timings.total = now() - timings.total_start
    
    log_metrics(timings)
    return result, timings
```

**Typical breakdown (BERT inference):**
```
Preprocessing:     2ms   (10%)
Model forward:    15ms   (75%)
Postprocessing:    3ms   (15%)
Total:            20ms
```

### Optimization Priorities

Based on profiling, focus on:

1. **If model forward dominates (>70%):**
   - Quantization
   - Model pruning
   - Better hardware

2. **If preprocessing dominates (>30%):**
   - Optimize tokenization
   - Batch tokenization
   - Cache common tokens

3. **If memory is bottleneck:**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use memory pooling

4. **If throughput is low:**
   - Increase batch size
   - Add more workers
   - Enable batching aggregation

---

## Cost Analysis

### Cloud Deployment Costs

**GPU-based serving (AWS p3.2xlarge, V100):**
```
Instance cost: $3.06/hr
Model: BERT-base
Throughput: 200 req/s
Uptime: 730 hrs/month

Monthly cost: $2,234
Cost per 1M requests: $0.43
```

**CPU-based serving (AWS c5.2xlarge, 8 vCPUs):**
```
Instance cost: $0.34/hr
Model: BERT-base (quantized INT8)
Throughput: 50 req/s
Uptime: 730 hrs/month

Monthly cost: $248
Cost per 1M requests: $0.23
```

**Serverless (AWS Lambda + custom runtime):**
```
Cost per request: $0.0000167 (1GB memory, 500ms)
Cold start penalty: 2-5s first request
Throughput: scales automatically

Cost per 1M requests: $16.70 (if no cold starts)
Best for: sporadic traffic, low QPS
```

### Training Costs

**BERT-base from scratch:**
```
Hardware: 8x V100 GPUs
Training time: 4 days
Instance: p3.16xlarge ($24.48/hr)

Total cost: ~$2,350
```

**GPT-3 equivalent (estimated):**
```
Hardware: 1024x A100 GPUs
Training time: 30 days
Cost: ~$4.6 million
```

---

## Deployment Checklist

Before production deployment:

- [ ] **Performance benchmarked** on representative data
- [ ] **Latency p99** meets requirements
- [ ] **Throughput** meets expected load
- [ ] **Memory usage** stays within limits
- [ ] **Error handling** for edge cases
- [ ] **Monitoring** in place (latency, throughput, errors)
- [ ] **A/B testing** framework ready
- [ ] **Rollback plan** if issues arise
- [ ] **Cost analysis** completed
- [ ] **Scaling strategy** defined
- [ ] **Load testing** at 2x expected traffic
- [ ] **Documentation** for ops team

---

**Next:** [Part V - Production Testing and Deployment](./05_production.md)
