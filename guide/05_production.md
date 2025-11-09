# Part V: Production Testing and Deployment

> **Implementation:** `tests/integration_tests.zig`, `src/bench.zig`

## Testing Strategy

Production transformer systems require comprehensive testing across multiple dimensions.

### Unit Testing

**Core kernel tests** (inline with implementation):

```pseudocode
test "softmax numerical stability":
    // Test with large values (should not overflow)
    input = [1000.0, 1001.0, 999.0]
    output = softmax(input)
    
    assert(all(is_finite(output)))
    assert(abs(sum(output) - 1.0) < 1e-6)
    
    // Test with small values (should not underflow)
    input = [-1000.0, -1001.0, -999.0]
    output = softmax(input)
    
    assert(all(is_finite(output)))
    assert(abs(sum(output) - 1.0) < 1e-6)

test "layer norm correctness":
    input = random_tensor(shape=[32, 768])
    output = layer_norm(input, epsilon=1e-5)
    
    // Output should have mean ≈ 0, variance ≈ 1
    for batch_idx in 0..32:
        mean = compute_mean(output[batch_idx])
        variance = compute_variance(output[batch_idx])
        
        assert(abs(mean) < 1e-4)
        assert(abs(variance - 1.0) < 1e-4)

test "attention output shape":
    batch_size = 8
    seq_len = 128
    hidden_dim = 768
    num_heads = 12
    
    Q = random_tensor([batch_size, seq_len, hidden_dim])
    K = random_tensor([batch_size, seq_len, hidden_dim])
    V = random_tensor([batch_size, seq_len, hidden_dim])
    
    output = multi_head_attention(Q, K, V, num_heads)
    
    assert(output.shape == [batch_size, seq_len, hidden_dim])
    assert(all(is_finite(output)))
```

**Implementation:** Unit tests are inline with kernel implementations in `src/core/kernels.zig`

### Integration Testing

**End-to-end transformer tests:**

```pseudocode
test "integration: full transformer forward pass":
    config = TransformerConfig{
        num_layers: 3,
        hidden_dim: 256,
        num_heads: 8,
        ff_dim: 1024,
        seq_len: 64,
    }
    
    model = create_transformer(config)
    input = random_tensor([4, 64, 256])  // batch=4, seq=64, hidden=256
    
    output = model.forward(input)
    
    // Verify output properties
    assert(output.shape == input.shape)
    assert(all(is_finite(output)))
    assert(not_all_same(output))  // Model actually transforms input
    
    // Verify output statistics are reasonable
    mean = compute_mean(output)
    std = compute_std(output)
    assert(abs(mean) < 1.0)
    assert(std > 0.1 and std < 10.0)

test "integration: stacked transformer blocks":
    // Test multiple layers compose correctly
    input = random_tensor([2, 32, 512])
    
    output = input
    for layer in 0..6:
        output = transformer_block(output, layer_weights[layer])
        
        // Each layer should maintain shape
        assert(output.shape == input.shape)
        assert(all(is_finite(output)))
    
    // Final output should be different from input
    assert(mean_squared_error(output, input) > 0.1)

test "integration: autoregressive generation with KV cache":
    model = load_model("gpt2-small")
    prompt = tokenize("The capital of France is")
    max_tokens = 10
    
    // Generate with caching
    cache = KVCache.init()
    tokens = prompt.clone()
    
    for i in 0..max_tokens:
        logits = model.forward_with_cache(tokens[-1:], cache)
        next_token = argmax(logits)
        tokens.append(next_token)
        
        // Verify cache grows
        assert(cache.size() == len(tokens))
    
    // Verify generation is deterministic with same seed
    cache2 = KVCache.init()
    tokens2 = generate_cached(model, prompt, max_tokens)
    assert(tokens == tokens2)

test "integration: batch processing consistency":
    // Batched results should match individual results
    inputs = [random_tensor([1, 64, 512]) for _ in 0..8]
    
    // Process individually
    individual_results = []
    for input in inputs:
        output = model.forward(input)
        individual_results.append(output)
    
    // Process as batch
    batched_input = concatenate(inputs, dim=0)
    batched_output = model.forward(batched_input)
    
    // Results should match
    for i in 0..8:
        diff = max_absolute_difference(
            individual_results[i],
            batched_output[i]
        )
        assert(diff < 1e-5)  // Small numerical difference acceptable
```

**Implementation:** `tests/integration_tests.zig`

### Numerical Stability Testing

```pseudocode
test "numerical stability: extreme inputs":
    // Very large values
    input_large = full_tensor([4, 32, 256], value=1000.0)
    output_large = model.forward(input_large)
    assert(all(is_finite(output_large)))
    
    // Very small values
    input_small = full_tensor([4, 32, 256], value=1e-8)
    output_small = model.forward(input_small)
    assert(all(is_finite(output_small)))
    
    // Mixed values
    input_mixed = random_tensor([4, 32, 256]) * 1000
    output_mixed = model.forward(input_mixed)
    assert(all(is_finite(output_mixed)))
    assert(no_nans(output_mixed))
    assert(no_infs(output_mixed))

test "numerical stability: zero inputs":
    input_zero = zeros([4, 32, 256])
    output = model.forward(input_zero)
    
    // Should not produce NaN or Inf
    assert(all(is_finite(output)))
    
    // Output should not be all zeros (bias terms should activate)
    assert(sum(abs(output)) > 1e-6)

test "gradient stability during training":
    model = create_trainable_model()
    input = random_tensor([8, 64, 512])
    target = random_tensor([8, 64, 512])
    
    output = model.forward(input)
    loss = mse_loss(output, target)
    gradients = backward(loss)
    
    // Check gradient magnitudes
    for param_name, grad in gradients:
        grad_norm = compute_norm(grad)
        assert(is_finite(grad_norm))
        assert(grad_norm < 100.0)  // Detect exploding gradients
        assert(grad_norm > 1e-7)   // Detect vanishing gradients
```

### Performance Regression Testing

```pseudocode
test "performance: attention latency bounds":
    config = {
        batch_size: 8,
        seq_len: 128,
        hidden_dim: 768,
        num_heads: 12
    }
    
    Q, K, V = create_test_tensors(config)
    
    // Warmup
    for _ in 0..10:
        _ = multi_head_attention(Q, K, V, config.num_heads)
    
    // Measure
    latencies = []
    for _ in 0..100:
        start = timer()
        output = multi_head_attention(Q, K, V, config.num_heads)
        latency = timer() - start
        latencies.append(latency)
    
    median_latency = percentile(latencies, 50)
    p99_latency = percentile(latencies, 99)
    
    // Assert performance bounds
    assert(median_latency < 5.0)   // 5ms median
    assert(p99_latency < 10.0)     // 10ms p99

test "performance: full model throughput":
    model = load_model("bert-base")
    batch_size = 32
    seq_len = 128
    
    inputs = random_tensor([batch_size, seq_len, 768])
    
    // Warmup
    for _ in 0..5:
        _ = model.forward(inputs)
    
    // Measure throughput
    start = timer()
    iterations = 50
    for _ in 0..iterations:
        _ = model.forward(inputs)
    elapsed = timer() - start
    
    throughput = (iterations * batch_size) / elapsed
    
    // Expect at least 100 samples/second
    assert(throughput > 100.0)

test "performance: memory pool efficiency":
    pool = MemoryPool.init(max_size=100_000_000)  // 100MB
    
    // Simulate workload: allocate and free tensors
    allocations = 0
    cache_hits = 0
    
    for _ in 0..1000:
        shape = random_shape()
        tensor = pool.allocate(shape)
        
        if tensor.was_reused:
            cache_hits += 1
        allocations += 1
        
        // Do some work
        fill_random(tensor)
        
        // Return to pool
        pool.free(tensor)
    
    cache_hit_rate = cache_hits / allocations
    
    // Expect >80% cache hit rate after warmup
    assert(cache_hit_rate > 0.8)
    
    // Memory usage should stay within bounds
    assert(pool.current_usage() < pool.max_size)
```

### Memory Alignment Testing

```pseudocode
test "memory alignment: tensor data aligned":
    tensor = Tensor.init([128, 768])
    
    // Check alignment (should be 64-byte aligned for AVX-512)
    address = get_memory_address(tensor.data)
    assert(address % 64 == 0)

test "memory alignment: SIMD operations valid":
    size = 1024
    data = allocate_aligned(size, alignment=64)
    fill_random(data)
    
    // Should not crash with SIMD operations
    result = simd_reduce_sum(data, size)
    assert(is_finite(result))
```

---

## Benchmarking

### Micro-benchmarks

**Individual kernel performance:**

```pseudocode
benchmark "softmax":
    sizes = [256, 512, 1024, 2048, 4096]
    
    for size in sizes:
        input = random_tensor([32, size])
        
        // Warmup
        for _ in 0..100:
            softmax(input)
        
        // Measure
        start = timer()
        for _ in 0..1000:
            softmax(input)
        elapsed = timer() - start
        
        throughput = (1000 * 32 * size) / elapsed
        print(f"Softmax {size}: {throughput:.2f} elements/sec")

benchmark "matrix multiplication":
    shapes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]
    
    for (M, N) in shapes:
        A = random_tensor([M, N])
        B = random_tensor([N, N])
        
        start = timer()
        for _ in 0..10:
            C = matmul(A, B)
        elapsed = timer() - start
        
        flops = (2 * M * N * N * 10) / elapsed
        gflops = flops / 1e9
        print(f"MatMul {M}x{N}: {gflops:.2f} GFLOPS")
```

### End-to-end Benchmarks

**Full model inference:**

```pseudocode
benchmark "bert-base inference":
    model = load_model("bert-base")
    batch_sizes = [1, 4, 16, 32, 64]
    seq_len = 128
    
    for batch_size in batch_sizes:
        input = random_tensor([batch_size, seq_len, 768])
        
        // Warmup
        for _ in 0..10:
            model.forward(input)
        
        // Measure
        latencies = []
        for _ in 0..100:
            start = timer()
            output = model.forward(input)
            latency = timer() - start
            latencies.append(latency)
        
        p50 = percentile(latencies, 50)
        p99 = percentile(latencies, 99)
        throughput = batch_size / p50
        
        print(f"Batch {batch_size}:")
        print(f"  p50: {p50*1000:.2f}ms")
        print(f"  p99: {p99*1000:.2f}ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")

benchmark "autoregressive generation":
    model = load_model("gpt2")
    prompt_lengths = [10, 50, 100, 200]
    generation_lengths = [10, 50, 100]
    
    for prompt_len in prompt_lengths:
        for gen_len in generation_lengths:
            prompt = random_tokens(prompt_len)
            
            // Measure with KV cache
            start = timer()
            for _ in 0..10:
                tokens = generate_with_cache(model, prompt, gen_len)
            elapsed_cached = timer() - start
            
            // Measure without cache (for comparison)
            start = timer()
            for _ in 0..10:
                tokens = generate_naive(model, prompt, gen_len)
            elapsed_naive = timer() - start
            
            speedup = elapsed_naive / elapsed_cached
            
            print(f"Prompt {prompt_len}, Generate {gen_len}:")
            print(f"  With cache: {elapsed_cached*100:.1f}ms")
            print(f"  Without: {elapsed_naive*100:.1f}ms")
            print(f"  Speedup: {speedup:.1f}x")
```

**Implementation:** `src/bench.zig`

---

## Production Deployment

### Pre-deployment Checklist

**Model validation:**
- [ ] Model accuracy validated on test set
- [ ] Accuracy on production-like data checked
- [ ] Edge cases tested (empty input, very long sequences, special characters)
- [ ] Numerical stability verified
- [ ] Model size and memory footprint acceptable

**Performance validation:**
- [ ] Latency meets SLA (p50, p95, p99)
- [ ] Throughput meets expected load
- [ ] Resource usage (CPU, GPU, memory) within limits
- [ ] Performance regression tests passing
- [ ] Load testing at 2x expected traffic completed

**Infrastructure:**
- [ ] Monitoring and alerting configured
- [ ] Logging strategy in place
- [ ] Error handling and retry logic implemented
- [ ] Health check endpoints working
- [ ] Graceful shutdown implemented
- [ ] Auto-scaling configured (if applicable)

**Operational:**
- [ ] Deployment runbook created
- [ ] Rollback procedure documented
- [ ] On-call rotation established
- [ ] Documentation for ops team complete
- [ ] Disaster recovery plan in place

### Deployment Strategies

**1. Blue-Green Deployment**

```
    [Load Balancer]
         |
         +--> [Blue (v1.0)] ←-- Current production
         |
         +--> [Green (v1.1)] ←-- New version (idle)

Step 1: Deploy v1.1 to Green
Step 2: Run smoke tests on Green
Step 3: Switch traffic to Green
Step 4: Monitor for issues
Step 5: Keep Blue as rollback option for 24h
```

**2. Canary Deployment**

```
    [Load Balancer]
         |
         +--> [v1.0] (95% traffic)
         |
         +--> [v1.1] (5% traffic) ←-- Canary

Gradually increase canary traffic:
  5% → 10% → 25% → 50% → 100%

Monitor at each step:
  - Error rate
  - Latency (p50, p99)
  - Model accuracy
  - User feedback
```

**3. Shadow Deployment**

```
    [Load Balancer]
         |
         +--> [v1.0] ←-- Serves real traffic
         |
         +--> [v1.1] ←-- Receives copy of traffic (results ignored)

Compare:
  - Latency distributions
  - Error rates
  - Output differences
```

### Monitoring

**Key metrics to track:**

```pseudocode
// Request-level metrics
log_request_metrics({
    request_id: uuid,
    model_version: "v1.2.3",
    input_length: tokens.length,
    output_length: result.length,
    latency_ms: elapsed,
    
    // Detailed timings
    preprocessing_ms: preproc_time,
    model_forward_ms: forward_time,
    postprocessing_ms: postproc_time,
    
    // Resource usage
    memory_peak_mb: peak_memory,
    gpu_utilization: gpu_util,
})

// Aggregate metrics (per minute)
aggregate_metrics({
    timestamp: now,
    requests_per_sec: rps,
    
    // Latency distribution
    latency_p50: p50,
    latency_p95: p95,
    latency_p99: p99,
    latency_p999: p999,
    
    // Error tracking
    error_rate: errors / total_requests,
    timeout_rate: timeouts / total_requests,
    
    // Resource utilization
    cpu_avg: avg_cpu,
    memory_avg_mb: avg_memory,
    gpu_utilization_avg: avg_gpu_util,
    
    // Model-specific
    avg_input_length: avg_input_len,
    avg_output_length: avg_output_len,
})
```

**Alerts:**

```yaml
# Latency alert
- name: high_latency
  condition: latency_p99 > 100ms for 5 minutes
  severity: warning
  action: page on-call

# Error rate alert
- name: high_error_rate
  condition: error_rate > 1% for 3 minutes
  severity: critical
  action: page on-call, auto-rollback

# Throughput alert
- name: low_throughput
  condition: requests_per_sec < 50 for 5 minutes
  severity: warning
  action: notify team

# Resource exhaustion
- name: high_memory
  condition: memory_usage > 90% for 2 minutes
  severity: critical
  action: auto-scale or restart
```

### Incident Response

**When something goes wrong:**

```pseudocode
procedure handle_production_incident():
    
    // 1. Detect and alert
    if detect_anomaly():
        send_alert(on_call_team)
        create_incident_ticket()
    
    // 2. Immediate mitigation
    if error_rate > 10%:
        trigger_automatic_rollback()
    elif latency_p99 > 500ms:
        increase_timeout_limits()
        scale_up_instances()
    elif memory_usage > 95%:
        restart_gracefully()
    
    // 3. Investigate
    check_recent_changes()
    analyze_error_logs()
    compare_metrics_to_baseline()
    check_dependencies()
    
    // 4. Fix
    if issue_identified:
        apply_fix()
        test_in_staging()
        deploy_fix()
    else:
        rollback_to_last_known_good()
    
    // 5. Post-mortem
    document_incident()
    identify_root_cause()
    create_action_items()
    update_runbooks()
```

---

## Continuous Improvement

### A/B Testing Models

```pseudocode
function ab_test_new_model(traffic_split=0.05):
    
    // Route traffic
    for request in incoming_requests:
        if random() < traffic_split:
            result = model_b.forward(request)
            log_result(request, result, model="B")
        else:
            result = model_a.forward(request)
            log_result(request, result, model="A")
    
    // After 1 week, analyze results
    results_a = query_logs(model="A", days=7)
    results_b = query_logs(model="B", days=7)
    
    comparison = {
        latency: compare_latency(results_a, results_b),
        accuracy: compare_accuracy(results_a, results_b),
        user_engagement: compare_engagement(results_a, results_b),
    }
    
    if all_metrics_better(comparison):
        promote_model_b_to_production()
    else:
        keep_model_a()
```

### Performance Optimization Cycle

```
1. Profile production traffic
   ↓
2. Identify bottlenecks
   ↓
3. Implement optimization
   ↓
4. Test in staging
   ↓
5. Canary deploy
   ↓
6. Validate improvement
   ↓
7. Full rollout or rollback
   ↓
(repeat)
```

### Model Updates

**Regular update cadence:**

```
Weekly:
  - Retrain on recent data (if applicable)
  - Update tokenizer vocabulary
  - Fine-tune on user feedback

Monthly:
  - Evaluate new model architectures
  - Benchmark against competition
  - Review performance metrics

Quarterly:
  - Major model version updates
  - Infrastructure upgrades
  - Cost optimization review
```

---

## Best Practices Summary

### Code Quality
- Unit test all kernels with edge cases
- Integration test full pipelines
- Benchmark performance regularly
- Profile before optimizing
- Document assumptions and trade-offs

### Deployment
- Always have a rollback plan
- Monitor everything
- Start with small traffic percentage
- Validate on production-like data
- Keep deployment simple

### Operations
- Automate alerting and monitoring
- Document incident procedures
- Regular load testing
- Capacity planning
- Cost tracking

### Continuous Improvement
- A/B test model changes
- Profile production workloads
- Iterate on performance
- Learn from incidents
- Share knowledge

---

## Final Deployment Example

**Complete deployment workflow for a BERT classifier:**

```bash
# 1. Prepare model
./scripts/export_model.sh bert-base-classifier v2.1.0
./scripts/quantize_model.sh v2.1.0 int8

# 2. Test locally
zig build test-all
zig build bench
./scripts/validate_accuracy.sh v2.1.0

# 3. Deploy to staging
./scripts/deploy_staging.sh v2.1.0
./scripts/smoke_test_staging.sh

# 4. Canary deploy
./scripts/deploy_canary.sh v2.1.0 --traffic=5%
./scripts/monitor_canary.sh --duration=1h

# 5. Gradual rollout
./scripts/increase_traffic.sh v2.1.0 --to=25%  # Monitor 2h
./scripts/increase_traffic.sh v2.1.0 --to=50%  # Monitor 2h
./scripts/increase_traffic.sh v2.1.0 --to=100% # Full rollout

# 6. Monitor and validate
./scripts/validate_production.sh v2.1.0

# 7. If issues: rollback
./scripts/rollback.sh v2.0.9
```

---

**This completes the Transformer Systems Guide.**

**Key Takeaways:**
1. Implementation matters more than theory
2. Profile before optimizing
3. Test everything, especially edge cases
4. Monitor in production
5. Always have a rollback plan

**See also:**
- `src/` - Full implementation
- `tests/` - Test suite
- `README.md` - Quick start guide
- `build.zig` - Build system
