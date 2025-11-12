# Tensor Normalization Optimization Plan

## Current Implementation Analysis

### Architecture Overview
- **Backend Priority**: GPU > BLAS > CPU
- **Memory Management**: Tensors use `std::shared_ptr` with custom deleters
- **GPU Flag**: `uses_gpu()` indicates data location
- **Data Access**: `data_ptr()` returns raw pointer to data (CPU or GPU memory)

### Current Normalization Implementation
All normalization functions (L1, L2, Z-score, Min-Max) currently use:
- **Pure CPU implementation** with sequential loops
- **No GPU kernels**
- **No BLAS utilization**
- **No parallel execution**

### Performance Characteristics
1. **axis=-1 (all elements)**: 
   - Single reduction over entire tensor
   - Good candidate for GPU parallel reduction
   
2. **axis=N (specific axis)**:
   - Multiple independent reductions
   - Excellent parallelization opportunity
   - Each slice can be processed independently

## Optimization Strategy

### Phase 1: GPU Kernels
Create CUDA kernels for:
- L1 norm computation and normalization
- L2 norm computation and normalization  
- Z-score normalization (mean, variance, normalize)
- Min-Max normalization (find min/max, scale)

### Phase 2: BLAS Integration
Use BLAS for:
- Vector norms (dnrm2, snrm2)
- Sum operations (dasum, sasum for L1)
- Dot products for variance calculation

### Phase 3: CPU Parallelization
Use C++17 parallel algorithms for CPU fallback:
- `std::reduce` for reductions
- `std::transform_reduce` for compound operations
- `std::transform` with `std::execution::par` for element-wise ops

### Phase 4: Smart Data Management
- Keep data in GPU when tensor uses GPU
- Avoid unnecessary CPU↔GPU transfers
- Use direct GPU operations when available
- Fall back to CPU only when necessary

## Implementation Details

### File Structure
```
include/
  tensor_normalize.h         - Main header (declarations)
  tensor_normalize_gpu.cuh   - GPU kernel declarations
src/
  tensor_normalize_gpu.cu    - GPU kernel implementations
```

### Backend Selection Logic
```cpp
if (tensor.uses_gpu() && is_gpu_available()) {
    // Use GPU kernels
} else if (is_blas_available()) {
    // Use BLAS
} else {
    // Use parallel CPU
}
```

### Memory Safety
- GPU tensors stay on GPU throughout operation
- Results inherit input tensor's GPU flag
- No data movement unless backend mismatch

## Expected Performance Improvements

### GPU vs CPU (baseline)
- Small tensors (1K): 2-5x faster
- Medium tensors (100K): 10-20x faster  
- Large tensors (1M+): 50-100x faster

### BLAS vs CPU
- 2-4x improvement for reductions
- 3-6x improvement for complex operations

### Parallel CPU vs Sequential
- 2-4x on 4-core systems
- 4-8x on 8+ core systems

## Testing Strategy
1. Verify correctness against current implementation
2. Benchmark each backend separately
3. Test data location preservation
4. Validate mixed-backend scenarios
5. Compare against baseline benchmarks
