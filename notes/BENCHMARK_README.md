# Tensor Performance Benchmark Suite

## Overview

Comprehensive performance benchmarking suite for the Tensor library with statistical analysis and CSV export.

## Features

### Statistical Metrics
- **Mean**: Average execution time across all iterations
- **Standard Deviation**: Measure of variance in execution times
- **Min/Max**: Fastest and slowest execution times
- **Iterations**: Number of test runs for each benchmark

### Output Formats
1. **Console Output**: Real-time results with statistics displayed during execution
2. **CSV Files**: 
   - `tensor_benchmark_results.csv` - Latest results (overwritten each run)
   - `tensor_benchmark_results_<timestamp>.csv` - Timestamped archive

### Benchmark Categories

1. **Construction** - Tensor creation and initialization (1D-4D)
2. **Copy** - Deep copy operations
3. **Element_Access** - Sequential element access patterns
4. **1D_Dot_Product** - Vector dot products (100 to 1M elements)
5. **2D_Matrix_Multiplication** - Matrix multiplication (various sizes)
6. **3D_Dot_1D** - 3D tensor × 1D vector contraction
7. **3D_Dot_2D** - 3D tensor × 2D matrix contraction
8. **4D_Dot_1D** - 4D tensor × 1D vector contraction
9. **4D_Dot_2D** - 4D tensor × 2D matrix contraction
10. **5D_Dot_1D** - 5D tensor × 1D vector contraction
11. **5D_Dot_2D** - 5D tensor × 2D matrix contraction

## Building and Running

```bash
# Build
cmake -B build -S .
cmake --build build

# Run benchmarks
./build/tensor_perf

# Or use custom target
cmake --build build --target run_perf
```

## CSV Output Format

```csv
Category,Benchmark,Mean(ms),StdDev(ms),Min(ms),Max(ms),Iterations,Total_Operations
Construction,"1D Tensor (1M elements)",1.234567,0.056789,1.123456,1.345678,5,1000000
...
```

## Console Output Example

```
=====================================
  Tensor Performance Benchmark Suite
=====================================

=== System Information ===
C++ Standard: 201703
Parallel STL: Available

=== 1D Vector Dot Product Benchmark ===
Running: Vector size 1000000 (5 iterations)
  Mean: 1.234 ms ± 0.057 ms
  Min: 1.123 ms, Max: 1.346 ms

...

Results written to: tensor_benchmark_results_1699468800.csv
Results written to: tensor_benchmark_results.csv
```

## Analyzing Results

The CSV files can be imported into:
- Excel/Google Sheets for visualization
- Python/R for statistical analysis
- Grafana for monitoring over time

Example Python analysis:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('tensor_benchmark_results.csv')
df_by_category = df.groupby('Category')['Mean(ms)'].mean()
df_by_category.plot(kind='bar')
plt.show()
```

## Performance Tips

1. Close other applications during benchmarking
2. Run multiple times and compare results
3. Check CPU frequency scaling settings
4. Ensure TBB is installed for best parallel performance
5. Monitor CPU temperature (thermal throttling affects results)

