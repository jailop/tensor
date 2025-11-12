#!/bin/bash

# Script to run tensor normalization benchmarks and save results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
RESULTS_DIR="${SCRIPT_DIR}/results"
BENCHMARK_BIN="${BUILD_DIR}/tensor_normalize_benchmark"

# Check if benchmark binary exists
if [ ! -f "$BENCHMARK_BIN" ]; then
    echo "Error: Benchmark binary not found at $BENCHMARK_BIN"
    echo "Please build the project first: cd build && make tensor_normalize_benchmark"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Generate timestamp for unique filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}"

echo "Running tensor normalization benchmarks..."
echo "Results will be saved to:"
echo "  JSON: ${OUTPUT_FILE}.json"
echo "  TXT:  ${OUTPUT_FILE}.txt"
echo ""

# Run benchmark and save results in both JSON and text format
"$BENCHMARK_BIN" \
    --benchmark_out="${OUTPUT_FILE}.json" \
    --benchmark_out_format=json \
    --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true \
    2>&1 | tee "${OUTPUT_FILE}.txt"

echo ""
echo "Benchmark completed!"
echo "Results saved to: ${OUTPUT_FILE}.{json,txt}"

# Compare with previous run if it exists
PREVIOUS_JSON="${RESULTS_DIR}/latest.json"
if [ -L "$PREVIOUS_JSON" ] && [ -e "$PREVIOUS_JSON" ]; then
    PREVIOUS_REAL=$(readlink -f "$PREVIOUS_JSON")
    CURRENT_JSON="${OUTPUT_FILE}.json"
    COMPARISON_FILE="${OUTPUT_FILE}_comparison.txt"
    
    echo ""
    echo "Comparing with previous run..."
    echo "Previous: $(basename $PREVIOUS_REAL)"
    echo "Current:  $(basename $CURRENT_JSON)"
    echo ""
    
    python3 "${SCRIPT_DIR}/compare_benchmarks.py" \
        "$PREVIOUS_REAL" \
        "$CURRENT_JSON" \
        "$COMPARISON_FILE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Comparison saved to: ${COMPARISON_FILE}"
        # Create symlink to latest comparison
        ln -sf "$(basename ${COMPARISON_FILE})" "${RESULTS_DIR}/latest_comparison.txt"
    fi
fi

# Create symlink to latest results
ln -sf "$(basename ${OUTPUT_FILE}.json)" "${RESULTS_DIR}/latest.json"
ln -sf "$(basename ${OUTPUT_FILE}.txt)" "${RESULTS_DIR}/latest.txt"

echo ""
echo "Latest results symlinked as: ${RESULTS_DIR}/latest.{json,txt}"
