#!/bin/bash

echo "=================================================="
echo "Running All NumPy Interoperability Tests"
echo "=================================================="
echo ""

# Test 1: NumPy interop test suite
echo "1. Running NumPy interoperability test suite..."
python3 test_numpy_interop.py
if [ $? -ne 0 ]; then
    echo "❌ NumPy interop tests FAILED"
    exit 1
fi
echo "✓ NumPy interop tests PASSED"
echo ""

# Test 2: Existing bindings test suite
echo "2. Running existing bindings test suite..."
python3 test_bindings.py > /tmp/test_bindings.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Bindings tests FAILED"
    cat /tmp/test_bindings.log
    exit 1
fi
echo "✓ Bindings tests PASSED"
echo ""

# Test 3: NumPy interop examples
echo "3. Running NumPy interop examples..."
python3 example_numpy_interop.py > /tmp/example_numpy.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ NumPy interop example FAILED"
    cat /tmp/example_numpy.log
    exit 1
fi
echo "✓ NumPy interop example PASSED"
echo ""

# Test 4: Real-world example
echo "4. Running real-world pipeline example..."
python3 example_realworld.py > /tmp/example_realworld.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Real-world example FAILED"
    cat /tmp/example_realworld.log
    exit 1
fi
echo "✓ Real-world example PASSED"
echo ""

# Summary
echo "=================================================="
echo "✓ All NumPy Interoperability Tests PASSED!"
echo "=================================================="
echo ""
echo "Features verified:"
echo "  ✓ NumPy to Tensor conversion (from_numpy)"
echo "  ✓ Tensor to NumPy conversion (numpy())"
echo "  ✓ Auto-detection in constructor"
echo "  ✓ All tensor types (1D, 2D, 3D, 4D)"
echo "  ✓ Operations compatibility"
echo "  ✓ Math functions"
echo "  ✓ Real-world pipeline integration"
echo ""
