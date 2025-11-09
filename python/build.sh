#!/bin/bash
# Build script for Tensor4D Python bindings

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Building Tensor4D Python Bindings ==="
echo ""

# Check for pybind11
echo "Checking for pybind11..."
if python3 -c "import pybind11" 2>/dev/null; then
    echo "  ✓ pybind11 found"
else
    echo "  ✗ pybind11 not found"
    echo ""
    echo "Installing pybind11..."
    pip3 install pybind11
fi

# Check for NumPy
echo "Checking for NumPy..."
if python3 -c "import numpy" 2>/dev/null; then
    echo "  ✓ NumPy found"
else
    echo "  ✗ NumPy not found"
    echo ""
    echo "Installing NumPy..."
    pip3 install numpy
fi

echo ""
echo "Building extension module..."
echo ""

# Build using setup.py
python3 setup.py build_ext --inplace

echo ""
echo "=== Build Complete ==="
echo ""
echo "Test the installation with:"
echo "  python3 test_bindings.py"
echo ""
echo "Or run examples:"
echo "  python3 example_basic.py"
echo "  python3 example_autograd.py"
echo "  python3 example_linalg.py"
echo ""
