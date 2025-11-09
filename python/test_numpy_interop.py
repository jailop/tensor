#!/usr/bin/env python3
"""Test NumPy interoperability with Tensor4D library"""

import sys
import numpy as np
import tensor4d as t4d

def test_vector_numpy_conversion():
    """Test Vector <-> NumPy conversion"""
    print("Testing Vector <-> NumPy conversion...")
    
    # Create NumPy array
    np_vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    print(f"  NumPy vector: {np_vec}")
    
    # Convert to Vectorf
    vec = t4d.Vectorf.from_numpy(np_vec)
    print(f"  Vectorf shape: {vec.shape}")
    print(f"  Vectorf values: [{vec.at([0])}, {vec.at([1])}, {vec.at([2])}, {vec.at([3])}]")
    
    # Convert back to NumPy
    np_result = vec.numpy()
    print(f"  Back to NumPy: {np_result}")
    
    # Verify
    assert np.allclose(np_vec, np_result), "Round-trip conversion failed"
    print("  ✓ Vector conversion successful\n")

def test_matrix_numpy_conversion():
    """Test Matrix <-> NumPy conversion"""
    print("Testing Matrix <-> NumPy conversion...")
    
    # Create NumPy matrix
    np_mat = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    print(f"  NumPy matrix shape: {np_mat.shape}")
    print(f"  NumPy matrix:\n{np_mat}")
    
    # Convert to Matrixf
    mat = t4d.Matrixf.from_numpy(np_mat)
    print(f"  Matrixf shape: {mat.shape}")
    
    # Convert back to NumPy
    np_result = mat.numpy()
    print(f"  Back to NumPy:\n{np_result}")
    
    # Verify
    assert np.allclose(np_mat, np_result), "Round-trip conversion failed"
    print("  ✓ Matrix conversion successful\n")

def test_tensor3_numpy_conversion():
    """Test Tensor3 <-> NumPy conversion"""
    print("Testing Tensor3 <-> NumPy conversion...")
    
    # Create 3D NumPy array
    np_t3 = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    print(f"  NumPy tensor3 shape: {np_t3.shape}")
    
    # Convert to Tensor3d
    t3 = t4d.Tensor3d.from_numpy(np_t3)
    print(f"  Tensor3d shape: {t3.shape}")
    
    # Convert back to NumPy
    np_result = t3.numpy()
    print(f"  Back to NumPy shape: {np_result.shape}")
    
    # Verify
    assert np.allclose(np_t3, np_result), "Round-trip conversion failed"
    print("  ✓ Tensor3 conversion successful\n")

def test_constructor_with_numpy():
    """Test creating tensors directly from NumPy arrays"""
    print("Testing tensor construction from NumPy...")
    
    # Create from NumPy array
    np_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mat = t4d.Matrixf(np_data)
    
    print(f"  Created Matrixf from NumPy array")
    print(f"  Shape: {mat.shape}")
    
    # Verify
    np_result = mat.numpy()
    assert np.allclose(np_data, np_result), "Constructor conversion failed"
    print("  ✓ Constructor with NumPy successful\n")

def test_operations_with_numpy():
    """Test that operations preserve NumPy compatibility"""
    print("Testing operations with NumPy conversion...")
    
    # Create tensors from NumPy
    np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    
    a = t4d.Vectorf.from_numpy(np_a)
    b = t4d.Vectorf.from_numpy(np_b)
    
    # Perform operations
    c = a + b
    d = a * b
    
    # Convert back and verify
    np_c = c.numpy()
    np_d = d.numpy()
    
    expected_c = np_a + np_b
    expected_d = np_a * np_b
    
    assert np.allclose(np_c, expected_c), "Addition failed"
    assert np.allclose(np_d, expected_d), "Multiplication failed"
    
    print("  ✓ Operations with NumPy successful\n")

def test_matrix_multiplication_with_numpy():
    """Test matrix multiplication with NumPy"""
    print("Testing matrix multiplication with NumPy...")
    
    # Create matrices
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    
    a = t4d.Matrixf.from_numpy(np_a)
    b = t4d.Matrixf.from_numpy(np_b)
    
    # Matrix multiply
    c = a.matmul(b)
    
    # Compare with NumPy result
    expected = np.matmul(np_a, np_b)
    result = c.numpy()
    
    print(f"  Expected:\n{expected}")
    print(f"  Got:\n{result}")
    
    assert np.allclose(result, expected), "Matrix multiplication mismatch"
    print("  ✓ Matrix multiplication successful\n")

def test_math_functions_with_numpy():
    """Test math functions with NumPy"""
    print("Testing math functions with NumPy...")
    
    np_vec = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    vec = t4d.Vectorf.from_numpy(np_vec)
    
    # Test exp
    exp_result = vec.exp().numpy()
    exp_expected = np.exp(np_vec)
    assert np.allclose(exp_result, exp_expected, rtol=1e-5), "exp() mismatch"
    print("  ✓ exp() correct")
    
    # Test log
    log_result = vec.log().numpy()
    log_expected = np.log(np_vec)
    assert np.allclose(log_result, log_expected, rtol=1e-5), "log() mismatch"
    print("  ✓ log() correct")
    
    # Test sqrt
    sqrt_result = vec.sqrt().numpy()
    sqrt_expected = np.sqrt(np_vec)
    assert np.allclose(sqrt_result, sqrt_expected, rtol=1e-5), "sqrt() mismatch"
    print("  ✓ sqrt() correct")
    
    print()

def main():
    """Run all tests"""
    print("=" * 60)
    print("NumPy Interoperability Tests")
    print("=" * 60 + "\n")
    
    try:
        test_vector_numpy_conversion()
        test_matrix_numpy_conversion()
        test_tensor3_numpy_conversion()
        test_constructor_with_numpy()
        test_operations_with_numpy()
        test_matrix_multiplication_with_numpy()
        test_math_functions_with_numpy()
        
        print("=" * 60)
        print("All NumPy interop tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
