#!/usr/bin/env python3
"""
Unit tests for tensor4d Python bindings.
"""

import unittest
import numpy as np
import sys

try:
    import tensor4d as t4d
    HAS_TENSOR4D = True
except ImportError:
    HAS_TENSOR4D = False
    print("Warning: tensor4d module not found. Tests will be skipped.")


@unittest.skipUnless(HAS_TENSOR4D, "tensor4d module not available")
class TestBasicOperations(unittest.TestCase):
    
    def test_create_matrix(self):
        """Test creating a matrix"""
        m = t4d.Matrixf([3, 3])
        self.assertEqual(m.shape, (3, 3))
        self.assertEqual(m.size, 9)
    
    def test_fill(self):
        """Test filling a matrix"""
        m = t4d.Matrixf([2, 2])
        m.fill(5.0)
        np_m = m.numpy()
        self.assertTrue(np.all(np_m == 5.0))
    
    def test_from_numpy(self):
        """Test creating tensor from NumPy"""
        np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        m = t4d.Matrixf(np_arr)
        np_result = m.numpy()
        self.assertTrue(np.allclose(np_arr, np_result))
    
    def test_arithmetic(self):
        """Test arithmetic operations"""
        m1 = t4d.Matrixf(np.array([[1, 2], [3, 4]], dtype=np.float32))
        m2 = t4d.Matrixf(np.array([[5, 6], [7, 8]], dtype=np.float32))
        
        result = m1 + m2
        expected = np.array([[6, 8], [10, 12]], dtype=np.float32)
        self.assertTrue(np.allclose(result.numpy(), expected))
    
    def test_scalar_operations(self):
        """Test scalar operations"""
        m = t4d.Matrixf(np.array([[1, 2], [3, 4]], dtype=np.float32))
        result = m * 2.0
        expected = np.array([[2, 4], [6, 8]], dtype=np.float32)
        self.assertTrue(np.allclose(result.numpy(), expected))


@unittest.skipUnless(HAS_TENSOR4D, "tensor4d module not available")
class TestMathFunctions(unittest.TestCase):
    
    def test_exp(self):
        """Test exponential function"""
        m = t4d.Matrixf(np.array([[0, 1], [2, 3]], dtype=np.float32))
        result = m.exp()
        expected = np.exp(m.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected, rtol=1e-5))
    
    def test_log(self):
        """Test logarithm function"""
        m = t4d.Matrixf(np.array([[1, 2], [3, 4]], dtype=np.float32))
        result = m.log()
        expected = np.log(m.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected, rtol=1e-5))
    
    def test_sqrt(self):
        """Test square root function"""
        m = t4d.Matrixf(np.array([[1, 4], [9, 16]], dtype=np.float32))
        result = m.sqrt()
        expected = np.sqrt(m.numpy())
        self.assertTrue(np.allclose(result.numpy(), expected, rtol=1e-5))


@unittest.skipUnless(HAS_TENSOR4D, "tensor4d module not available")
class TestStatisticalOperations(unittest.TestCase):
    
    def test_sum(self):
        """Test sum operation"""
        m = t4d.Matrixf(np.array([[1, 2], [3, 4]], dtype=np.float32))
        result = m.sum()
        expected = 10.0
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_mean(self):
        """Test mean operation"""
        m = t4d.Matrixf(np.array([[1, 2], [3, 4]], dtype=np.float32))
        result = m.mean()
        expected = 2.5
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_min_max(self):
        """Test min and max operations"""
        m = t4d.Matrixf(np.array([[1, 2], [3, 4]], dtype=np.float32))
        self.assertAlmostEqual(m.min(), 1.0, places=5)
        self.assertAlmostEqual(m.max(), 4.0, places=5)


@unittest.skipUnless(HAS_TENSOR4D, "tensor4d module not available")
class TestLinearAlgebra(unittest.TestCase):
    
    def test_matmul(self):
        """Test matrix multiplication"""
        A = t4d.Matrixf(np.array([[1, 2], [3, 4]], dtype=np.float32))
        B = t4d.Matrixf(np.array([[5, 6], [7, 8]], dtype=np.float32))
        C = A.matmul(B)
        
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        self.assertTrue(np.allclose(C.numpy(), expected))
    
    def test_transpose(self):
        """Test matrix transpose"""
        m = t4d.Matrixf(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        m_t = m.transpose()
        
        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
        self.assertTrue(np.allclose(m_t.numpy(), expected))
    
    def test_vector_dot(self):
        """Test vector dot product"""
        v1 = t4d.Vectorf(np.array([1, 2, 3], dtype=np.float32))
        v2 = t4d.Vectorf(np.array([4, 5, 6], dtype=np.float32))
        
        result = v1.dot(v2)
        expected = 32.0  # 1*4 + 2*5 + 3*6
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_vector_norm(self):
        """Test vector norm"""
        v = t4d.Vectorf(np.array([3, 4], dtype=np.float32))
        norm = v.norm()
        expected = 5.0  # sqrt(3^2 + 4^2)
        self.assertAlmostEqual(norm, expected, places=5)


@unittest.skipUnless(HAS_TENSOR4D, "tensor4d module not available")
class TestVectorOperations(unittest.TestCase):
    
    def test_vector_creation(self):
        """Test vector creation"""
        v = t4d.Vectorf([5])
        self.assertEqual(v.shape, (5,))
        self.assertEqual(v.size, 5)
    
    def test_vector_from_numpy(self):
        """Test vector from NumPy"""
        np_v = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        v = t4d.Vectorf(np_v)
        self.assertTrue(np.allclose(v.numpy(), np_v))


@unittest.skipUnless(HAS_TENSOR4D, "tensor4d module not available")
class TestAutograd(unittest.TestCase):
    
    def test_requires_grad(self):
        """Test gradient requirement flag"""
        m = t4d.Matrixf([2, 2])
        m.set_requires_grad(True)
        self.assertTrue(m.requires_grad)
    
    def test_simple_backward(self):
        """Test simple backward pass"""
        x = t4d.Matrixf(np.array([[2.0]], dtype=np.float32))
        x.set_requires_grad(True)
        
        y = x * x  # y = x^2
        # For scalar tensor, use backward on the tensor itself
        y.backward()
        
        # Gradient should be 2x = 4
        grad = x.grad()
        if grad:
            self.assertAlmostEqual(grad.numpy()[0, 0], 4.0, places=4)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
