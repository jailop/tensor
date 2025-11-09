"""
Tensor4D Python Bindings

A high-performance tensor library with automatic differentiation,
GPU acceleration, and NumPy interoperability.
"""

try:
    from .tensor4d import *
    __all__ = ['Vectorf', 'Vectord', 'Matrixf', 'Matrixd',
               'Tensor3f', 'Tensor3d', 'Tensor4f', 'Tensor4d',
               'TensorError', 'TensorIOFormat',
               'mse_loss', 'cross_entropy_loss', 'binary_cross_entropy_loss',
               'SGD', 'Adam', 'RMSprop', 'ExponentialLR', 'Optimizer',
               'save_tensor', 'load_tensor_f2']
except ImportError:
    # Module not yet built
    pass

__version__ = '1.0.0'
