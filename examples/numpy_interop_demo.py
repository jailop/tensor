#!/usr/bin/env python3
"""
Example demonstrating NumPy interoperability with tensor library.
This script loads a .npy file created by the C++ tensor library.
"""

import numpy as np
import sys

def load_and_verify():
    """Load tensor from .npy file and display it."""
    try:
        # Load the tensor saved by C++
        data = np.load('tensor.npy')
        
        print("=== Python NumPy Interoperability Demo ===\n")
        print(f"Loaded tensor shape: {data.shape}")
        print(f"Loaded tensor dtype: {data.dtype}")
        print(f"\nTensor contents:")
        print(data)
        
        # Perform some NumPy operations
        print(f"\nNumPy operations:")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Std:  {np.std(data):.2f}")
        print(f"  Min:  {np.min(data):.2f}")
        print(f"  Max:  {np.max(data):.2f}")
        
        # Save modified version back
        modified = data * 2.0
        np.save('tensor_modified.npy', modified)
        print(f"\nSaved modified tensor (scaled by 2) to 'tensor_modified.npy'")
        print("This file can be loaded by the C++ tensor library.")
        
        return True
        
    except FileNotFoundError:
        print("Error: tensor.npy not found")
        print("Please run './build/tensor_io_demo' first to create the file")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = load_and_verify()
    sys.exit(0 if success else 1)
