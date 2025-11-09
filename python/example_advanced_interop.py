#!/usr/bin/env python3
"""
Advanced interoperability example.
This demonstrates that tensor4d can work with native Python data structures
without requiring NumPy. However, if you have NumPy installed, you can still
use it for convenience.
"""

import tensor4d as t4d

def main():
    print("=== Tensor4D Advanced Interoperability Example ===\n")
    
    print("1. Working with native Python lists:")
    # Create from nested list
    data = [[1, 2, 3], [4, 5, 6]]
    m = t4d.Matrixf(data)
    print(f"Created from list: {m}")
    
    # Convert back to list
    result = m.tolist()
    print(f"Converted to list: {result}")
    print(f"Type: {type(result)}")
    print()
    
    print("2. Working with tuples:")
    # Tuples work too
    data_tuple = ((1.0, 2.0), (3.0, 4.0))
    m2 = t4d.Matrixf(data_tuple)
    print(f"Created from tuple: {m2}")
    print()
    
    print("3. Vector operations:")
    v1 = t4d.Vectorf([1.0, 2.0, 3.0])
    v2 = t4d.Vectorf([4.0, 5.0, 6.0])
    print(f"v1 = {v1.tolist()}")
    print(f"v2 = {v2.tolist()}")
    print(f"v1 + v2 = {(v1 + v2).tolist()}")
    print(f"v1 · v2 = {v1.dot(v2)}")
    print()
    
    print("4. Matrix operations:")
    A = t4d.Matrixf([[1, 2], [3, 4]])
    B = t4d.Matrixf([[5, 6], [7, 8]])
    C = A.matmul(B)
    print(f"A @ B = {C.tolist()}")
    print()
    
    # Optional: NumPy integration (if available)
    try:
        import numpy as np
        print("5. Optional NumPy integration (NumPy detected):")
        
        # You can still create from NumPy if you want
        np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        # Convert to list first, then to tensor
        m_from_np = t4d.Matrixf(np_arr.tolist())
        print(f"Created from NumPy array: {m_from_np}")
        
        # Convert tensor to list, then to NumPy if needed
        tensor_list = m_from_np.tolist()
        np_result = np.array(tensor_list, dtype=np.float32)
        print(f"Converted back to NumPy: shape={np_result.shape}, dtype={np_result.dtype}")
        print()
    except ImportError:
        print("5. NumPy not installed (this is fine - not required!)")
        print()
    
    print("Note: tensor4d does NOT require NumPy!")
    print("All operations work with native Python lists and tuples.")

if __name__ == "__main__":
    main()
