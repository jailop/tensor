#!/usr/bin/env python3
"""
Basic example demonstrating tensor creation and operations.
"""

import tensor4d as t4d

def main():
    print("=== Tensor4D Python Bindings - Basic Example ===\n")
    
    # Create a float matrix
    print("1. Creating a 3x3 float matrix:")
    m = t4d.Matrixf([3, 3])
    m.fill(2.0)
    print(m)
    print()
    
    # Create from Python list
    print("2. Creating from Python list:")
    data = [[1, 2, 3], [4, 5, 6]]
    m2 = t4d.Matrixf(data)
    print(m2)
    print()
    
    # Arithmetic operations
    print("3. Arithmetic operations:")
    m3 = m2 + m2
    print("m2 + m2 =")
    print(m3)
    print()
    
    m4 = m2 * 2.0
    print("m2 * 2.0 =")
    print(m4)
    print()
    
    # Mathematical functions
    print("4. Mathematical functions:")
    m_exp = m2.exp()
    print("exp(m2) =")
    print(m_exp)
    print()
    
    # Statistical operations
    print("5. Statistical operations:")
    print(f"Sum: {m2.sum()}")
    print(f"Mean: {m2.mean()}")
    print(f"Min: {m2.min()}")
    print(f"Max: {m2.max()}")
    print()
    
    # Convert to Python list
    print("6. Convert to Python list:")
    list_result = m3.tolist()
    print(f"Type: {type(list_result)}")
    print(f"Data: {list_result}")
    print()
    
    # Vector operations
    print("7. Vector operations:")
    v1 = t4d.Vectorf([3])
    v1.fill(1.0)
    v2 = t4d.Vectorf([3])
    v2.fill(2.0)
    
    dot_product = v1.dot(v2)
    print(f"v1 · v2 = {dot_product}")
    print(f"||v2|| = {v2.norm()}")
    print()

if __name__ == "__main__":
    main()
