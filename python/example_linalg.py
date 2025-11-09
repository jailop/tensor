#!/usr/bin/env python3
"""
Example demonstrating matrix operations and linear algebra.
"""

import tensor4d as t4d

def main():
    print("=== Linear Algebra Example ===\n")
    
    # Matrix multiplication
    print("1. Matrix multiplication:")
    A = t4d.Matrixf([[1, 2], [3, 4]])
    B = t4d.Matrixf([[5, 6], [7, 8]])
    
    C = A.matmul(B)
    
    print("A =")
    print(A)
    print("\nB =")
    print(B)
    print("\nA @ B =")
    print(C)
    print()
    
    # Transpose
    print("2. Matrix transpose:")
    print("A^T =")
    print(A.transpose())
    print()
    
    # Matrix inverse
    print("3. Matrix inverse:")
    M = t4d.Matrixf([[4, 7], [2, 6]])
    print("M =")
    print(M)
    
    print("\n(Note: Matrix inverse not yet available in Python bindings)")
    print()
    
    # Vector operations
    print("4. Vector operations:")
    v1 = t4d.Vectorf([1, 2, 3])
    v2 = t4d.Vectorf([4, 5, 6])
    
    print(f"v1 = {v1.tolist()}")
    print(f"v2 = {v2.tolist()}")
    print(f"v1 · v2 = {v1.dot(v2)}")
    print(f"||v1|| = {v1.norm():.4f}")
    print(f"||v2|| = {v2.norm():.4f}")
    print()

if __name__ == "__main__":
    main()
