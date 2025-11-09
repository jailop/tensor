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
    
    # Using @ operator
    C2 = A @ B
    print("\nUsing @ operator: A @ B =")
    print(C2)
    print()
    
    # Transpose
    print("2. Matrix transpose:")
    print("A^T =")
    print(A.transpose())
    
    # Using linalg module
    At = t4d.linalg.transpose(A)
    print("\nUsing linalg.transpose(A):")
    print(At)
    print()
    
    # Matrix norms
    print("3. Matrix norms:")
    M = t4d.Matrixf([[1, 2], [3, 4]])
    print("M =")
    print(M)
    print(f"Frobenius norm: {t4d.linalg.frobenius_norm(M):.4f}")
    print(f"L1 norm: {t4d.linalg.norm_l1(M):.4f}")
    print(f"L-infinity norm: {t4d.linalg.norm_inf(M):.4f}")
    print(f"Trace: {t4d.linalg.trace(M):.4f}")
    print(f"Rank: {t4d.linalg.matrix_rank(M)}")
    print(f"Condition number: {M.condition_number():.4f}")
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
    
    v1_normalized = t4d.linalg.normalize(v1)
    print(f"normalize(v1) = {v1_normalized.tolist()}")
    print(f"||normalize(v1)|| = {v1_normalized.norm():.6f}")
    print()
    
    # Outer product
    print("5. Outer product:")
    a = t4d.Vectorf([1, 2])
    b = t4d.Vectorf([3, 4])
    outer = t4d.linalg.outer(a, b)
    print(f"a = {a.tolist()}")
    print(f"b = {b.tolist()}")
    print("outer(a, b) =")
    print(outer)
    print()
    
    # Matrix-vector multiplication
    print("6. Matrix-vector multiplication:")
    A = t4d.Matrixf([[1, 2], [3, 4], [5, 6]])
    v = t4d.Vectorf([7, 8])
    result = t4d.linalg.matvec(A, v)
    print("A =")
    print(A)
    print(f"v = {v.tolist()}")
    print(f"A @ v = {result.tolist()}")
    print()
    
    # Diagonal operations
    print("7. Diagonal operations:")
    diag_vec = t4d.Vectorf([1, 2, 3])
    diag_mat = t4d.linalg.diag(diag_vec)
    print(f"diag({diag_vec.tolist()}) =")
    print(diag_mat)
    
    extracted = t4d.linalg.diag(diag_mat)
    print(f"diag(matrix) = {extracted.tolist()}")
    print()
    
    # Identity matrix
    print("8. Identity matrix:")
    I = t4d.linalg.eye(4, False)  # CPU only
    print("eye(4) =")
    print(I)
    print()
    
    # Least squares
    print("9. Least squares solution:")
    A = t4d.Matrixf([[1, 1], [1, 2], [1, 3]])
    b = t4d.Vectorf([1, 2, 3])
    x = t4d.linalg.lstsq(A, b)
    print("A =")
    print(A)
    print(f"b = {b.tolist()}")
    print(f"Least squares solution x = {x.tolist()}")
    print()

if __name__ == "__main__":
    main()
