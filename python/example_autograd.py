#!/usr/bin/env python3
"""
Example demonstrating autograd and neural network training.
"""

import tensor4d as t4d

def main():
    print("=== Autograd Example ===\n")
    
    # Enable gradient tracking
    print("1. Simple gradient computation:")
    x = t4d.Matrixf([[2.0]])
    x.set_requires_grad(True)
    
    # Forward pass: y = x^2
    y = x * x
    
    # Compute gradients (on scalar tensor)
    y.backward()
    
    print("x =")
    print(x)
    print("\ny = x²")
    print(y)
    print("\n∂y/∂x (where y = x²) =")
    if x.grad():
        print(x.grad())
    print()
    
    # More complex example
    print("2. Chained operations with autograd:")
    
    # Create a scalar input
    a = t4d.Matrixf([[3.0]])
    a.set_requires_grad(True)
    
    # Chain of operations
    b = a * a       # b = a²
    c = b + a       # c = a² + a
    d = c * 2.0     # d = 2(a² + a)
    
    print("a =", a.tolist()[0][0])
    print("d = 2(a² + a) =", d.tolist()[0][0])
    
    # Backward pass
    d.backward()
    
    print("\nGradient ∂d/∂a =", a.grad().tolist()[0][0])
    print("Expected: 2(2a + 1) = 2(6 + 1) = 14")
    print()
    
    # Detach example
    print("3. Gradient detachment:")
    x = t4d.Matrixf([[5.0]])
    x.set_requires_grad(True)
    
    y = x * x
    z_detached = y.detach()  # Stop gradient tracking
    
    print("After detach, z requires_grad:", z_detached.requires_grad)
    print()

if __name__ == "__main__":
    main()
