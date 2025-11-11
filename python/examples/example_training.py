#!/usr/bin/env python3
"""
Example demonstrating loss functions and optimizers.
"""

import tensor4d as t4d
import random

def main():
    print("=== Loss Functions and Optimizers Example ===\n")
    
    # Create sample data
    batch_size = 4
    num_classes = 3
    
    # Predictions (logits) - random values
    pred_data = [[random.gauss(0, 1) for _ in range(num_classes)] for _ in range(batch_size)]
    predictions = t4d.Matrixf(pred_data)
    
    # Targets (one-hot encoded)
    targets_data = [[0.0] * num_classes for _ in range(batch_size)]
    targets_data[0][0] = 1.0
    targets_data[1][2] = 1.0
    targets_data[2][1] = 1.0
    targets_data[3][0] = 1.0
    targets = t4d.Matrixf(targets_data)
    
    print("1. Loss Functions:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print()
    
    # MSE Loss
    try:
        mse = t4d.mse_loss(predictions, targets)
        print(f"MSE Loss: {mse}")
    except Exception as e:
        print(f"MSE Loss error: {e}")
    
    # Cross Entropy Loss
    try:
        ce = t4d.cross_entropy_loss(predictions, targets)
        print(f"Cross Entropy Loss: {ce}")
    except Exception as e:
        print(f"Cross Entropy Loss error: {e}")
    
    # Binary Cross Entropy
    binary_pred = t4d.Matrixf([[random.random()] for _ in range(batch_size)])
    binary_target = t4d.Matrixf([[float(random.randint(0, 1))] for _ in range(batch_size)])
    
    try:
        bce = t4d.binary_cross_entropy_loss(binary_pred, binary_target)
        print(f"Binary Cross Entropy Loss: {bce}")
    except Exception as e:
        print(f"BCE Loss error: {e}")
    print()
    
    # Optimizers
    print("2. Optimizers:")
    
    # Create a simple parameter
    W_data = [[random.gauss(0, 0.1) for _ in range(5)] for _ in range(10)]
    W = t4d.Matrixf(W_data)
    W.set_requires_grad(True)
    
    # Target for simple regression
    target = t4d.Matrixf([[1.0] * 5 for _ in range(10)])
    
    # SGD optimizer
    print("\nSGD Optimizer:")
    sgd = t4d.SGD([W], learning_rate=0.01, momentum=0.0)  # No momentum for simpler example
    
    for i in range(3):
        sgd.zero_grad()
        
        # Dummy forward pass - simple prediction
        X_data = [[random.gauss(0, 1) for _ in range(10)] for _ in range(10)]
        X = t4d.Matrixf(X_data)
        output = X.matmul(W)
        
        # Compute loss manually as element-wise squared difference, then mean
        diff = output - target
        squared = diff * diff
        loss_val = squared.mean()
        
        print(f"  Step {i+1}, Loss: {loss_val:.4f}, W norm: {(W * W).sum():.4f}")
    
    print("\nAdam Optimizer:")
    W2_data = [[random.gauss(0, 0.1) for _ in range(5)] for _ in range(10)]
    W2 = t4d.Matrixf(W2_data)
    W2.set_requires_grad(True)
    
    adam = t4d.Adam([W2], learning_rate=0.001, beta1=0.9, beta2=0.999)
    
    for i in range(3):
        adam.zero_grad()
        
        # Dummy forward pass
        X_data = [[random.gauss(0, 1) for _ in range(10)] for _ in range(10)]
        X = t4d.Matrixf(X_data)
        output = X.matmul(W2)
        
        # Compute loss
        diff = output - target
        squared = diff * diff
        loss_val = squared.mean()
        
        print(f"  Step {i+1}, Loss: {loss_val:.4f}, W norm: {(W2 * W2).sum():.4f}")
    
    print("\nNote: Actual gradient computation requires forward pass tracking.")
    print("      This example demonstrates the optimizer API.")
    print()

if __name__ == "__main__":
    main()
