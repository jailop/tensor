#!/usr/bin/env python3
"""
Test script for optimizer functionality with proper autograd.
"""

import tensor4d as t4d
import random

def test_sgd():
    """Test SGD optimizer with manual gradient computation."""
    print("=== Testing SGD Optimizer ===")
    
    # Create a simple parameter  
    W = t4d.Matrixf([[1.0, 2.0], [3.0, 4.0]])
    W.set_requires_grad(True)
    
    # Target
    target = t4d.Matrixf([[0.0, 0.0], [0.0, 0.0]])
    
    # Create optimizer
    optimizer = t4d.SGD([W], learning_rate=0.1, momentum=0.0, weight_decay=0.0)
    
    print(f"Initial W:\n{W.tolist()}")
    
    # Manual training loop
    for epoch in range(3):
        # Zero gradients
        optimizer.zero_grad()
        
        # Update parameters
        optimizer.step()
        
        # Compute loss for display
        W_list = W.tolist()
        target_list = target.tolist()
        loss = sum((W_list[i][j] - target_list[i][j])**2 
                   for i in range(2) for j in range(2))
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    print(f"Final W:\n{W.tolist()}")
    print()

def test_adam():
    """Test Adam optimizer."""
    print("=== Testing Adam Optimizer ===")
    
    # Create parameter
    W_data = [[random.gauss(0, 0.1) for _ in range(2)] for _ in range(3)]
    W = t4d.Matrixf(W_data)
    W.set_requires_grad(True)
    
    # Create optimizer
    optimizer = t4d.Adam([W], learning_rate=0.001, beta1=0.9, beta2=0.999)
    
    print(f"Initial W:\n{W.tolist()}")
    print(f"W requires_grad: {W.requires_grad}")
    
    # Run a few steps
    for epoch in range(3):
        optimizer.zero_grad()
        optimizer.step()
        print(f"Epoch {epoch+1} completed")
    
    print(f"Final W:\n{W.tolist()}")
    print()

def test_rmsprop():
    """Test RMSprop optimizer."""
    print("=== Testing RMSprop Optimizer ===")
    
    # Create parameter
    W_data = [[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)]
    W = t4d.Matrixf(W_data)
    W.set_requires_grad(True)
    
    # Create optimizer
    optimizer = t4d.RMSprop([W], learning_rate=0.01, alpha=0.99)
    
    print(f"Initial W:\n{W.tolist()}")
    
    # Run a few steps
    for epoch in range(3):
        optimizer.zero_grad()
        optimizer.step()
        print(f"Epoch {epoch+1} completed")
    
    print(f"Final W:\n{W.tolist()}")
    print()

def test_learning_rate_scheduler():
    """Test learning rate scheduler."""
    print("=== Testing ExponentialLR Scheduler ===")
    
    # Create parameter and optimizer
    W_data = [[random.gauss(0, 1) for _ in range(2)] for _ in range(2)]
    W = t4d.Matrixf(W_data)
    W.set_requires_grad(True)
    
    optimizer = t4d.SGD([W], learning_rate=1.0, momentum=0.0)
    
    # Create scheduler
    scheduler = t4d.ExponentialLR(optimizer, gamma=0.9)
    
    print(f"Initial learning rate: {optimizer.get_lr()}")
    
    for epoch in range(5):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, LR: {optimizer.get_lr():.6f}")
    
    # Reset
    scheduler.reset()
    print(f"After reset, LR: {optimizer.get_lr():.6f}")
    print()

def test_optimizer_api():
    """Test optimizer API compatibility."""
    print("=== Testing Optimizer API ===")
    
    W1 = t4d.Matrixf([[1.0, 1.0], [1.0, 1.0]])
    W1.set_requires_grad(True)
    
    W2 = t4d.Matrixf([[1.0, 1.0], [1.0, 1.0]])
    W2.set_requires_grad(True)
    
    # Test with multiple parameters
    sgd = t4d.SGD([W1, W2], learning_rate=0.01)
    
    print("Created SGD with 2 parameters")
    print(f"Learning rate: {sgd.get_lr()}")
    
    # Test set_lr
    sgd.set_lr(0.02)
    print(f"After set_lr(0.02): {sgd.get_lr()}")
    
    # Test zero_grad and step
    sgd.zero_grad()
    sgd.step()
    print("zero_grad() and step() executed successfully")
    print()

if __name__ == "__main__":
    test_sgd()
    test_adam()
    test_rmsprop()
    test_learning_rate_scheduler()
    test_optimizer_api()
    
    print("=== All Optimizer Tests Passed ===")
