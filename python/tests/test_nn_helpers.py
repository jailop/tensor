#!/usr/bin/env python3
"""
Test neural network helper functions in Python bindings
"""

import sys
sys.path.insert(0, '/home/jailop/shared/guides/transformers/python')

import tensor4d as t4d
import numpy as np

def test_label_to_onehot():
    print("Testing label_to_onehot...")
    
    # Create a batch of one-hot targets (batch_size=3, num_classes=5)
    batch_size = 3
    num_classes = 5
    targets = t4d.Matrixf([[0.0] * num_classes for _ in range(batch_size)])
    
    # Convert labels to one-hot
    labels = [2, 0, 4]  # Example labels
    for i, label in enumerate(labels):
        t4d.nn.label_to_onehot_f(label, targets, i, num_classes)
    
    # Verify
    assert abs(targets[0, 2] - 1.0) < 1e-5, "Expected targets[0,2]=1.0"
    assert abs(targets[1, 0] - 1.0) < 1e-5, "Expected targets[1,0]=1.0"
    assert abs(targets[2, 4] - 1.0) < 1e-5, "Expected targets[2,4]=1.0"
    
    # Verify other values are zero
    assert abs(targets[0, 0]) < 1e-5, "Expected targets[0,0]=0.0"
    assert abs(targets[1, 1]) < 1e-5, "Expected targets[1,1]=0.0"
    
    print("  ✓ label_to_onehot passed")

def test_cross_entropy_loss():
    print("Testing cross_entropy_loss...")
    
    # Create predictions (softmax output)
    predictions = t4d.Matrixf([
        [0.7, 0.2, 0.1],  # Confident about class 0
        [0.1, 0.8, 0.1],  # Confident about class 1
        [0.1, 0.1, 0.8]   # Confident about class 2
    ])
    
    # Create one-hot targets
    targets = t4d.Matrixf([
        [1.0, 0.0, 0.0],  # True class 0
        [0.0, 1.0, 0.0],  # True class 1
        [0.0, 0.0, 1.0]   # True class 2
    ])
    
    # Compute loss
    loss = t4d.nn.cross_entropy_loss_f(predictions, targets)
    
    # Loss should be low since predictions match targets
    assert loss > 0.0, "Loss should be positive"
    assert loss < 1.0, f"Loss should be low for correct predictions, got {loss}"
    
    print(f"  Cross-entropy loss: {loss:.4f}")
    print("  ✓ cross_entropy_loss passed")

def test_cross_entropy_loss_mismatch():
    print("Testing cross_entropy_loss with mismatch...")
    
    # Create predictions (confident about wrong class)
    predictions = t4d.Matrixf([
        [0.1, 0.8, 0.1],  # Predicts class 1
        [0.8, 0.1, 0.1],  # Predicts class 0
    ])
    
    # Create one-hot targets (different from predictions)
    targets = t4d.Matrixf([
        [1.0, 0.0, 0.0],  # True class 0
        [0.0, 1.0, 0.0],  # True class 1
    ])
    
    # Compute loss
    loss = t4d.nn.cross_entropy_loss_f(predictions, targets)
    
    # Loss should be high since predictions don't match targets
    assert loss > 1.0, f"Loss should be high for wrong predictions, got {loss}"
    
    print(f"  Cross-entropy loss (mismatch): {loss:.4f}")
    print("  ✓ cross_entropy_loss with mismatch passed")

def test_compute_accuracy():
    print("Testing compute_accuracy...")
    
    # Create predictions (softmax output)
    # Class with highest probability is the prediction
    predictions = t4d.Matrixf([
        [0.7, 0.2, 0.1],  # Predicts class 0
        [0.1, 0.8, 0.1],  # Predicts class 1
        [0.1, 0.1, 0.8],  # Predicts class 2
        [0.3, 0.6, 0.1],  # Predicts class 1
    ])
    
    # True labels
    labels = [0, 1, 2, 0]  # First 3 correct, last one wrong
    
    # Compute accuracy
    accuracy = t4d.nn.compute_accuracy_f(predictions, labels)
    
    # Expected: 3 out of 4 correct = 0.75
    expected_accuracy = 0.75
    assert abs(accuracy - expected_accuracy) < 1e-5, \
        f"Expected accuracy {expected_accuracy}, got {accuracy}"
    
    print(f"  Accuracy: {accuracy:.2%}")
    print("  ✓ compute_accuracy passed")

def test_compute_accuracy_offset():
    print("Testing compute_accuracy with offset...")
    
    # Create predictions
    predictions = t4d.Matrixf([
        [0.8, 0.2],  # Predicts class 0
        [0.3, 0.7],  # Predicts class 1
    ])
    
    # Labels array with offset
    labels = [99, 99, 0, 1]  # Start from index 2
    
    # Compute accuracy with offset=2
    accuracy = t4d.nn.compute_accuracy_f(predictions, labels, offset=2)
    
    # Both predictions correct
    assert abs(accuracy - 1.0) < 1e-5, f"Expected accuracy 1.0, got {accuracy}"
    
    print("  ✓ compute_accuracy with offset passed")

def test_update_linear_layer():
    print("Testing update_linear_layer...")
    
    # Create a simple linear layer
    layer = t4d.nn.Linearf(3, 2, use_bias=True)
    
    # Get initial weights
    params = layer.parameters()
    weights_before = params[0]
    initial_value = weights_before[0, 0]
    
    # Create fake input and do forward/backward to generate gradients
    input_mat = t4d.Matrixf([[1.0, 2.0, 3.0]])
    output = layer.forward(input_mat)
    
    # Create fake gradient
    grad_output = t4d.Matrixf([[1.0, 1.0]])
    layer.backward(grad_output)
    
    # Update with learning rate
    lr = 0.01
    t4d.nn.update_linear_layer_f(layer, lr)
    
    # Get updated weights
    params_after = layer.parameters()
    weights_after = params_after[0]
    final_value = weights_after[0, 0]
    
    # Weight should have changed after update
    assert abs(final_value - initial_value) > 1e-6, \
        f"Expected weight to change, but it didn't (before={initial_value}, after={final_value})"
    
    print("  ✓ update_linear_layer passed")

def test_fill_with_nn_helpers():
    print("Testing fill() in training loop context...")
    
    # Simulate preparing batch targets
    batch_size = 4
    num_classes = 3
    
    # Allocate targets once
    batch_targets = t4d.Matrixf([[0.0] * num_classes for _ in range(batch_size)])
    
    # Simulate multiple batches
    for batch_idx in range(2):
        # Reset to zeros (reusing tensor)
        batch_targets.fill(0.0)
        
        # Fill with labels
        labels = [0, 1, 2, 1]
        for i, label in enumerate(labels):
            t4d.nn.label_to_onehot_f(label, batch_targets, i, num_classes)
        
        # Verify
        assert abs(batch_targets[0, 0] - 1.0) < 1e-5
        assert abs(batch_targets[1, 1] - 1.0) < 1e-5
        assert abs(batch_targets[2, 2] - 1.0) < 1e-5
        assert abs(batch_targets[3, 1] - 1.0) < 1e-5
    
    print("  ✓ fill() with NN helpers passed")

def test_simple_training_step():
    print("Testing complete training step with helpers...")
    
    # Create network: 4 -> 3 (3 classes)
    fc = t4d.nn.Linearf(4, 3, True)
    softmax = t4d.nn.Softmaxf()
    
    # Batch of 2 samples
    batch_input = t4d.Matrixf([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])
    
    # Labels
    labels = [0, 2]
    
    # Prepare targets
    batch_targets = t4d.Matrixf([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i, label in enumerate(labels):
        t4d.nn.label_to_onehot_f(label, batch_targets, i, 3)
    
    # Forward pass
    logits = fc.forward(batch_input)
    predictions = softmax.forward(logits)
    
    # Compute loss
    loss = t4d.nn.cross_entropy_loss_f(predictions, batch_targets)
    assert loss > 0.0, "Loss should be positive"
    
    # Compute accuracy
    accuracy = t4d.nn.compute_accuracy_f(predictions, labels)
    assert 0.0 <= accuracy <= 1.0, "Accuracy should be in [0, 1]"
    
    # Backward pass (compute gradients)
    # grad = predictions - targets
    grad_diff = predictions - batch_targets
    grad_output = grad_diff / 2.0  # divide by batch_size
    
    softmax.backward(grad_output)
    fc.backward(grad_output)
    
    # Update weights
    t4d.nn.update_linear_layer_f(fc, 0.01)
    
    print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
    print("  ✓ Complete training step passed")

def main():
    print("=== Testing Neural Network Helper Functions (Python) ===\n")
    
    test_label_to_onehot()
    test_cross_entropy_loss()
    test_cross_entropy_loss_mismatch()
    test_compute_accuracy()
    test_compute_accuracy_offset()
    test_update_linear_layer()
    test_fill_with_nn_helpers()
    test_simple_training_step()
    
    print("\n=== All NN Helper Tests Passed! ===")

if __name__ == "__main__":
    main()
