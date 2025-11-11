#!/usr/bin/env python3
"""
Test neural network layers in Python
"""

import sys
sys.path.insert(0, '/home/jailop/shared/guides/transformers/python')

import tensor4d as t4d
import numpy as np

def test_linear_layer():
    print("Testing Linear layer...")
    
    # Create layer: 3 inputs -> 5 outputs
    layer = t4d.nn.Linearf(3, 5, True)
    
    # Create input (2 samples, 3 features)
    input_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    input_mat = t4d.Matrixf(input_data)
    
    # Forward pass
    output = layer.forward(input_mat)
    
    # Check shape
    assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"
    
    # Check that layer has parameters
    params = layer.parameters()
    assert len(params) == 2, f"Expected 2 parameters (weights and bias), got {len(params)}"
    
    print("  ✓ Linear layer passed")

def test_relu_layer():
    print("Testing ReLU layer...")
    
    layer = t4d.nn.ReLUf()
    
    # Create input with negative and positive values
    input_data = [[-1.0, 2.0], [-0.5, 3.0]]
    input_mat = t4d.Matrixf(input_data)
    
    # Forward pass
    output = layer.forward(input_mat)
    
    # Check that negative values are zeroed
    assert abs(output[0, 0]) < 1e-5, "Expected negative value to be zeroed"
    assert abs(output[0, 1] - 2.0) < 1e-5, "Expected positive value unchanged"
    
    print("  ✓ ReLU layer passed")

def test_sigmoid_layer():
    print("Testing Sigmoid layer...")
    
    layer = t4d.nn.Sigmoidf()
    
    # Create input
    input_data = [[0.0]]
    input_mat = t4d.Matrixf(input_data)
    
    # Forward pass
    output = layer.forward(input_mat)
    
    # sigmoid(0) = 0.5
    assert abs(output[0, 0] - 0.5) < 1e-5, f"Expected 0.5, got {output[0, 0]}"
    
    print("  ✓ Sigmoid layer passed")

def test_softmax_layer():
    print("Testing Softmax layer...")
    
    layer = t4d.nn.Softmaxf()
    
    # Create input
    input_data = [[1.0, 2.0, 3.0]]
    input_mat = t4d.Matrixf(input_data)
    
    # Forward pass
    output = layer.forward(input_mat)
    
    # Check that probabilities sum to 1
    prob_sum = output[0, 0] + output[0, 1] + output[0, 2]
    assert abs(prob_sum - 1.0) < 1e-5, f"Expected sum=1.0, got {prob_sum}"
    
    # Check that all values are between 0 and 1
    for i in range(3):
        val = output[0, i]
        assert 0.0 < val < 1.0, f"Expected probability in (0,1), got {val}"
    
    print("  ✓ Softmax layer passed")

def test_dropout_layer():
    print("Testing Dropout layer...")
    
    layer = t4d.nn.Dropoutf(0.5)
    
    # Test inference mode (no dropout)
    layer.train(False)
    assert not layer.is_training(), "Expected inference mode"
    
    # Create input
    input_data = [[1.0, 1.0], [1.0, 1.0]]
    input_mat = t4d.Matrixf(input_data)
    
    # Forward pass in inference mode
    output = layer.forward(input_mat)
    
    # In inference mode, values should be unchanged
    assert abs(output[0, 0] - 1.0) < 1e-5, "Expected no dropout in inference mode"
    
    print("  ✓ Dropout layer passed")

def test_batchnorm_layer():
    print("Testing BatchNorm layer...")
    
    layer = t4d.nn.BatchNorm1df(3, 1e-5, 0.1)
    
    # Create input
    input_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    input_mat = t4d.Matrixf(input_data)
    
    # Forward pass
    output = layer.forward(input_mat)
    
    # Check that output has same shape as input
    assert output.shape == input_mat.shape, "Expected same shape"
    
    # Check that layer has parameters (gamma and beta)
    params = layer.parameters()
    assert len(params) == 2, f"Expected 2 parameters, got {len(params)}"
    
    print("  ✓ BatchNorm layer passed")

def test_simple_network():
    print("Testing simple 2-layer network...")
    
    # Network: 4 -> 8 -> 2
    fc1 = t4d.nn.Linearf(4, 8, True)
    relu = t4d.nn.ReLUf()
    fc2 = t4d.nn.Linearf(8, 2, True)
    softmax = t4d.nn.Softmaxf()
    
    # Create input (batch of 3 samples)
    input_data = [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6]
    ]
    input_mat = t4d.Matrixf(input_data)
    
    # Forward pass
    h1 = fc1.forward(input_mat)
    h2 = relu.forward(h1)
    h3 = fc2.forward(h2)
    output = softmax.forward(h3)
    
    # Check output shape
    assert output.shape == (3, 2), f"Expected shape (3, 2), got {output.shape}"
    
    # Check that each row sums to 1 (softmax property)
    for i in range(3):
        row_sum = output[i, 0] + output[i, 1]
        assert abs(row_sum - 1.0) < 1e-5, f"Expected sum=1.0, got {row_sum}"
    
    print("  ✓ Simple network passed")

def test_training_mode():
    print("Testing training/inference mode switching...")
    
    layer = t4d.nn.Dropoutf(0.5)
    
    # Check default is training mode
    assert layer.is_training(), "Expected default training mode"
    
    # Switch to inference
    layer.train(False)
    assert not layer.is_training(), "Expected inference mode"
    
    # Switch back to training
    layer.train(True)
    assert layer.is_training(), "Expected training mode"
    
    print("  ✓ Training mode switching passed")

def main():
    print("=== Testing Neural Network Layers (Python) ===\n")
    
    test_linear_layer()
    test_relu_layer()
    test_sigmoid_layer()
    test_softmax_layer()
    test_dropout_layer()
    test_batchnorm_layer()
    test_simple_network()
    test_training_mode()
    
    print("\n=== All Python NN Layer Tests Passed! ===")

if __name__ == "__main__":
    main()
