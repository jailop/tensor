#!/usr/bin/env python3
"""
Real-World Example: Image Processing with NumPy Integration

This example demonstrates using Tensor4D with NumPy for a practical
image processing and simple neural network task.
"""

import numpy as np
import tensor4d as t4d

print("=" * 70)
print("Real-World Example: Image Processing Pipeline")
print("=" * 70 + "\n")

# ============================================================================
# 1. Generate Synthetic Image Data
# ============================================================================
print("1. Generating Synthetic Image Data")
print("-" * 70)

# Simulate loading a batch of small images (e.g., 28x28 grayscale)
# In practice, you might use: images = np.load('images.npy')
np.random.seed(42)
batch_size = 10
height, width = 28, 28
channels = 1

# Generate random "images" (in practice, load from disk)
images_np = np.random.randn(batch_size, height, width).astype(np.float32)
print(f"Generated {batch_size} images of size {height}x{width}")
print(f"NumPy shape: {images_np.shape}")
print(f"Data range: [{images_np.min():.2f}, {images_np.max():.2f}]\n")

# ============================================================================
# 2. Preprocessing with NumPy
# ============================================================================
print("2. Preprocessing with NumPy")
print("-" * 70)

# Common preprocessing steps
# Normalize to [0, 1]
images_normalized = (images_np - images_np.min()) / (images_np.max() - images_np.min())
print(f"Normalized range: [{images_normalized.min():.2f}, {images_normalized.max():.2f}]")

# Standardize (zero mean, unit variance)
images_standardized = (images_np - images_np.mean()) / (images_np.std() + 1e-8)
print(f"Standardized mean: {images_standardized.mean():.6f}, std: {images_standardized.std():.4f}")

# Add noise (data augmentation)
noise = np.random.randn(*images_np.shape).astype(np.float32) * 0.1
images_augmented = images_normalized + noise
images_augmented = np.clip(images_augmented, 0, 1)
print(f"Augmented {batch_size} images with noise\n")

# ============================================================================
# 3. Convert to Tensor4D for Model Operations
# ============================================================================
print("3. Converting to Tensor4D")
print("-" * 70)

# Convert preprocessed data to Tensor4D
# For a single batch, we'll flatten to 2D: (batch_size, height*width)
images_flat = images_augmented.reshape(batch_size, -1)
X = t4d.Matrixf.from_numpy(images_flat)

print(f"Tensor4D shape: {X.shape}")
print(f"Total elements: {X.size}\n")

# ============================================================================
# 4. Simple Neural Network Forward Pass
# ============================================================================
print("4. Neural Network Forward Pass")
print("-" * 70)

# Define network architecture: 784 -> 128 -> 64 -> 10
input_size = height * width  # 784
hidden1_size = 128
hidden2_size = 64
output_size = 10

# Initialize weights with NumPy (e.g., from a trained model)
# In practice: W1 = np.load('weights_layer1.npy')
W1_np = np.random.randn(input_size, hidden1_size).astype(np.float32) * 0.01
b1_np = np.zeros(hidden1_size, dtype=np.float32)

W2_np = np.random.randn(hidden1_size, hidden2_size).astype(np.float32) * 0.01
b2_np = np.zeros(hidden2_size, dtype=np.float32)

W3_np = np.random.randn(hidden2_size, output_size).astype(np.float32) * 0.01
b3_np = np.zeros(output_size, dtype=np.float32)

print("Initialized network weights")
print(f"  Layer 1: {input_size} -> {hidden1_size}")
print(f"  Layer 2: {hidden1_size} -> {hidden2_size}")
print(f"  Layer 3: {hidden2_size} -> {output_size}\n")

# Convert weights to Tensor4D
W1 = t4d.Matrixf.from_numpy(W1_np)
W2 = t4d.Matrixf.from_numpy(W2_np)
W3 = t4d.Matrixf.from_numpy(W3_np)

# Note: For simplicity, we'll skip biases in this example
# In a real implementation, you would add them after each matmul

# Forward pass: Layer 1
z1 = X.matmul(W1)  # (batch_size, 128)
a1 = z1.relu()      # ReLU activation
print(f"Layer 1 output shape: {a1.shape}")

# Forward pass: Layer 2
z2 = a1.matmul(W2)  # (batch_size, 64)
a2 = z2.relu()      # ReLU activation
print(f"Layer 2 output shape: {a2.shape}")

# Forward pass: Layer 3 (output)
z3 = a2.matmul(W3)  # (batch_size, 10)
output = z3.sigmoid()  # Sigmoid for probabilities
print(f"Output shape: {output.shape}\n")

# ============================================================================
# 5. Convert Back to NumPy for Analysis
# ============================================================================
print("5. Analysis with NumPy")
print("-" * 70)

# Convert predictions back to NumPy
predictions_np = output.numpy()

print(f"Predictions shape: {predictions_np.shape}")
print(f"Sample predictions (first image):")
print(f"  {predictions_np[0, :]}\n")

# Compute softmax in NumPy for proper probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

probs_np = softmax(predictions_np)
predicted_classes = np.argmax(probs_np, axis=1)

print("Predicted classes for each image:")
print(f"  {predicted_classes}\n")

# ============================================================================
# 6. Compute Features with Tensor4D, Analyze with NumPy
# ============================================================================
print("6. Feature Extraction and Statistical Analysis")
print("-" * 70)

# Extract features from hidden layer (a2)
features_np = a2.numpy()

print(f"Extracted features shape: {features_np.shape}")

# Compute feature statistics with NumPy
feature_means = features_np.mean(axis=0)
feature_stds = features_np.std(axis=0)
feature_maxs = features_np.max(axis=0)

print(f"Feature statistics:")
print(f"  Mean range: [{feature_means.min():.4f}, {feature_means.max():.4f}]")
print(f"  Std range: [{feature_stds.min():.4f}, {feature_stds.max():.4f}]")
print(f"  Max activation: {feature_maxs.max():.4f}\n")

# ============================================================================
# 7. Compute Similarity Matrix (NumPy + Tensor4D)
# ============================================================================
print("7. Computing Image Similarity")
print("-" * 70)

# Normalize features in NumPy
features_normalized = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)

# Convert to Tensor4D
features_t = t4d.Matrixf.from_numpy(features_normalized)

# Compute similarity matrix (cosine similarity)
# similarity[i, j] = features[i] · features[j]
similarity = features_t.matmul(features_t.transpose())
similarity_np = similarity.numpy()

print(f"Similarity matrix shape: {similarity_np.shape}")
print(f"Similarity range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")

# Find most similar pairs
np.fill_diagonal(similarity_np, -np.inf)  # Ignore self-similarity
most_similar = np.unravel_index(np.argmax(similarity_np), similarity_np.shape)
print(f"Most similar images: {most_similar[0]} and {most_similar[1]}")
print(f"  Similarity score: {similarity_np[most_similar]:.4f}\n")

# ============================================================================
# 8. Batch Processing with Mixed NumPy/Tensor4D
# ============================================================================
print("8. Batch Processing Pipeline")
print("-" * 70)

def process_batch(images_np, W1, W2, W3):
    """Process a batch using NumPy preprocessing and Tensor4D computation"""
    # Preprocess with NumPy
    normalized = (images_np - images_np.min()) / (images_np.max() - images_np.min() + 1e-8)
    flattened = normalized.reshape(len(normalized), -1)
    
    # Convert to Tensor4D
    X = t4d.Matrixf.from_numpy(flattened)
    
    # Forward pass
    a1 = X.matmul(W1).relu()
    a2 = a1.matmul(W2).relu()
    output = a2.matmul(W3).sigmoid()
    
    # Convert back to NumPy
    return output.numpy()

# Process multiple batches
num_batches = 3
all_predictions = []

for i in range(num_batches):
    # Generate new batch
    batch = np.random.randn(batch_size, height, width).astype(np.float32)
    
    # Process
    preds = process_batch(batch, W1, W2, W3)
    all_predictions.append(preds)
    
    print(f"Processed batch {i+1}/{num_batches}")

# Concatenate all predictions
all_predictions_np = np.concatenate(all_predictions, axis=0)
print(f"\nTotal predictions shape: {all_predictions_np.shape}")

# Aggregate statistics with NumPy
print(f"Aggregate prediction statistics:")
print(f"  Mean: {all_predictions_np.mean():.4f}")
print(f"  Std: {all_predictions_np.std():.4f}")
print(f"  Min: {all_predictions_np.min():.4f}")
print(f"  Max: {all_predictions_np.max():.4f}\n")

# ============================================================================
# 9. Save Results with NumPy
# ============================================================================
print("9. Saving Results")
print("-" * 70)

# Convert final results to NumPy and save
# In practice, you might do:
# np.save('predictions.npy', all_predictions_np)
# np.save('features.npy', features_np)
# np.savez('results.npz', predictions=all_predictions_np, features=features_np)

print("Results saved (in practice):")
print("  - predictions.npy")
print("  - features.npy")
print("  - results.npz\n")

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("Summary:")
print("  ✓ Generated and preprocessed image data with NumPy")
print("  ✓ Converted to Tensor4D for neural network forward pass")
print("  ✓ Performed efficient matrix operations with Tensor4D")
print("  ✓ Converted results back to NumPy for analysis")
print("  ✓ Computed statistics and similarity with mixed NumPy/Tensor4D")
print("  ✓ Demonstrated batch processing pipeline")
print("  ✓ Showed seamless integration between libraries")
print("=" * 70)
print("\nThis example demonstrates how Tensor4D and NumPy can work together")
print("in a real-world scenario, leveraging the strengths of both libraries:")
print("  - NumPy: Data loading, preprocessing, I/O, analysis")
print("  - Tensor4D: High-performance tensor operations, GPU acceleration")
