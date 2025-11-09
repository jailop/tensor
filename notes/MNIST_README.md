# MNIST Digit Classification Demo

This demo implements a simple feedforward neural network for classifying handwritten digits from the MNIST dataset using the `nn_layers.h` library.

## Network Architecture

- **Input Layer**: 784 neurons (28×28 flattened image)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation

Total parameters: ~109,000

## Dataset

The MNIST database contains 70,000 grayscale images of handwritten digits (0-9):
- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28×28 pixels

## Quick Start

### 1. Download the Dataset

#### Option A: Using the provided script (Recommended)
```bash
chmod +x download_mnist.sh
./download_mnist.sh
```

#### Option B: Manual download
Visit http://yann.lecun.com/exdb/mnist/ and download:
- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Then extract and place in `data/mnist/`:
```bash
mkdir -p data/mnist
cd data/mnist
# ... download files here ...
gunzip *.gz
cd ../..
```

#### Option C: One-line command
```bash
mkdir -p data/mnist && cd data/mnist && \
for file in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do \
  wget http://yann.lecun.com/exdb/mnist/${file}.gz && gunzip ${file}.gz; \
done && cd ../..
```

### 2. Build the Demo

```bash
mkdir -p build
cd build
cmake ..
make mnist_demo
cd ..
```

### 3. Run the Demo

```bash
./build/mnist_demo
```

Or specify a custom data path:
```bash
./build/mnist_demo /path/to/mnist/data/
```

## Expected Output

The program will:
1. Load the MNIST training and test datasets
2. Display the network architecture
3. Train for 5 epochs with batch size 32
4. Report training loss and accuracy after each epoch
5. Evaluate on the test set
6. Show sample predictions with confidence scores

Example output:
```
=== MNIST Digit Classification Demo ===
Using nn_layers.h neural network implementation

Data path: data/mnist/

--- Loading Training Data ---
Loading 60000 images (28x28)...
Loading 60000 labels...

--- Loading Test Data ---
Loading 10000 images (28x28)...
Loading 10000 labels...

Dataset loaded successfully!
Training samples: 60000
Test samples: 10000

--- Network Architecture ---
Input: 784 (28x28 flattened)
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 64 neurons (ReLU)
Output: 10 classes (Softmax)

--- Training Configuration ---
Batch size: 32
Number of epochs: 5
Learning rate: 0.01
Batches per epoch: 1875

=== Training Started ===
...
>>> Epoch 1 Summary:
    Average Loss: 0.4523
    Average Accuracy: 87.23%

...

=== Evaluating on Test Set ===
Test Accuracy: 89.45%

=== Sample Predictions ===
Image 0: True label = 7, Predicted = 7, Confidence = 95.32%
Image 1: True label = 2, Predicted = 2, Confidence = 91.87%
...
```

## Training Hyperparameters

You can modify these in the code:
- **Batch size**: 32 (line ~378)
- **Number of epochs**: 5 (line ~379)
- **Learning rate**: 0.01 (line ~380)

## Implementation Details

### Data Loading
The demo includes custom IDX file format readers for loading MNIST data:
- `load_mnist_images()`: Loads and normalizes images to [0, 1]
- `load_mnist_labels()`: Loads integer labels

### Training Loop
- Uses mini-batch gradient descent
- Cross-entropy loss function
- Simple SGD optimizer (can be extended to Adam, RMSprop, etc.)
- Reports loss and accuracy every 100 batches

### Neural Network Layers Used
From `nn_layers.h`:
- `Linear`: Fully connected layers with Xavier initialization
- `ReLU`: Rectified Linear Unit activation
- `Softmax`: Softmax activation for output probabilities

## Extending the Demo

### Improve Accuracy
1. Add more hidden layers or increase layer sizes
2. Implement data augmentation (rotation, translation)
3. Use better optimizers (Adam, momentum)
4. Add batch normalization layers
5. Implement learning rate scheduling
6. Add dropout for regularization

### Example: Add Dropout
```cpp
class MNISTNet {
public:
    MNISTNet() 
        : fc1_(IMAGE_PIXELS, 128, true),
          dropout1_(0.5f),  // Add dropout
          fc2_(128, 64, true),
          dropout2_(0.5f),  // Add dropout
          fc3_(64, NUM_CLASSES, true),
          // ... rest of initialization
    { }
    
    Tensor<float, 2> forward(const Tensor<float, 2>& input) {
        auto h1 = fc1_.forward(input);
        auto a1 = relu1_.forward(h1);
        auto d1 = dropout1_.forward(a1);  // Apply dropout
        auto h2 = fc2_.forward(d1);
        auto a2 = relu2_.forward(h2);
        auto d2 = dropout2_.forward(a2);  // Apply dropout
        auto h3 = fc3_.forward(d2);
        auto output = softmax_.forward(h3);
        return output;
    }
    
private:
    Linear<float> fc1_;
    Dropout<float> dropout1_;
    Linear<float> fc2_;
    Dropout<float> dropout2_;
    Linear<float> fc3_;
    // ... rest of members
};
```

## Performance Notes

- Training on CPU takes approximately 5-10 minutes for 5 epochs
- Expected test accuracy: 88-92% (without advanced techniques)
- With proper tuning and more epochs, can reach 95-98% accuracy

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits

## Troubleshooting

### "Cannot open file" error
Make sure the MNIST dataset is downloaded and placed in the correct directory (`data/mnist/` by default).

### Low accuracy
- Try training for more epochs
- Adjust learning rate (0.001 - 0.1 range)
- Increase batch size for more stable gradients

### Compilation errors
Ensure you have C++20 support:
```bash
g++ --version  # Should be GCC 10+ or Clang 10+
```
