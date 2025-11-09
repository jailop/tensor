# Python Bindings Changelog

## [Latest] - NumPy Interoperability

### Added

#### Core Features
- **NumPy to Tensor Conversion**: `Tensor.from_numpy(np_array)` class method
- **Tensor to NumPy Conversion**: `.numpy()` instance method
- **Auto-detection in Constructor**: Constructor now accepts NumPy arrays directly
- **All Tensor Types Supported**: Vectorf/d, Matrixf/d, Tensor3f/d, Tensor4f/d

#### Implementation Details
- Added `numpy_to_tensor<T, N>()` helper function in `tensor_wrapper.cc`
- Added `tensor_to_numpy<T, N>()` helper function in `tensor_wrapper.cc`
- Proper handling of NumPy strides (contiguous and non-contiguous arrays)
- Dimension and type validation with clear error messages
- Efficient memory copying optimized for contiguous arrays

#### Testing
- **test_numpy_interop.py**: Comprehensive test suite with 8 test functions
  - Vector conversion tests
  - Matrix conversion tests
  - 3D tensor conversion tests
  - Constructor tests
  - Operations compatibility tests
  - Matrix multiplication tests
  - Math functions tests
- All tests pass successfully

#### Examples
- **example_numpy_interop.py**: Comprehensive examples covering:
  1. Creating tensors from NumPy (2 methods)
  2. Converting tensors to NumPy
  3. Interoperability with NumPy operations
  4. Mathematical functions
  5. Higher-dimensional tensors
  6. Autograd with NumPy data
  7. Data preprocessing integration

- **example_realworld.py**: Real-world image processing pipeline showing:
  1. Synthetic image data generation
  2. Preprocessing with NumPy
  3. Neural network forward pass with Tensor4D
  4. Feature extraction and analysis
  5. Similarity computation
  6. Batch processing pipeline
  7. Result saving workflow

#### Documentation
- **README.md**: Complete Python bindings documentation
  - Installation instructions
  - Quick start guide
  - NumPy integration examples
  - API reference table
  - Performance notes
  - Ecosystem integration

- **NUMPY_INTEROP.md**: Detailed feature documentation
  - Overview of changes
  - Implementation details
  - Usage examples
  - Supported types
  - Benefits analysis
  - Testing results

- **Updated userguide/17-python-integration.md**:
  - New "NumPy Interoperability" section
  - Conversion examples
  - Working with NumPy data
  - Operations compatibility
  - Integration patterns
  - Complete workflow example

### Benefits

1. **Ecosystem Integration**: Seamless use with NumPy, pandas, matplotlib, scikit-learn
2. **No Hard Dependency**: NumPy is optional; library works with Python lists
3. **Data Loading**: Use NumPy's efficient I/O (`np.load`, `np.save`)
4. **Preprocessing**: Leverage NumPy/SciPy/scikit-learn for data preprocessing
5. **Visualization**: Direct integration with matplotlib
6. **Verification**: Easy validation against NumPy implementations
7. **Performance**: Efficient conversion with proper stride handling

### Technical Details

#### Memory Management
- Data is copied during conversion (not shared) for safety
- Optimized copy path for contiguous arrays
- Proper handling of non-contiguous arrays with stride calculations

#### Type Safety
- Dimension validation during conversion
- Data type compatibility checking
- Clear error messages for mismatches

#### Performance
- Zero-overhead for contiguous arrays
- Efficient element-wise copy for non-contiguous arrays
- Uses pybind11's NumPy integration for optimal performance

### Testing Results

```bash
$ python3 test_numpy_interop.py
============================================================
All NumPy interop tests passed!
============================================================

$ python3 test_bindings.py
Ran 19 tests in 0.233s
OK

$ python3 example_numpy_interop.py
[All examples run successfully]

$ python3 example_realworld.py
[Complete pipeline runs successfully]
```

### Files Modified

- `python/tensor_wrapper.cc` - Added NumPy conversion support

### Files Created

- `python/test_numpy_interop.py` - Test suite
- `python/example_numpy_interop.py` - Comprehensive example
- `python/example_realworld.py` - Real-world pipeline example
- `python/README.md` - Python bindings documentation
- `python/NUMPY_INTEROP.md` - Feature documentation
- `python/CHANGELOG.md` - This file

### Files Updated

- `userguide/17-python-integration.md` - Added NumPy section

### Compatibility

- Backward compatible with existing code
- Existing list-based constructors unchanged
- New NumPy methods are additive
- No breaking changes

### Future Enhancements

Potential future improvements:
- Zero-copy conversion using buffer protocol
- Support for more NumPy dtypes (int32, int64, etc.)
- Broadcasting compatibility
- Memory-mapped array support
- Sparse array support

---

## Previous Versions

See git history for previous changes to Python bindings.
