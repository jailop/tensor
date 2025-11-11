/**
 * @file tensor_wrapper.cc
 * @brief Python bindings for the Tensor4D library using pybind11
 * 
 * This file provides Python bindings for the instantiated tensor types:
 * - Vectorf, Vectord (1D tensors)
 * - Matrixf, Matrixd (2D tensors)
 * - Tensor3f, Tensor3d (3D tensors)
 * - Tensor4f, Tensor4d (4D tensors)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "tensor.h"
#include "tensor_types.h"
#include "linalg.h"
#include "loss_functions.h"
#include "tensor_io.h"
#include "optimizers.h"
#include "nn_layers.h"

namespace py = pybind11;
using namespace tensor4d;
using namespace tensor4d::nn;

// Helper function to flatten nested Python list
template<typename T>
void flatten_list(py::handle obj, std::vector<T>& data, std::vector<size_t>& shape, size_t depth = 0) {
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        auto seq = obj.cast<py::sequence>();
        if (depth >= shape.size()) {
            shape.push_back(seq.size());
        }
        for (size_t i = 0; i < seq.size(); ++i) {
            flatten_list<T>(seq[i], data, shape, depth + 1);
        }
    } else {
        // Leaf element
        data.push_back(obj.cast<T>());
    }
}

// Helper function to convert Python list/tuple to Tensor
template<typename T, size_t N>
Tensor<T, N> list_to_tensor(py::handle obj) {
    std::vector<T> data;
    std::vector<size_t> shape;
    
    flatten_list<T>(obj, data, shape, 0);
    
    if (shape.size() != N) {
        throw std::runtime_error("List dimension mismatch: expected " + std::to_string(N) + 
                               " dimensions, got " + std::to_string(shape.size()));
    }
    
    std::array<size_t, N> arr_shape;
    for (size_t i = 0; i < N; ++i) {
        arr_shape[i] = shape[i];
    }
    
    Tensor<T, N> tensor(arr_shape);
    if (data.size() != tensor.total_size()) {
        throw std::runtime_error("Data size mismatch");
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        tensor.data()[i] = data[i];
    }
    
    return tensor;
}

// Helper function to convert Tensor to nested Python list
template<typename T, size_t N>
py::list tensor_to_list(const Tensor<T, N>& tensor) {
    std::function<py::list(const T*, const std::array<size_t, N>&, size_t)> convert;
    
    convert = [&](const T* data, const std::array<size_t, N>& shape, size_t dim) -> py::list {
        py::list result;
        if (dim == N - 1) {
            // Last dimension: add elements directly
            for (size_t i = 0; i < shape[dim]; ++i) {
                result.append(data[i]);
            }
        } else {
            // Recursive case
            size_t stride = 1;
            for (size_t d = dim + 1; d < N; ++d) {
                stride *= shape[d];
            }
            for (size_t i = 0; i < shape[dim]; ++i) {
                result.append(convert(data + i * stride, shape, dim + 1));
            }
        }
        return result;
    };
    
    return convert(tensor.data(), tensor.shape(), 0);
}

// Helper function to convert NumPy array to Tensor
template<typename T, size_t N>
Tensor<T, N> numpy_to_tensor(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    
    if (buf.ndim != N) {
        throw std::runtime_error("NumPy array dimension mismatch: expected " + 
                               std::to_string(N) + " dimensions, got " + 
                               std::to_string(buf.ndim));
    }
    
    std::array<size_t, N> shape;
    for (size_t i = 0; i < N; ++i) {
        shape[i] = buf.shape[i];
    }
    
    Tensor<T, N> tensor(shape);
    T* src = static_cast<T*>(buf.ptr);
    
    // Copy data from NumPy array (handles different strides)
    if (buf.strides[N-1] == sizeof(T)) {
        // Contiguous case - direct copy
        std::copy(src, src + tensor.total_size(), tensor.data());
    } else {
        // Non-contiguous case - copy element by element
        std::function<void(size_t, const ssize_t*, T*)> copy_recursive;
        copy_recursive = [&](size_t dim, const ssize_t* indices, T* dst) {
            if (dim == N) {
                size_t offset = 0;
                for (size_t i = 0; i < N; ++i) {
                    offset += indices[i] * buf.strides[i] / sizeof(T);
                }
                *dst = src[offset];
            } else {
                for (ssize_t i = 0; i < buf.shape[dim]; ++i) {
                    ssize_t new_indices[N];
                    for (size_t j = 0; j < dim; ++j) {
                        new_indices[j] = indices[j];
                    }
                    new_indices[dim] = i;
                    copy_recursive(dim + 1, new_indices, dst);
                    if (dim == N - 1) dst++;
                }
            }
        };
        ssize_t indices[N] = {0};
        copy_recursive(0, indices, tensor.data());
    }
    
    return tensor;
}

// Helper function to convert Tensor to NumPy array
template<typename T, size_t N>
py::array_t<T> tensor_to_numpy(const Tensor<T, N>& tensor) {
    auto shape = tensor.shape();
    std::vector<ssize_t> np_shape(N);
    for (size_t i = 0; i < N; ++i) {
        np_shape[i] = shape[i];
    }
    
    // Create NumPy array with a copy of the data
    py::array_t<T> arr(np_shape);
    py::buffer_info buf = arr.request();
    T* dst = static_cast<T*>(buf.ptr);
    
    std::copy(tensor.data(), tensor.data() + tensor.total_size(), dst);
    
    return arr;
}

// Binding for a single Tensor type
template<typename T, size_t N>
py::class_<Tensor<T, N>> bind_tensor(py::module& m, const std::string& name) {
    using TensorType = Tensor<T, N>;
    
    py::class_<TensorType> cls(m, name.c_str());
    
    cls.def(py::init<const std::array<size_t, N>&>(), py::arg("shape"),
             "Create a tensor with the given shape")
        .def(py::init([](py::handle data) {
            // Try to interpret as NumPy array first
            if (py::isinstance<py::array>(data)) {
                return numpy_to_tensor<T, N>(data.cast<py::array_t<T>>());
            }
            // Otherwise treat as list/tuple
            return list_to_tensor<T, N>(data);
        }), py::arg("data"), "Create tensor from Python list/tuple or NumPy array")
        
        // Properties
        .def_property_readonly("shape", [](const TensorType& t) {
            auto shp = t.shape();
            py::tuple result(N);
            for (size_t i = 0; i < N; ++i) {
                result[i] = shp[i];
            }
            return result;
        }, "Get tensor shape")
        .def_property_readonly("size", &TensorType::total_size, "Get total number of elements")
        .def_property_readonly("requires_grad", &TensorType::requires_grad, "Check if tensor requires gradient")
        .def("set_requires_grad", &TensorType::set_requires_grad, py::arg("value"),
             "Set requires_grad flag")
        
        // Element access
        .def("__getitem__", [](const TensorType& t, const std::array<size_t, N>& idx) {
            return t[idx];
        })
        .def("__setitem__", [](TensorType& t, const std::array<size_t, N>& idx, T value) {
            t[idx] = value;
        })
        .def("at", [](const TensorType& t, const std::vector<size_t>& idx) {
            std::array<size_t, N> arr_idx;
            for (size_t i = 0; i < N && i < idx.size(); ++i) {
                arr_idx[i] = idx[i];
            }
            return t[arr_idx];
        }, py::arg("index"), "Access element at index")
        
        // Fill operations
        .def("fill", &TensorType::fill, py::arg("value"), "Fill tensor with value")
        
        // Arithmetic operations - tensor-tensor (returns TensorResult)
        .def("__add__", [](const TensorType& a, const TensorType& b) -> TensorType {
            auto result = a + b;
            if (std::holds_alternative<TensorError>(result)) {
                throw std::runtime_error("Tensor addition failed");
            }
            return std::get<TensorType>(result);
        })
        .def("__sub__", [](const TensorType& a, const TensorType& b) -> TensorType {
            auto result = a - b;
            if (std::holds_alternative<TensorError>(result)) {
                throw std::runtime_error("Tensor subtraction failed");
            }
            return std::get<TensorType>(result);
        })
        .def("__mul__", [](const TensorType& a, const TensorType& b) -> TensorType {
            auto result = a * b;
            if (std::holds_alternative<TensorError>(result)) {
                throw std::runtime_error("Tensor multiplication failed");
            }
            return std::get<TensorType>(result);
        })
        .def("__truediv__", [](const TensorType& a, const TensorType& b) -> TensorType {
            auto result = a / b;
            if (std::holds_alternative<TensorError>(result)) {
                throw std::runtime_error("Tensor division failed");
            }
            return std::get<TensorType>(result);
        })
        // Arithmetic operations - tensor-scalar (returns Tensor directly)
        .def("__add__", [](const TensorType& a, T scalar) -> TensorType {
            return a + scalar;
        })
        .def("__sub__", [](const TensorType& a, T scalar) -> TensorType {
            return a - scalar;
        })
        .def("__mul__", [](const TensorType& a, T scalar) -> TensorType {
            return a * scalar;
        })
        .def("__truediv__", [](const TensorType& a, T scalar) -> TensorType {
            return a / scalar;
        })
        // Arithmetic operations - scalar-tensor (returns Tensor directly)
        .def("__radd__", [](const TensorType& a, T scalar) -> TensorType {
            return scalar + a;
        })
        .def("__rsub__", [](const TensorType& a, T scalar) -> TensorType {
            return scalar - a;
        })
        .def("__rmul__", [](const TensorType& a, T scalar) -> TensorType {
            return scalar * a;
        })
        .def("__rtruediv__", [](const TensorType& a, T scalar) -> TensorType {
            return scalar / a;
        })
        .def("__iadd__", [](TensorType& self, const TensorType& other) -> TensorType& { 
            self += other; 
            return self; 
        })
        .def("__isub__", [](TensorType& self, const TensorType& other) -> TensorType& { 
            self -= other; 
            return self; 
        })
        .def("__imul__", [](TensorType& self, const TensorType& other) -> TensorType& { 
            self *= other; 
            return self; 
        })
        .def("__itruediv__", [](TensorType& self, const TensorType& other) -> TensorType& { 
            self /= other; 
            return self; 
        })
        
        // Math functions
        .def("exp", &TensorType::exp, "Element-wise exponential")
        .def("log", &TensorType::log, "Element-wise natural logarithm")
        .def("sqrt", &TensorType::sqrt, "Element-wise square root")
        .def("pow", &TensorType::pow, py::arg("exponent"), "Element-wise power")
        .def("sin", &TensorType::sin, "Element-wise sine")
        .def("cos", &TensorType::cos, "Element-wise cosine")
        .def("tan", &TensorType::tan, "Element-wise tangent")
        .def("abs", &TensorType::abs, "Element-wise absolute value")
        .def("clip", &TensorType::clip, py::arg("min"), py::arg("max"), "Clip values to range")
        
        // Activation functions
        .def("sigmoid", &TensorType::sigmoid, "Sigmoid activation")
        .def("relu", &TensorType::relu, "ReLU activation")
        
        // Statistical operations
        .def("sum", &TensorType::sum, "Sum of all elements")
        .def("mean", &TensorType::mean, "Mean of all elements")
        .def("variance", &TensorType::variance, "Variance of all elements")
        .def("std", &TensorType::std, "Standard deviation")
        .def("min", &TensorType::min, "Minimum element")
        .def("max", &TensorType::max, "Maximum element")
        .def("median", &TensorType::median, "Median element")
        .def("prod", &TensorType::prod, "Product of all elements")
        
        // Reduction operations
        .def("all", &TensorType::all, "Check if all elements are non-zero")
        .def("any", &TensorType::any, "Check if any element is non-zero")
        .def("argmin", &TensorType::argmin, "Index of minimum element")
        .def("argmax", &TensorType::argmax, "Index of maximum element")
        .def("cumsum", &TensorType::cumsum, "Cumulative sum")
        .def("cumprod", &TensorType::cumprod, "Cumulative product")
        
        // Autograd
        .def("backward", 
             [](TensorType& self) {
                 auto err = self.backward(nullptr);
                 if (err.has_value()) {
                     throw std::runtime_error("backward() failed");
                 }
             },
             "Compute gradients via backpropagation")
        .def("grad", [](TensorType& self) -> TensorType* {
            return self.grad();
        }, py::return_value_policy::reference_internal, "Get gradient tensor")
        .def("zero_grad", &TensorType::zero_grad, "Zero out gradients")
        .def("detach", &TensorType::detach, "Detach from computation graph")
        
        // Conversion
        .def("tolist", [](const TensorType& t) {
            return tensor_to_list(t);
        }, "Convert to Python list")
        .def("numpy", [](const TensorType& t) {
            return tensor_to_numpy(t);
        }, "Convert to NumPy array")
        .def_static("from_numpy", [](py::array_t<T> arr) {
            return numpy_to_tensor<T, N>(arr);
        }, py::arg("array"), "Create tensor from NumPy array")
        .def("__repr__", [](const TensorType& t) {
            return ::to_string(t);
        });
        
    return cls;
}

// Bind Matrix-specific operations
template<typename T>
void bind_matrix_ops(py::module& m, py::class_<Tensor<T, 2>>& cls) {
    using MatrixType = Tensor<T, 2>;
    
    cls.def("matmul", 
            [](const MatrixType& a, const MatrixType& b) -> MatrixType {
                return linalg::matmul(a, b);
            }, 
            py::arg("other"), "Matrix multiplication")
       .def("transpose", 
            [](const MatrixType& a) {
                return a.transpose();
            }, "Matrix transpose")
       .def("__matmul__", 
            [](const MatrixType& a, const MatrixType& b) -> MatrixType {
                return linalg::matmul(a, b);
            }, 
            py::arg("other"), "Matrix multiplication (@)")
       .def("rank", [](const MatrixType& a, T tol) {
            return linalg::rank(a, tol);
       }, py::arg("tol") = T(-1), "Matrix rank")
       .def("trace", [](const MatrixType& a) {
            return linalg::trace(a);
       }, "Matrix trace")
       .def("frobenius_norm", [](const MatrixType& a) {
            return linalg::frobenius_norm(a);
       }, "Frobenius norm")
       .def("condition_number", [](const MatrixType& a) {
            return linalg::condition_number(a);
       }, "Condition number");
}

// Bind Vector-specific operations
template<typename T>
void bind_vector_ops(py::module& m, py::class_<Tensor<T, 1>>& cls) {
    using VectorType = Tensor<T, 1>;
    
    cls.def("dot", 
            [](const VectorType& a, const VectorType& b) {
                return linalg::dot(a, b);
            }, 
            py::arg("other"), "Dot product")
       .def("norm", [](const VectorType& v) {
            return linalg::norm(v);
       }, "Euclidean norm")
       .def("normalize", [](const VectorType& v) {
            return linalg::normalize(v);
       }, "Normalize vector");
}

PYBIND11_MODULE(tensor4d, m) {
    m.doc() = "Python bindings for Tensor4D library";
    
    // TensorError enum
    py::enum_<TensorError>(m, "TensorError")
        .value("DimensionMismatch", TensorError::DimensionMismatch)
        .value("ContractionMismatch", TensorError::ContractionMismatch)
        .value("InvalidArgument", TensorError::InvalidArgument)
        .export_values();
    
    // Float tensors with specialized operations
    {
        auto cls = bind_tensor<float, 1>(m, "Vectorf");
        bind_vector_ops<float>(m, cls);
    }
    
    {
        auto cls = bind_tensor<float, 2>(m, "Matrixf");
        bind_matrix_ops<float>(m, cls);
    }
    
    bind_tensor<float, 3>(m, "Tensor3f");
    bind_tensor<float, 4>(m, "Tensor4f");
    
    // Double tensors with specialized operations
    {
        auto cls = bind_tensor<double, 1>(m, "Vectord");
        bind_vector_ops<double>(m, cls);
    }
    
    {
        auto cls = bind_tensor<double, 2>(m, "Matrixd");
        bind_matrix_ops<double>(m, cls);
    }
    
    bind_tensor<double, 3>(m, "Tensor3d");
    bind_tensor<double, 4>(m, "Tensor4d");
    
    // Loss functions
    m.def("mse_loss", 
          [](const Tensor<float, 2>& pred, const Tensor<float, 2>& target, const std::string& reduction) -> Tensor<float, 2> {
              return loss::mse_loss(pred, target, reduction);
          }, 
          py::arg("predictions"), py::arg("targets"), py::arg("reduction") = "mean",
          "Mean Squared Error loss");
    
    m.def("cross_entropy_loss", 
          [](const Tensor<float, 2>& pred, const Tensor<float, 2>& target, const std::string& reduction) -> Tensor<float, 1> {
              return loss::cross_entropy_loss(pred, target, reduction);
          },
          py::arg("predictions"), py::arg("targets"), py::arg("reduction") = "mean",
          "Cross Entropy loss");
    
    m.def("binary_cross_entropy", 
          [](const Tensor<float, 2>& pred, const Tensor<float, 2>& target, const std::string& reduction) -> Tensor<float, 1> {
              return loss::binary_cross_entropy(pred, target, reduction);
          },
          py::arg("predictions"), py::arg("targets"), py::arg("reduction") = "mean",
          "Binary Cross Entropy loss");
    
    m.def("binary_cross_entropy_loss", 
          [](const Tensor<float, 2>& pred, const Tensor<float, 2>& target, const std::string& reduction) -> Tensor<float, 1> {
              return loss::binary_cross_entropy(pred, target, reduction);
          },
          py::arg("predictions"), py::arg("targets"), py::arg("reduction") = "mean",
          "Binary Cross Entropy loss (alias)");
    
    // File I/O  
    py::enum_<TensorFileFormat>(m, "TensorIOFormat")
        .value("BINARY", TensorFileFormat::BINARY)
        .value("TEXT", TensorFileFormat::TEXT)
        .value("NPY", TensorFileFormat::NPY)
        .export_values();
    
    m.def("save_tensor", 
          [](const Tensor<float, 2>& tensor, const std::string& filename, TensorFileFormat format) {
              bool success = save(tensor, filename, format);
              if (!success) {
                  throw std::runtime_error("Failed to save tensor");
              }
          },
          py::arg("tensor"), py::arg("filename"), py::arg("format") = TensorFileFormat::BINARY,
          "Save tensor to file");
    
    m.def("load_tensor_f2", [](const std::string& filename) -> Tensor<float, 2> {
        auto result = load<float, 2>(filename);
        if (std::holds_alternative<TensorIOError>(result)) {
            throw std::runtime_error("Failed to load tensor from file");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("filename"), "Load float matrix from file");
    
    m.def("save_binary", 
          [](const Tensor<float, 2>& tensor, const std::string& filename) {
              return save_binary(tensor, filename);
          },
          py::arg("tensor"), py::arg("filename"),
          "Save tensor to file in binary format");
    
    m.def("load_binary_f2", [](const std::string& filename) {
        auto result = load_binary<float, 2>(filename);
        if (std::holds_alternative<TensorIOError>(result)) {
            throw std::runtime_error("Failed to load tensor from file");
        }
        return std::get<Tensor<float, 2>>(result);
    }, py::arg("filename"), "Load float matrix from binary file");
    
    // Optimizer bindings
    
    // Base Optimizer class (abstract)
    py::class_<Optimizer<float>>(m, "Optimizer")
        .def("get_lr", &Optimizer<float>::get_lr, "Get current learning rate")
        .def("set_lr", &Optimizer<float>::set_lr, py::arg("lr"), "Set learning rate");
    
    // SGD optimizer for float matrices
    py::class_<SGD<float, 2>, Optimizer<float>>(m, "SGD")
        .def(py::init<std::vector<Tensor<float, 2>*>, float, float, float>(),
            py::arg("parameters"),
            py::arg("learning_rate") = 0.01f,
            py::arg("momentum") = 0.0f,
            py::arg("weight_decay") = 0.0f,
            "Create SGD optimizer with parameters")
        .def("step", &SGD<float, 2>::step, "Perform optimization step")
        .def("zero_grad", &SGD<float, 2>::zero_grad, "Zero out gradients");
    
    // Adam optimizer for float matrices
    py::class_<Adam<float, 2>, Optimizer<float>>(m, "Adam")
        .def(py::init<std::vector<Tensor<float, 2>*>, float, float, float, float, float>(),
            py::arg("parameters"),
            py::arg("learning_rate") = 0.001f,
            py::arg("beta1") = 0.9f,
            py::arg("beta2") = 0.999f,
            py::arg("epsilon") = 1e-8f,
            py::arg("weight_decay") = 0.0f,
            "Create Adam optimizer with parameters")
        .def("step", &Adam<float, 2>::step, "Perform optimization step")
        .def("zero_grad", &Adam<float, 2>::zero_grad, "Zero out gradients")
        .def("reset", &Adam<float, 2>::reset, "Reset optimizer state");
    
    // RMSprop optimizer for float matrices
    py::class_<RMSprop<float, 2>, Optimizer<float>>(m, "RMSprop")
        .def(py::init<std::vector<Tensor<float, 2>*>, float, float, float, float, float>(),
            py::arg("parameters"),
            py::arg("learning_rate") = 0.01f,
            py::arg("alpha") = 0.99f,
            py::arg("epsilon") = 1e-8f,
            py::arg("weight_decay") = 0.0f,
            py::arg("momentum") = 0.0f,
            "Create RMSprop optimizer with parameters")
        .def("step", &RMSprop<float, 2>::step, "Perform optimization step")
        .def("zero_grad", &RMSprop<float, 2>::zero_grad, "Zero out gradients");
    
    // Learning rate scheduler
    py::class_<ExponentialLR<float>>(m, "ExponentialLR")
        .def(py::init<Optimizer<float>*, float>(),
            py::arg("optimizer"),
            py::arg("gamma"),
            "Create exponential learning rate scheduler")
        .def("step", &ExponentialLR<float>::step, "Decay learning rate")
        .def("reset", &ExponentialLR<float>::reset, "Reset to initial learning rate");
    
    // Linear algebra functions module
    py::module linalg_mod = m.def_submodule("linalg", "Linear algebra functions");
    
    // Vector operations
    linalg_mod.def("norm", [](const Tensor<float, 1>& v) {
        return linalg::norm(v);
    }, py::arg("v"), "Vector norm");
    
    linalg_mod.def("norm", [](const Tensor<double, 1>& v) {
        return linalg::norm(v);
    }, py::arg("v"), "Vector norm (double)");
    
    linalg_mod.def("normalize", [](const Tensor<float, 1>& v) {
        return linalg::normalize(v);
    }, py::arg("v"), "Normalize vector");
    
    linalg_mod.def("normalize", [](const Tensor<double, 1>& v) {
        return linalg::normalize(v);
    }, py::arg("v"), "Normalize vector (double)");
    
    linalg_mod.def("dot", [](const Tensor<float, 1>& a, const Tensor<float, 1>& b) {
        return linalg::dot(a, b);
    }, py::arg("a"), py::arg("b"), "Dot product");
    
    linalg_mod.def("dot", [](const Tensor<double, 1>& a, const Tensor<double, 1>& b) {
        return linalg::dot(a, b);
    }, py::arg("a"), py::arg("b"), "Dot product (double)");
    
    linalg_mod.def("outer", [](const Tensor<float, 1>& a, const Tensor<float, 1>& b) {
        return linalg::outer(a, b);
    }, py::arg("a"), py::arg("b"), "Outer product");
    
    linalg_mod.def("outer", [](const Tensor<double, 1>& a, const Tensor<double, 1>& b) {
        return linalg::outer(a, b);
    }, py::arg("a"), py::arg("b"), "Outer product (double)");
    
    // Matrix operations
    linalg_mod.def("matmul", [](const Tensor<float, 2>& A, const Tensor<float, 2>& B) {
        return linalg::matmul(A, B);
    }, py::arg("A"), py::arg("B"), "Matrix multiplication");
    
    linalg_mod.def("matmul", [](const Tensor<double, 2>& A, const Tensor<double, 2>& B) {
        return linalg::matmul(A, B);
    }, py::arg("A"), py::arg("B"), "Matrix multiplication (double)");
    
    linalg_mod.def("matvec", [](const Tensor<float, 2>& mat, const Tensor<float, 1>& vec) {
        return linalg::matvec(mat, vec);
    }, py::arg("mat"), py::arg("vec"), "Matrix-vector multiplication");
    
    linalg_mod.def("matvec", [](const Tensor<double, 2>& mat, const Tensor<double, 1>& vec) {
        return linalg::matvec(mat, vec);
    }, py::arg("mat"), py::arg("vec"), "Matrix-vector multiplication (double)");
    
    linalg_mod.def("transpose", [](const Tensor<float, 2>& mat) {
        return linalg::transpose(mat);
    }, py::arg("mat"), "Matrix transpose");
    
    linalg_mod.def("transpose", [](const Tensor<double, 2>& mat) {
        return linalg::transpose(mat);
    }, py::arg("mat"), "Matrix transpose (double)");
    
    linalg_mod.def("trace", [](const Tensor<float, 2>& mat) {
        return linalg::trace(mat);
    }, py::arg("mat"), "Matrix trace");
    
    linalg_mod.def("trace", [](const Tensor<double, 2>& mat) {
        return linalg::trace(mat);
    }, py::arg("mat"), "Matrix trace (double)");
    
    linalg_mod.def("diag", [](const Tensor<float, 2>& mat) {
        return linalg::diag(mat);
    }, py::arg("mat"), "Extract diagonal");
    
    linalg_mod.def("diag", [](const Tensor<double, 2>& mat) {
        return linalg::diag(mat);
    }, py::arg("mat"), "Extract diagonal (double)");
    
    linalg_mod.def("diag", [](const Tensor<float, 1>& vec) {
        return linalg::diag(vec);
    }, py::arg("vec"), "Create diagonal matrix");
    
    linalg_mod.def("diag", [](const Tensor<double, 1>& vec) {
        return linalg::diag(vec);
    }, py::arg("vec"), "Create diagonal matrix (double)");
    
    linalg_mod.def("eye", [](size_t n, bool use_gpu) {
        return linalg::eye<float>(n, use_gpu);
    }, py::arg("n"), py::arg("use_gpu") = true, "Identity matrix");
    
    linalg_mod.def("eye", [](size_t n, bool use_gpu) {
        return linalg::eye<double>(n, use_gpu);
    }, py::arg("n"), py::arg("use_gpu") = true, "Identity matrix (double)");
    
    linalg_mod.def("frobenius_norm", [](const Tensor<float, 2>& mat) {
        return linalg::frobenius_norm(mat);
    }, py::arg("mat"), "Frobenius norm");
    
    linalg_mod.def("frobenius_norm", [](const Tensor<double, 2>& mat) {
        return linalg::frobenius_norm(mat);
    }, py::arg("mat"), "Frobenius norm (double)");
    
    linalg_mod.def("norm_l1", [](const Tensor<float, 2>& mat) {
        return linalg::norm_l1(mat);
    }, py::arg("mat"), "L1 matrix norm");
    
    linalg_mod.def("norm_l1", [](const Tensor<double, 2>& mat) {
        return linalg::norm_l1(mat);
    }, py::arg("mat"), "L1 matrix norm (double)");
    
    linalg_mod.def("norm_inf", [](const Tensor<float, 2>& mat) {
        return linalg::norm_inf(mat);
    }, py::arg("mat"), "Infinity matrix norm");
    
    linalg_mod.def("norm_inf", [](const Tensor<double, 2>& mat) {
        return linalg::norm_inf(mat);
    }, py::arg("mat"), "Infinity matrix norm (double)");
    
    linalg_mod.def("matrix_rank", [](const Tensor<float, 2>& mat, float tol) {
        return linalg::rank(mat, tol);
    }, py::arg("mat"), py::arg("tol") = -1.0f, "Matrix rank");
    
    linalg_mod.def("matrix_rank", [](const Tensor<double, 2>& mat, double tol) {
        return linalg::rank(mat, tol);
    }, py::arg("mat"), py::arg("tol") = -1.0, "Matrix rank (double)");
    
    linalg_mod.def("condition_number", [](const Tensor<float, 2>& mat) {
        return linalg::condition_number(mat);
    }, py::arg("mat"), "Condition number");
    
    linalg_mod.def("condition_number", [](const Tensor<double, 2>& mat) {
        return linalg::condition_number(mat);
    }, py::arg("mat"), "Condition number (double)");
    
    linalg_mod.def("lstsq", [](const Tensor<float, 2>& A, const Tensor<float, 1>& b) {
        return linalg::least_squares(A, b);
    }, py::arg("A"), py::arg("b"), "Least squares solution");
    
    linalg_mod.def("lstsq", [](const Tensor<double, 2>& A, const Tensor<double, 1>& b) {
        return linalg::least_squares(A, b);
    }, py::arg("A"), py::arg("b"), "Least squares solution (double)");
    
    // ========================================================================
    // Neural Network Layers Module
    // ========================================================================
    
    auto nn_mod = m.def_submodule("nn", "Neural network layers");
    
    // Linear Layer (Float)
    py::class_<Linear<float>>(nn_mod, "Linearf")
        .def(py::init<size_t, size_t, bool>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("use_bias") = true,
             "Create a linear (fully connected) layer")
        .def("forward", &Linear<float>::forward,
             py::arg("input"),
             "Forward pass through the layer")
        .def("backward", &Linear<float>::backward,
             py::arg("grad_output"),
             "Backward pass (compute gradients)")
        .def("parameters", &Linear<float>::parameters,
             py::return_value_policy::reference_internal,
             "Get list of trainable parameters")
        .def("train", &Linear<float>::train,
             py::arg("mode") = true,
             "Set training mode")
        .def("is_training", &Linear<float>::is_training,
             "Check if layer is in training mode")
        .def_property_readonly("weights", 
             [](Linear<float>& layer) -> Matrixf& { return layer.weights(); },
             py::return_value_policy::reference_internal,
             "Access weights matrix")
        .def_property_readonly("bias",
             [](Linear<float>& layer) -> Matrixf& { return layer.bias(); },
             py::return_value_policy::reference_internal,
             "Access bias vector")
        .def("__repr__", [](const Linear<float>&) {
            return "<tensor4d.nn.Linearf>";
        });
    
    // Linear Layer (Double)
    py::class_<Linear<double>>(nn_mod, "Lineard")
        .def(py::init<size_t, size_t, bool>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("use_bias") = true)
        .def("forward", &Linear<double>::forward)
        .def("backward", &Linear<double>::backward)
        .def("parameters", &Linear<double>::parameters)
        .def("train", &Linear<double>::train, py::arg("mode") = true)
        .def("is_training", &Linear<double>::is_training);
    
    // ReLU Layer (Float)
    py::class_<ReLU<float>>(nn_mod, "ReLUf")
        .def(py::init<>(), "Create a ReLU activation layer")
        .def("forward", &ReLU<float>::forward,
             py::arg("input"),
             "Apply ReLU activation")
        .def("backward", &ReLU<float>::backward,
             py::arg("grad_output"),
             "Backward pass for ReLU")
        .def("train", &ReLU<float>::train, py::arg("mode") = true)
        .def("is_training", &ReLU<float>::is_training)
        .def("__repr__", [](const ReLU<float>&) {
            return "<tensor4d.nn.ReLUf>";
        });
    
    // ReLU Layer (Double)
    py::class_<ReLU<double>>(nn_mod, "ReLUd")
        .def(py::init<>())
        .def("forward", &ReLU<double>::forward)
        .def("backward", &ReLU<double>::backward)
        .def("train", &ReLU<double>::train, py::arg("mode") = true)
        .def("is_training", &ReLU<double>::is_training);
    
    // Sigmoid Layer (Float)
    py::class_<Sigmoid<float>>(nn_mod, "Sigmoidf")
        .def(py::init<>(), "Create a Sigmoid activation layer")
        .def("forward", &Sigmoid<float>::forward,
             py::arg("input"),
             "Apply Sigmoid activation")
        .def("backward", &Sigmoid<float>::backward,
             py::arg("grad_output"),
             "Backward pass for Sigmoid")
        .def("train", &Sigmoid<float>::train, py::arg("mode") = true)
        .def("is_training", &Sigmoid<float>::is_training)
        .def("__repr__", [](const Sigmoid<float>&) {
            return "<tensor4d.nn.Sigmoidf>";
        });
    
    // Sigmoid Layer (Double)
    py::class_<Sigmoid<double>>(nn_mod, "Sigmoidd")
        .def(py::init<>())
        .def("forward", &Sigmoid<double>::forward)
        .def("backward", &Sigmoid<double>::backward)
        .def("train", &Sigmoid<double>::train, py::arg("mode") = true)
        .def("is_training", &Sigmoid<double>::is_training);
    
    // Tanh Layer (Float)
    py::class_<Tanh<float>>(nn_mod, "Tanhf")
        .def(py::init<>(), "Create a Tanh activation layer")
        .def("forward", &Tanh<float>::forward,
             py::arg("input"),
             "Apply Tanh activation")
        .def("backward", &Tanh<float>::backward,
             py::arg("grad_output"),
             "Backward pass for Tanh")
        .def("train", &Tanh<float>::train, py::arg("mode") = true)
        .def("is_training", &Tanh<float>::is_training)
        .def("__repr__", [](const Tanh<float>&) {
            return "<tensor4d.nn.Tanhf>";
        });
    
    // Tanh Layer (Double)
    py::class_<Tanh<double>>(nn_mod, "Tanhd")
        .def(py::init<>())
        .def("forward", &Tanh<double>::forward)
        .def("backward", &Tanh<double>::backward)
        .def("train", &Tanh<double>::train, py::arg("mode") = true)
        .def("is_training", &Tanh<double>::is_training);
    
    // Softmax Layer (Float)
    py::class_<Softmax<float>>(nn_mod, "Softmaxf")
        .def(py::init<>(), "Create a Softmax activation layer")
        .def("forward", &Softmax<float>::forward,
             py::arg("input"),
             "Apply Softmax (converts to probabilities)")
        .def("backward", &Softmax<float>::backward,
             py::arg("grad_output"),
             "Backward pass for Softmax")
        .def("train", &Softmax<float>::train, py::arg("mode") = true)
        .def("is_training", &Softmax<float>::is_training)
        .def("__repr__", [](const Softmax<float>&) {
            return "<tensor4d.nn.Softmaxf>";
        });
    
    // Softmax Layer (Double)
    py::class_<Softmax<double>>(nn_mod, "Softmaxd")
        .def(py::init<>())
        .def("forward", &Softmax<double>::forward)
        .def("backward", &Softmax<double>::backward)
        .def("train", &Softmax<double>::train, py::arg("mode") = true)
        .def("is_training", &Softmax<double>::is_training);
    
    // Dropout Layer (Float)
    py::class_<Dropout<float>>(nn_mod, "Dropoutf")
        .def(py::init<float>(),
             py::arg("p") = 0.5f,
             "Create a Dropout layer (p = dropout probability)")
        .def("forward", &Dropout<float>::forward,
             py::arg("input"),
             "Apply dropout (training) or pass-through (inference)")
        .def("backward", &Dropout<float>::backward,
             py::arg("grad_output"),
             "Backward pass for Dropout")
        .def("train", &Dropout<float>::train,
             py::arg("mode") = true,
             "Set training/inference mode")
        .def("is_training", &Dropout<float>::is_training)
        .def("__repr__", [](const Dropout<float>&) {
            return "<tensor4d.nn.Dropoutf>";
        });
    
    // Dropout Layer (Double)
    py::class_<Dropout<double>>(nn_mod, "Dropoutd")
        .def(py::init<double>(), py::arg("p") = 0.5)
        .def("forward", &Dropout<double>::forward)
        .def("backward", &Dropout<double>::backward)
        .def("train", &Dropout<double>::train, py::arg("mode") = true)
        .def("is_training", &Dropout<double>::is_training);
    
    // Batch Normalization Layer (Float)
    py::class_<BatchNorm1d<float>>(nn_mod, "BatchNorm1df")
        .def(py::init<size_t, float, float>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5f,
             py::arg("momentum") = 0.1f,
             "Create a Batch Normalization layer")
        .def("forward", &BatchNorm1d<float>::forward,
             py::arg("input"),
             "Apply batch normalization")
        .def("backward", &BatchNorm1d<float>::backward,
             py::arg("grad_output"),
             "Backward pass for Batch Normalization")
        .def("parameters", &BatchNorm1d<float>::parameters,
             py::return_value_policy::reference_internal,
             "Get gamma and beta parameters")
        .def("train", &BatchNorm1d<float>::train,
             py::arg("mode") = true,
             "Set training/inference mode")
        .def("is_training", &BatchNorm1d<float>::is_training)
        .def("__repr__", [](const BatchNorm1d<float>&) {
            return "<tensor4d.nn.BatchNorm1df>";
        });
    
    // Batch Normalization Layer (Double)
    py::class_<BatchNorm1d<double>>(nn_mod, "BatchNorm1dd")
        .def(py::init<size_t, double, double>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5,
             py::arg("momentum") = 0.1)
        .def("forward", &BatchNorm1d<double>::forward)
        .def("backward", &BatchNorm1d<double>::backward)
        .def("parameters", &BatchNorm1d<double>::parameters)
        .def("train", &BatchNorm1d<double>::train, py::arg("mode") = true)
        .def("is_training", &BatchNorm1d<double>::is_training);
    
    // Helper functions for neural networks
    
    // label_to_onehot - Float
    nn_mod.def("label_to_onehot_f", 
        [](uint8_t label, Matrixf& onehot, size_t batch_idx, size_t num_classes) {
            label_to_onehot(label, onehot, batch_idx, num_classes);
        },
        py::arg("label"),
        py::arg("onehot"),
        py::arg("batch_idx"),
        py::arg("num_classes"),
        "Convert a single label to one-hot encoding in a batch tensor");
    
    // label_to_onehot - Double
    nn_mod.def("label_to_onehot_d",
        [](uint8_t label, Matrixd& onehot, size_t batch_idx, size_t num_classes) {
            label_to_onehot(label, onehot, batch_idx, num_classes);
        },
        py::arg("label"),
        py::arg("onehot"),
        py::arg("batch_idx"),
        py::arg("num_classes"),
        "Convert a single label to one-hot encoding in a batch tensor");
    
    // cross_entropy_loss - Float
    nn_mod.def("cross_entropy_loss_f",
        [](const Matrixf& predictions, const Matrixf& targets, float epsilon) {
            return cross_entropy_loss(predictions, targets, epsilon);
        },
        py::arg("predictions"),
        py::arg("targets"),
        py::arg("epsilon") = 1e-7f,
        "Compute cross-entropy loss between predictions and targets");
    
    // cross_entropy_loss - Double
    nn_mod.def("cross_entropy_loss_d",
        [](const Matrixd& predictions, const Matrixd& targets, double epsilon) {
            return cross_entropy_loss(predictions, targets, epsilon);
        },
        py::arg("predictions"),
        py::arg("targets"),
        py::arg("epsilon") = 1e-7,
        "Compute cross-entropy loss between predictions and targets");
    
    // compute_accuracy - Float
    nn_mod.def("compute_accuracy_f",
        [](const Matrixf& predictions, const std::vector<uint8_t>& labels, size_t offset) {
            return compute_accuracy(predictions, labels, offset);
        },
        py::arg("predictions"),
        py::arg("labels"),
        py::arg("offset") = 0,
        "Compute classification accuracy");
    
    // compute_accuracy - Double
    nn_mod.def("compute_accuracy_d",
        [](const Matrixd& predictions, const std::vector<uint8_t>& labels, size_t offset) {
            return compute_accuracy(predictions, labels, offset);
        },
        py::arg("predictions"),
        py::arg("labels"),
        py::arg("offset") = 0,
        "Compute classification accuracy");
    
    // update_linear_layer - Float
    nn_mod.def("update_linear_layer_f",
        [](Linear<float>& layer, float lr) {
            update_linear_layer(layer, lr);
        },
        py::arg("layer"),
        py::arg("lr"),
        "Update linear layer weights using SGD");
    
    // update_linear_layer - Double
    nn_mod.def("update_linear_layer_d",
        [](Linear<double>& layer, double lr) {
            update_linear_layer(layer, lr);
        },
        py::arg("layer"),
        py::arg("lr"),
        "Update linear layer weights using SGD");
    
    // Constants and enums
    
    // Version info
    m.attr("__version__") = "1.6.0";
}
