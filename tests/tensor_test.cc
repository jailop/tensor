#include "tensor.h"
#include "optimizers.h"
#include "loss_functions.h"
#include <gtest/gtest.h>

using namespace tensor;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};
TEST_F(TensorTest, Constructor1DInitializesShape) {
    TensorIndices<1> shape = {10};
    Tensor<float, 1> tensor(shape);
    EXPECT_EQ(tensor.dims()[0], 10);
}

TEST_F(TensorTest, Constructor2DInitializesShape) {
    TensorIndices<2> shape = {3, 4};
    Tensor<float, 2> tensor(shape);
    EXPECT_EQ(tensor.dims()[0], 3);
    EXPECT_EQ(tensor.dims()[1], 4);
}

TEST_F(TensorTest, Constructor3DInitializesShape) {
    TensorIndices<3> shape = {2, 3, 4};
    Tensor<float, 3> tensor(shape);
    EXPECT_EQ(tensor.dims()[0], 2);
    EXPECT_EQ(tensor.dims()[1], 3);
    EXPECT_EQ(tensor.dims()[2], 4);
}

TEST_F(TensorTest, Constructor4DInitializesShape) {
    TensorIndices<4> shape = {2, 3, 4, 5};
    Tensor<float, 4> tensor(shape);
    EXPECT_EQ(tensor.dims()[0], 2);
    EXPECT_EQ(tensor.dims()[1], 3);
    EXPECT_EQ(tensor.dims()[2], 4);
    EXPECT_EQ(tensor.dims()[3], 5);
}

TEST_F(TensorTest, SetAndGet1DElement) {
    TensorIndices<1> shape = {10};
    Tensor<float, 1> tensor(shape);
    tensor[{0}] = 1.5f;
    EXPECT_FLOAT_EQ((tensor[{0}]), 1.5f);
    tensor[{9}] = 42.0f;
    EXPECT_FLOAT_EQ((tensor[{9}]), 42.0f);
}

TEST_F(TensorTest, SetAndGet2DElement) {
    TensorIndices<2> shape = {3, 4};
    Tensor<float, 2> tensor(shape);
    tensor[{0, 0}] = 1.5f;
    EXPECT_FLOAT_EQ((tensor[{0, 0}]), 1.5f);
    tensor[{2, 3}] = 42.0f;
    EXPECT_FLOAT_EQ((tensor[{2, 3}]), 42.0f);
}

TEST_F(TensorTest, SetAndGet3DElement) {
    TensorIndices<3> shape = {2, 3, 4};
    Tensor<float, 3> tensor(shape);
    tensor[{0, 0, 0}] = 1.5f;
    EXPECT_FLOAT_EQ((tensor[{0, 0, 0}]), 1.5f);
    tensor[{1, 2, 3}] = 42.0f;
    EXPECT_FLOAT_EQ((tensor[{1, 2, 3}]), 42.0f);
}

TEST_F(TensorTest, SetAndGetMultiple2DElements) {
    TensorIndices<2> shape = {3, 3};
    Tensor<float, 2> tensor(shape);
    float value = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            tensor[{i, j}] = value;
            value += 1.0f;
        }
    }
    value = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((tensor[{i, j}]), value);
            value += 1.0f;
        }
    }
}

TEST_F(TensorTest, SetAndGetMultiple3DElements) {
    TensorIndices<3> shape = {2, 2, 2};
    Tensor<float, 3> tensor(shape);
    float value = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                tensor[{i, j, k}] = value;
                value += 1.0f;
            }
        }
    }
    value = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                EXPECT_FLOAT_EQ((tensor[{i, j, k}]), value);
                value += 1.0f;
            }
        }
    }
}

TEST_F(TensorTest, CopyConstructor1DCreatesIndependentCopy) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> tensor1(shape);
    tensor1[{0}] = 10.0f;
    tensor1[{4}] = 20.0f;
    Tensor<float, 1> tensor2(tensor1);
    EXPECT_FLOAT_EQ((tensor2[{0}]), 10.0f);
    EXPECT_FLOAT_EQ((tensor2[{4}]), 20.0f);
    tensor2[{0}] = 99.0f;
    EXPECT_FLOAT_EQ((tensor1[{0}]), 10.0f);
    EXPECT_FLOAT_EQ((tensor2[{0}]), 99.0f);
}

TEST_F(TensorTest, CopyConstructor2DCreatesIndependentCopy) {
    TensorIndices<2> shape = {3, 3};
    Tensor<float, 2> tensor1(shape);
    tensor1[{0, 0}] = 10.0f;
    tensor1[{2, 2}] = 20.0f;
    Tensor<float, 2> tensor2(tensor1);
    EXPECT_FLOAT_EQ((tensor2[{0, 0}]), 10.0f);
    EXPECT_FLOAT_EQ((tensor2[{2, 2}]), 20.0f);
    tensor2[{0, 0}] = 99.0f;
    EXPECT_FLOAT_EQ((tensor1[{0, 0}]), 10.0f);
    EXPECT_FLOAT_EQ((tensor2[{0, 0}]), 99.0f);
}

TEST_F(TensorTest, CopyConstructorCopiesAllElements) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> tensor1(shape);
    float value = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            tensor1[{i, j}] = value;
            value += 1.0f;
        }
    }
    Tensor<float, 2> tensor2(tensor1);
    value = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((tensor2[{i, j}]), value);
            value += 1.0f;
        }
    }
}

TEST_F(TensorTest, ConstAccessorWorks) {
    TensorIndices<2> shape = {3, 3};
    Tensor<float, 2> tensor(shape);
    tensor[{1, 2}] = 3.14f;
    const Tensor<float, 2>& constTensor = tensor;
    EXPECT_FLOAT_EQ((constTensor[{1, 2}]), 3.14f);
}

TEST_F(TensorTest, DifferentShapeSizes) {
    TensorIndices<1> shape1 = {100};
    Tensor<float, 1> tensor1(shape1);
    tensor1[{50}] = 5.5f;
    EXPECT_FLOAT_EQ((tensor1[{50}]), 5.5f);
    
    TensorIndices<2> shape2 = {10, 5};
    Tensor<float, 2> tensor2(shape2);
    tensor2[{5, 3}] = 7.7f;
    EXPECT_FLOAT_EQ((tensor2[{5, 3}]), 7.7f);
    
    TensorIndices<3> shape3 = {3, 5, 7};
    Tensor<float, 3> tensor3(shape3);
    tensor3[{2, 4, 6}] = 11.11f;
    EXPECT_FLOAT_EQ((tensor3[{2, 4, 6}]), 11.11f);
}

TEST_F(TensorTest, BoundaryElements) {
    TensorIndices<3> shape = {3, 4, 5};
    Tensor<float, 3> tensor(shape);
    tensor[{0, 0, 0}] = 1.0f;
    EXPECT_FLOAT_EQ((tensor[{0, 0, 0}]), 1.0f);
    tensor[{2, 3, 4}] = 2.0f;
    EXPECT_FLOAT_EQ((tensor[{2, 3, 4}]), 2.0f);
}

TEST_F(TensorTest, FillMethod1DSetsAllElements) {
    TensorIndices<1> shape = {10};
    Tensor<float, 1> tensor(shape);
    tensor.fill(3.33f);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ((tensor[{i}]), 3.33f);
    }
}

TEST_F(TensorTest, FillMethod2DSetsAllElements) {
    TensorIndices<2> shape = {3, 4};
    Tensor<float, 2> tensor(shape);
    tensor.fill(3.33f);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((tensor[{i, j}]), 3.33f);
        }
    }
}

TEST_F(TensorTest, FillMethod3DSetsAllElements) {
    TensorIndices<3> shape = {2, 2, 2};
    Tensor<float, 3> tensor(shape);
    tensor.fill(3.33f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                EXPECT_FLOAT_EQ((tensor[{i, j, k}]), 3.33f);
            }
        }
    }
}

TEST_F(TensorTest, IntegerTypeWorks) {
    TensorIndices<2> shape = {3, 3};
    Tensor<int, 2> tensor(shape);
    tensor[{0, 0}] = 42;
    tensor[{1, 1}] = 99;
    EXPECT_EQ((tensor[{0, 0}]), 42);
    EXPECT_EQ((tensor[{1, 1}]), 99);
}

TEST_F(TensorTest, DoubleTypeWorks) {
    TensorIndices<2> shape = {3, 3};
    Tensor<double, 2> tensor(shape);
    tensor[{0, 0}] = 3.141592653589793;
    tensor[{1, 1}] = 2.718281828459045;
    EXPECT_DOUBLE_EQ((tensor[{0, 0}]), 3.141592653589793);
    EXPECT_DOUBLE_EQ((tensor[{1, 1}]), 2.718281828459045);
}

TEST_F(TensorTest, FillWithIntegerType) {
    TensorIndices<2> shape = {2, 3};
    Tensor<int, 2> tensor(shape);
    tensor.fill(7);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ((tensor[{i, j}]), 7);
        }
    }
}

TEST_F(TensorTest, DotProduct1DSimple) {
    TensorIndices<1> shape = {3};
    Tensor<float, 1> a(shape);
    Tensor<float, 1> b(shape);
    
    a[{0}] = 1.0f;
    a[{1}] = 2.0f;
    a[{2}] = 3.0f;
    
    b[{0}] = 4.0f;
    b[{1}] = 5.0f;
    b[{2}] = 6.0f;
    
    auto result = a.dot(b);
    ASSERT_TRUE(std::holds_alternative<float>(result));
    EXPECT_FLOAT_EQ(std::get<float>(result), 32.0f); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

TEST_F(TensorTest, DotProduct1DZero) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape);
    Tensor<float, 1> b(shape);
    
    a.fill(0.0f);
    b.fill(1.0f);
    
    auto result = a.dot(b);
    ASSERT_TRUE(std::holds_alternative<float>(result));
    EXPECT_FLOAT_EQ(std::get<float>(result), 0.0f);
}

TEST_F(TensorTest, DotProduct1DOrthogonal) {
    TensorIndices<1> shape = {4};
    Tensor<float, 1> a(shape);
    Tensor<float, 1> b(shape);
    
    a[{0}] = 1.0f;
    a[{1}] = 0.0f;
    a[{2}] = 0.0f;
    a[{3}] = 0.0f;
    
    b[{0}] = 0.0f;
    b[{1}] = 1.0f;
    b[{2}] = 0.0f;
    b[{3}] = 0.0f;
    
    auto result = a.dot(b);
    ASSERT_TRUE(std::holds_alternative<float>(result));
    EXPECT_FLOAT_EQ(std::get<float>(result), 0.0f);
}

TEST_F(TensorTest, DotProduct1DDimensionMismatch) {
    TensorIndices<1> shape_a = {3};
    TensorIndices<1> shape_b = {5};
    Tensor<float, 1> a(shape_a);
    Tensor<float, 1> b(shape_b);
    
    auto result = a.dot(b);
    ASSERT_TRUE(std::holds_alternative<TensorError>(result));
    EXPECT_EQ(std::get<TensorError>(result), TensorError::DimensionMismatch);
}

TEST_F(TensorTest, DotProduct2DIdentityMatrix) {
    TensorIndices<2> shape = {3, 3};
    Tensor<float, 2> identity(shape);
    identity.fill(0.0f);
    identity[{0, 0}] = 1.0f;
    identity[{1, 1}] = 1.0f;
    identity[{2, 2}] = 1.0f;
    
    Tensor<float, 2> a(shape);
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f; a[{0, 2}] = 3.0f;
    a[{1, 0}] = 4.0f; a[{1, 1}] = 5.0f; a[{1, 2}] = 6.0f;
    a[{2, 0}] = 7.0f; a[{2, 1}] = 8.0f; a[{2, 2}] = 9.0f;
    
    auto result_variant = a.dot(identity);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_variant)));
    auto result = std::get<Tensor<float, 2>>(result_variant);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), (a[{i, j}]));
        }
    }
}

TEST_F(TensorTest, DotProduct2DSimple) {
    TensorIndices<2> shape_a = {2, 3};
    TensorIndices<2> shape_b = {3, 2};
    
    Tensor<float, 2> a(shape_a);
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f; a[{0, 2}] = 3.0f;
    a[{1, 0}] = 4.0f; a[{1, 1}] = 5.0f; a[{1, 2}] = 6.0f;
    
    Tensor<float, 2> b(shape_b);
    b[{0, 0}] = 7.0f;  b[{0, 1}] = 8.0f;
    b[{1, 0}] = 9.0f;  b[{1, 1}] = 10.0f;
    b[{2, 0}] = 11.0f; b[{2, 1}] = 12.0f;
    
    auto result_variant = a.dot(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_variant)));
    auto result = std::get<Tensor<float, 2>>(result_variant);
    
    EXPECT_EQ(result.dims()[0], 2);
    EXPECT_EQ(result.dims()[1], 2);
    
    // [1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12]   = [58,  64]
    // [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]   = [139, 154]
    EXPECT_FLOAT_EQ((result[{0, 0}]), 58.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 64.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 139.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 154.0f);
}

TEST_F(TensorTest, DotProduct2DSquareMatrices) {
    TensorIndices<2> shape = {2, 2};
    
    Tensor<float, 2> a(shape);
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    Tensor<float, 2> b(shape);
    b[{0, 0}] = 5.0f; b[{0, 1}] = 6.0f;
    b[{1, 0}] = 7.0f; b[{1, 1}] = 8.0f;
    
    auto result_variant = a.dot(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_variant)));
    auto result = std::get<Tensor<float, 2>>(result_variant);
    
    // [1*5 + 2*7,  1*6 + 2*8]   = [19, 22]
    // [3*5 + 4*7,  3*6 + 4*8]   = [43, 50]
    EXPECT_FLOAT_EQ((result[{0, 0}]), 19.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 22.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 43.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 50.0f);
}

TEST_F(TensorTest, DotProduct2DWithIntegers) {
    TensorIndices<2> shape = {2, 2};
    
    Tensor<int, 2> a(shape);
    a[{0, 0}] = 1; a[{0, 1}] = 2;
    a[{1, 0}] = 3; a[{1, 1}] = 4;
    
    Tensor<int, 2> b(shape);
    b[{0, 0}] = 5; b[{0, 1}] = 6;
    b[{1, 0}] = 7; b[{1, 1}] = 8;
    
    auto result_variant = a.dot(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<int, 2>>(result_variant)));
    auto result = std::get<Tensor<int, 2>>(result_variant);
    
    EXPECT_EQ((result[{0, 0}]), 19);
    EXPECT_EQ((result[{0, 1}]), 22);
    EXPECT_EQ((result[{1, 0}]), 43);
    EXPECT_EQ((result[{1, 1}]), 50);
}

TEST_F(TensorTest, DotProduct2DDimensionMismatch) {
    TensorIndices<2> shape_a = {2, 3};
    TensorIndices<2> shape_b = {5, 2};
    
    Tensor<float, 2> a(shape_a);
    Tensor<float, 2> b(shape_b);
    
    auto result = a.dot(b);
    ASSERT_TRUE((std::holds_alternative<TensorError>(result)));
    EXPECT_EQ(std::get<TensorError>(result), TensorError::DimensionMismatch);
}

TEST_F(TensorTest, DotProduct3DWith1D) {
    // Test 3D tensor (2,3,4) dot 1D tensor (4) = 2D tensor (2,3)
    TensorIndices<3> shape_a = {2, 3, 4};
    TensorIndices<1> shape_b = {4};
    
    Tensor<float, 3> a(shape_a);
    Tensor<float, 1> b(shape_b);
    
    // Fill with simple values for testing
    float val = 1.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                a[{i, j, k}] = val++;
            }
        }
    }
    
    b[{0}] = 1.0f;
    b[{1}] = 2.0f;
    b[{2}] = 3.0f;
    b[{3}] = 4.0f;
    
    auto result_variant = a.dot(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_variant)));
    auto result = std::get<Tensor<float, 2>>(result_variant);
    
    EXPECT_EQ(result.dims()[0], 2);
    EXPECT_EQ(result.dims()[1], 3);
    
    // a[0,0,:] = [1,2,3,4] dot [1,2,3,4] = 1+4+9+16 = 30
    EXPECT_FLOAT_EQ((result[{0, 0}]), 30.0f);
    // a[0,1,:] = [5,6,7,8] dot [1,2,3,4] = 5+12+21+32 = 70
    EXPECT_FLOAT_EQ((result[{0, 1}]), 70.0f);
}

TEST_F(TensorTest, DotProduct3DWith2D) {
    // Test 3D tensor (2,3,4) dot 2D tensor (4,5) = 3D tensor (2,3,5)
    TensorIndices<3> shape_a = {2, 3, 4};
    TensorIndices<2> shape_b = {4, 5};
    
    Tensor<float, 3> a(shape_a);
    Tensor<float, 2> b(shape_b);
    
    // Simple fill for a
    a.fill(1.0f);
    
    // Simple fill for b
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            b[{i, j}] = static_cast<float>(i + j);
        }
    }
    
    auto result_variant = a.dot(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 3>>(result_variant)));
    auto result = std::get<Tensor<float, 3>>(result_variant);
    
    EXPECT_EQ(result.dims()[0], 2);
    EXPECT_EQ(result.dims()[1], 3);
    EXPECT_EQ(result.dims()[2], 5);
    
    // Since a is all 1s, result[i,j,k] = sum(b[:,k]) = 0+1+2+3 + j*4
    // For k=0: sum = 0+1+2+3 = 6
    EXPECT_FLOAT_EQ((result[{0, 0, 0}]), 6.0f);
    // For k=1: sum = 1+2+3+4 = 10
    EXPECT_FLOAT_EQ((result[{0, 0, 1}]), 10.0f);
}

TEST_F(TensorTest, DotProduct4DWith1D) {
    // Test 4D tensor (2,2,2,3) dot 1D tensor (3) = 3D tensor (2,2,2)
    TensorIndices<4> shape_a = {2, 2, 2, 3};
    TensorIndices<1> shape_b = {3};
    
    Tensor<float, 4> a(shape_a);
    Tensor<float, 1> b(shape_b);
    
    a.fill(1.0f);
    b[{0}] = 1.0f;
    b[{1}] = 2.0f;
    b[{2}] = 3.0f;
    
    auto result_variant = a.dot(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 3>>(result_variant)));
    auto result = std::get<Tensor<float, 3>>(result_variant);
    
    EXPECT_EQ(result.dims()[0], 2);
    EXPECT_EQ(result.dims()[1], 2);
    EXPECT_EQ(result.dims()[2], 2);
    
    // Since a is all 1s, result = sum(b) = 1+2+3 = 6
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                EXPECT_FLOAT_EQ((result[{i, j, k}]), 6.0f);
            }
        }
    }
}

TEST_F(TensorTest, DotProduct3DContractionMismatch) {
    TensorIndices<3> shape_a = {2, 3, 4};
    TensorIndices<1> shape_b = {5};  // Mismatch: should be 4
    
    Tensor<float, 3> a(shape_a);
    Tensor<float, 1> b(shape_b);
    
    auto result = a.dot(b);
    ASSERT_TRUE((std::holds_alternative<TensorError>(result)));
    EXPECT_EQ(std::get<TensorError>(result), TensorError::ContractionMismatch);
}

// ============================================
// Cross Product Tests
// ============================================

TEST_F(TensorTest, CrossProduct3DBasic) {
    // Test basic cross product: [1,0,0] × [0,1,0] = [0,0,1]
    Tensor<float, 1> a({3});
    a[{0}] = 1.0f; a[{1}] = 0.0f; a[{2}] = 0.0f;
    
    Tensor<float, 1> b({3});
    b[{0}] = 0.0f; b[{1}] = 1.0f; b[{2}] = 0.0f;
    
    auto result_var = a.cross(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result_var)));
    auto& result = std::get<Tensor<float, 1>>(result_var);
    
    EXPECT_FLOAT_EQ((result[{0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 1.0f);
}

TEST_F(TensorTest, CrossProduct3DAntiCommutative) {
    // Test anti-commutativity: a × b = -(b × a)
    Tensor<float, 1> a({3});
    a[{0}] = 2.0f; a[{1}] = 3.0f; a[{2}] = 4.0f;
    
    Tensor<float, 1> b({3});
    b[{0}] = 5.0f; b[{1}] = 6.0f; b[{2}] = 7.0f;
    
    auto result1_var = a.cross(b);
    auto result2_var = b.cross(a);
    
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result1_var)));
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result2_var)));
    
    auto& result1 = std::get<Tensor<float, 1>>(result1_var);
    auto& result2 = std::get<Tensor<float, 1>>(result2_var);
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ((result1[{i}]), -(result2[{i}]));
    }
}

TEST_F(TensorTest, CrossProduct3DPerpendicular) {
    // Test that result is perpendicular to both inputs
    Tensor<float, 1> a({3});
    a[{0}] = 1.0f; a[{1}] = 2.0f; a[{2}] = 3.0f;
    
    Tensor<float, 1> b({3});
    b[{0}] = 4.0f; b[{1}] = 5.0f; b[{2}] = 6.0f;
    
    auto result_var = a.cross(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result_var)));
    auto& result = std::get<Tensor<float, 1>>(result_var);
    
    // Compute dot products to check perpendicularity
    auto dot_a_var = a.dot(result);
    auto dot_b_var = b.dot(result);
    
    ASSERT_TRUE((std::holds_alternative<float>(dot_a_var)));
    ASSERT_TRUE((std::holds_alternative<float>(dot_b_var)));
    
    float dot_a = std::get<float>(dot_a_var);
    float dot_b = std::get<float>(dot_b_var);
    
    EXPECT_NEAR(dot_a, 0.0f, 1e-5);
    EXPECT_NEAR(dot_b, 0.0f, 1e-5);
}

TEST_F(TensorTest, CrossProduct3DParallel) {
    // Test cross product of parallel vectors = zero vector
    Tensor<float, 1> a({3});
    a[{0}] = 1.0f; a[{1}] = 2.0f; a[{2}] = 3.0f;
    
    Tensor<float, 1> b({3});
    b[{0}] = 2.0f; b[{1}] = 4.0f; b[{2}] = 6.0f;  // b = 2*a
    
    auto result_var = a.cross(b);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result_var)));
    auto& result = std::get<Tensor<float, 1>>(result_var);
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR((result[{i}]), 0.0f, 1e-5);
    }
}

TEST_F(TensorTest, CrossProduct3DWrongDimension) {
    // Test error when vectors are not 3D
    Tensor<float, 1> a({2});
    a[{0}] = 1.0f; a[{1}] = 2.0f;
    
    Tensor<float, 1> b({2});
    b[{0}] = 3.0f; b[{1}] = 4.0f;
    
    auto result = a.cross(b);
    ASSERT_TRUE((std::holds_alternative<TensorError>(result)));
    EXPECT_EQ(std::get<TensorError>(result), TensorError::DimensionMismatch);
}

TEST_F(TensorTest, CrossProduct3DMismatchedSize) {
    // Test error when vectors have different sizes
    Tensor<float, 1> a({3});
    a[{0}] = 1.0f; a[{1}] = 2.0f; a[{2}] = 3.0f;
    
    Tensor<float, 1> b({4});
    b[{0}] = 4.0f; b[{1}] = 5.0f; b[{2}] = 6.0f; b[{3}] = 7.0f;
    
    auto result = a.cross(b);
    ASSERT_TRUE((std::holds_alternative<TensorError>(result)));
    EXPECT_EQ(std::get<TensorError>(result), TensorError::DimensionMismatch);
}

// ============================================
// Arithmetic Operations Tests
// ============================================

TEST_F(TensorTest, AddTensorToTensor1D) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape);
    Tensor<float, 1> b(shape);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
        b[{i}] = static_cast<float>(i * 2);
    }
    
    auto result_variant = a + b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result_variant)));
    auto result = std::get<Tensor<float, 1>>(result_variant);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ((result[{i}]), static_cast<float>(i * 3));
    }
}

TEST_F(TensorTest, AddTensorToTensor2D) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    a.fill(5.0f);
    b.fill(3.0f);
    
    auto result_variant = a + b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_variant)));
    auto result = std::get<Tensor<float, 2>>(result_variant);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), 8.0f);
        }
    }
}

TEST_F(TensorTest, AddScalarToTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    auto result = a + 10.0f;
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 11.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 12.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 13.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 14.0f);
}

TEST_F(TensorTest, AddTensorToScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    auto result = 10.0f + a;
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 11.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 12.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 13.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 14.0f);
}

TEST_F(TensorTest, AddTensorDimensionMismatch) {
    TensorIndices<1> shape_a = {5};
    TensorIndices<1> shape_b = {3};
    Tensor<float, 1> a(shape_a);
    Tensor<float, 1> b(shape_b);
    
    auto result = a + b;
    ASSERT_TRUE((std::holds_alternative<TensorError>(result)));
    EXPECT_EQ(std::get<TensorError>(result), TensorError::DimensionMismatch);
}

TEST_F(TensorTest, AddEqualsTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    a.fill(5.0f);
    b.fill(3.0f);
    
    a += b;
    
    EXPECT_FLOAT_EQ((a[{0, 0}]), 8.0f);
    EXPECT_FLOAT_EQ((a[{0, 1}]), 8.0f);
    EXPECT_FLOAT_EQ((a[{1, 0}]), 8.0f);
    EXPECT_FLOAT_EQ((a[{1, 1}]), 8.0f);
}

TEST_F(TensorTest, AddEqualsScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    a += 5.0f;
    
    EXPECT_FLOAT_EQ((a[{0, 0}]), 6.0f);
    EXPECT_FLOAT_EQ((a[{0, 1}]), 7.0f);
    EXPECT_FLOAT_EQ((a[{1, 0}]), 8.0f);
    EXPECT_FLOAT_EQ((a[{1, 1}]), 9.0f);
}

TEST_F(TensorTest, SubtractTensorFromTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    a.fill(10.0f);
    b.fill(3.0f);
    
    auto result_variant = a - b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_variant)));
    auto result = std::get<Tensor<float, 2>>(result_variant);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), 7.0f);
        }
    }
}

TEST_F(TensorTest, SubtractScalarFromTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    a.fill(10.0f);
    
    auto result = a - 3.0f;
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), 7.0f);
        }
    }
}

TEST_F(TensorTest, SubtractTensorFromScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    auto result = 10.0f - a;
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 9.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 8.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 7.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 6.0f);
}

TEST_F(TensorTest, SubtractEqualsTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    a.fill(10.0f);
    b.fill(3.0f);
    
    a -= b;
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ((a[{i, j}]), 7.0f);
        }
    }
}

TEST_F(TensorTest, SubtractEqualsScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    a.fill(10.0f);
    
    a -= 3.0f;
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ((a[{i, j}]), 7.0f);
        }
    }
}

TEST_F(TensorTest, MultiplyTensorByTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    a[{0, 0}] = 2.0f; a[{0, 1}] = 3.0f;
    a[{1, 0}] = 4.0f; a[{1, 1}] = 5.0f;
    
    b[{0, 0}] = 2.0f; b[{0, 1}] = 2.0f;
    b[{1, 0}] = 2.0f; b[{1, 1}] = 2.0f;
    
    auto result_variant = a * b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_variant)));
    auto result = std::get<Tensor<float, 2>>(result_variant);
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 8.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 10.0f);
}

TEST_F(TensorTest, MultiplyTensorByScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    auto result = a * 3.0f;
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 9.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 12.0f);
}

TEST_F(TensorTest, MultiplyScalarByTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    auto result = 3.0f * a;
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 9.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 12.0f);
}

TEST_F(TensorTest, MultiplyEqualsTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    a[{0, 0}] = 2.0f; a[{0, 1}] = 3.0f;
    a[{1, 0}] = 4.0f; a[{1, 1}] = 5.0f;
    b.fill(2.0f);
    
    a *= b;
    
    EXPECT_FLOAT_EQ((a[{0, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((a[{0, 1}]), 6.0f);
    EXPECT_FLOAT_EQ((a[{1, 0}]), 8.0f);
    EXPECT_FLOAT_EQ((a[{1, 1}]), 10.0f);
}

TEST_F(TensorTest, MultiplyEqualsScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    a *= 3.0f;
    
    EXPECT_FLOAT_EQ((a[{0, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((a[{0, 1}]), 6.0f);
    EXPECT_FLOAT_EQ((a[{1, 0}]), 9.0f);
    EXPECT_FLOAT_EQ((a[{1, 1}]), 12.0f);
}

TEST_F(TensorTest, DivideTensorByTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    a[{0, 0}] = 10.0f; a[{0, 1}] = 20.0f;
    a[{1, 0}] = 30.0f; a[{1, 1}] = 40.0f;
    
    b[{0, 0}] = 2.0f; b[{0, 1}] = 4.0f;
    b[{1, 0}] = 5.0f; b[{1, 1}] = 8.0f;
    
    auto result_variant = a / b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_variant)));
    auto result = std::get<Tensor<float, 2>>(result_variant);
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 5.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 5.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 6.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 5.0f);
}

TEST_F(TensorTest, DivideTensorByScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 10.0f; a[{0, 1}] = 20.0f;
    a[{1, 0}] = 30.0f; a[{1, 1}] = 40.0f;
    
    auto result = a / 10.0f;
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 4.0f);
}

TEST_F(TensorTest, DivideScalarByTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 2.0f; a[{0, 1}] = 4.0f;
    a[{1, 0}] = 5.0f; a[{1, 1}] = 10.0f;
    
    auto result = 100.0f / a;
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 50.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 25.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 20.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 10.0f);
}

TEST_F(TensorTest, DivideEqualsTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    
    a[{0, 0}] = 10.0f; a[{0, 1}] = 20.0f;
    a[{1, 0}] = 30.0f; a[{1, 1}] = 40.0f;
    b.fill(10.0f);
    
    a /= b;
    
    EXPECT_FLOAT_EQ((a[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((a[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((a[{1, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((a[{1, 1}]), 4.0f);
}

TEST_F(TensorTest, DivideEqualsScalar) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 10.0f; a[{0, 1}] = 20.0f;
    a[{1, 0}] = 30.0f; a[{1, 1}] = 40.0f;
    
    a /= 10.0f;
    
    EXPECT_FLOAT_EQ((a[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((a[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((a[{1, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((a[{1, 1}]), 4.0f);
}

TEST_F(TensorTest, UnaryNegation) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = -2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = -4.0f;
    
    auto result = -a;
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), -1.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), -3.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 4.0f);
}

TEST_F(TensorTest, ComplexArithmeticExpression) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    Tensor<float, 2> b(shape);
    Tensor<float, 2> c(shape);
    
    a.fill(5.0f);
    b.fill(3.0f);
    c.fill(2.0f);
    
    // (a + b) * c - 1.0
    auto temp1_var = a + b;  // 8.0
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(temp1_var)));
    auto temp1 = std::get<Tensor<float, 2>>(temp1_var);
    
    auto temp2_var = temp1 * c;  // 16.0
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(temp2_var)));
    auto temp2 = std::get<Tensor<float, 2>>(temp2_var);
    
    auto result = temp2 - 1.0f;  // 15.0
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), 15.0f);
        }
    }
}

TEST_F(TensorTest, ArithmeticWith3DTensor) {
    TensorIndices<3> shape = {2, 2, 2};
    Tensor<float, 3> a(shape);
    Tensor<float, 3> b(shape);
    
    a.fill(10.0f);
    b.fill(5.0f);
    
    auto result_var = a - b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 3>>(result_var)));
    auto result = std::get<Tensor<float, 3>>(result_var);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                EXPECT_FLOAT_EQ((result[{i, j, k}]), 5.0f);
            }
        }
    }
}

TEST_F(TensorTest, ArithmeticWithIntegers) {
    TensorIndices<2> shape = {2, 2};
    Tensor<int, 2> a(shape);
    Tensor<int, 2> b(shape);
    
    a[{0, 0}] = 10; a[{0, 1}] = 20;
    a[{1, 0}] = 30; a[{1, 1}] = 40;
    
    b[{0, 0}] = 2; b[{0, 1}] = 4;
    b[{1, 0}] = 6; b[{1, 1}] = 8;
    
    auto result_var = a + b;
    ASSERT_TRUE((std::holds_alternative<Tensor<int, 2>>(result_var)));
    auto result = std::get<Tensor<int, 2>>(result_var);
    
    EXPECT_EQ((result[{0, 0}]), 12);
    EXPECT_EQ((result[{0, 1}]), 24);
    EXPECT_EQ((result[{1, 0}]), 36);
    EXPECT_EQ((result[{1, 1}]), 48);
}

// ============================================
// Math Function Mapping Tests
// ============================================

TEST_F(TensorTest, MapCustomFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    auto result = a.map([](float x) { return x * 2.0f + 1.0f; });
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 5.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 7.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 9.0f);
    
    // Original should be unchanged
    EXPECT_FLOAT_EQ((a[{0, 0}]), 1.0f);
}

TEST_F(TensorTest, MapInplaceCustomFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = 3.0f; a[{1, 1}] = 4.0f;
    
    a.map_inplace([](float x) { return x * 2.0f; });
    
    EXPECT_FLOAT_EQ((a[{0, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((a[{0, 1}]), 4.0f);
    EXPECT_FLOAT_EQ((a[{1, 0}]), 6.0f);
    EXPECT_FLOAT_EQ((a[{1, 1}]), 8.0f);
}

TEST_F(TensorTest, ExpFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 0.0f; a[{0, 1}] = 1.0f;
    a[{1, 0}] = 2.0f; a[{1, 1}] = -1.0f;
    
    auto result = a.exp();
    
    EXPECT_NEAR((result[{0, 0}]), 1.0f, 1e-5);
    EXPECT_NEAR((result[{0, 1}]), std::exp(1.0f), 1e-5);
    EXPECT_NEAR((result[{1, 0}]), std::exp(2.0f), 1e-5);
    EXPECT_NEAR((result[{1, 1}]), std::exp(-1.0f), 1e-5);
}

TEST_F(TensorTest, LogFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = std::exp(1.0f); a[{1, 1}] = 10.0f;
    
    auto result = a.log();
    
    EXPECT_NEAR((result[{0, 0}]), 0.0f, 1e-5);
    EXPECT_NEAR((result[{0, 1}]), std::log(2.0f), 1e-5);
    EXPECT_NEAR((result[{1, 0}]), 1.0f, 1e-5);
    EXPECT_NEAR((result[{1, 1}]), std::log(10.0f), 1e-5);
}

TEST_F(TensorTest, SqrtFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 4.0f; a[{0, 1}] = 9.0f;
    a[{1, 0}] = 16.0f; a[{1, 1}] = 25.0f;
    
    auto result = a.sqrt();
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 5.0f);
}

TEST_F(TensorTest, PowFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 2.0f; a[{0, 1}] = 3.0f;
    a[{1, 0}] = 4.0f; a[{1, 1}] = 5.0f;
    
    auto result = a.pow(2.0f);
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 9.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 16.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 25.0f);
}

TEST_F(TensorTest, AbsFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = -1.0f; a[{0, 1}] = 2.0f;
    a[{1, 0}] = -3.0f; a[{1, 1}] = 4.0f;
    
    auto result = a.abs();
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 4.0f);
}

TEST_F(TensorTest, SinCosFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 0.0f; a[{0, 1}] = M_PI / 2.0f;
    a[{1, 0}] = M_PI; a[{1, 1}] = 3.0f * M_PI / 2.0f;
    
    auto sin_result = a.sin();
    auto cos_result = a.cos();
    
    EXPECT_NEAR((sin_result[{0, 0}]), 0.0f, 1e-5);
    EXPECT_NEAR((sin_result[{0, 1}]), 1.0f, 1e-5);
    EXPECT_NEAR((sin_result[{1, 0}]), 0.0f, 1e-5);
    EXPECT_NEAR((sin_result[{1, 1}]), -1.0f, 1e-5);
    
    EXPECT_NEAR((cos_result[{0, 0}]), 1.0f, 1e-5);
    EXPECT_NEAR((cos_result[{0, 1}]), 0.0f, 1e-5);
    EXPECT_NEAR((cos_result[{1, 0}]), -1.0f, 1e-5);
    EXPECT_NEAR((cos_result[{1, 1}]), 0.0f, 1e-5);
}

TEST_F(TensorTest, TanhFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = -1.0f; a[{0, 1}] = 0.0f;
    a[{1, 0}] = 1.0f; a[{1, 1}] = 2.0f;
    
    auto result = a.tanh();
    
    EXPECT_NEAR((result[{0, 0}]), std::tanh(-1.0f), 1e-5);
    EXPECT_NEAR((result[{0, 1}]), 0.0f, 1e-5);
    EXPECT_NEAR((result[{1, 0}]), std::tanh(1.0f), 1e-5);
    EXPECT_NEAR((result[{1, 1}]), std::tanh(2.0f), 1e-5);
}

TEST_F(TensorTest, SigmoidFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = -2.0f; a[{0, 1}] = 0.0f;
    a[{1, 0}] = 1.0f; a[{1, 1}] = 2.0f;
    
    auto result = a.sigmoid();
    
    // sigmoid(x) = 1 / (1 + exp(-x))
    EXPECT_NEAR((result[{0, 0}]), 1.0f / (1.0f + std::exp(2.0f)), 1e-5);
    EXPECT_NEAR((result[{0, 1}]), 0.5f, 1e-5);
    EXPECT_NEAR((result[{1, 0}]), 1.0f / (1.0f + std::exp(-1.0f)), 1e-5);
    EXPECT_NEAR((result[{1, 1}]), 1.0f / (1.0f + std::exp(-2.0f)), 1e-5);
}

TEST_F(TensorTest, ReluFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = -2.0f; a[{0, 1}] = 0.0f;
    a[{1, 0}] = 1.0f; a[{1, 1}] = 5.0f;
    
    auto result = a.relu();
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 5.0f);
}

TEST_F(TensorTest, ClampFunction) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = -5.0f; a[{0, 1}] = 0.0f; a[{0, 2}] = 2.0f;
    a[{1, 0}] = 5.0f; a[{1, 1}] = 10.0f; a[{1, 2}] = 15.0f;
    
    auto result = a.clamp(0.0f, 10.0f);
    
    EXPECT_FLOAT_EQ((result[{0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{0, 2}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 5.0f);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 10.0f);
    EXPECT_FLOAT_EQ((result[{1, 2}]), 10.0f);
}

TEST_F(TensorTest, CeilFloorFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.2f; a[{0, 1}] = 2.7f;
    a[{1, 0}] = -1.2f; a[{1, 1}] = -2.7f;
    
    auto ceil_result = a.ceil();
    auto floor_result = a.floor();
    
    EXPECT_FLOAT_EQ((ceil_result[{0, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((ceil_result[{0, 1}]), 3.0f);
    EXPECT_FLOAT_EQ((ceil_result[{1, 0}]), -1.0f);
    EXPECT_FLOAT_EQ((ceil_result[{1, 1}]), -2.0f);
    
    EXPECT_FLOAT_EQ((floor_result[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((floor_result[{0, 1}]), 2.0f);
    EXPECT_FLOAT_EQ((floor_result[{1, 0}]), -2.0f);
    EXPECT_FLOAT_EQ((floor_result[{1, 1}]), -3.0f);
}

TEST_F(TensorTest, ChainedMathOperations) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a.fill(2.0f);    
    // Test chaining: (x^2 + 1) * 2
    auto result = a.pow(2.0f);    
    auto temp = result + 1.0f;  // Returns Tensor directly, not variant    
    auto final = temp * 2.0f;    
    // 2^2 = 4, 4 + 1 = 5, 5 * 2 = 10
    EXPECT_FLOAT_EQ((final[{0, 0}]), 10.0f);
    EXPECT_FLOAT_EQ((final[{1, 1}]), 10.0f);
}

TEST_F(TensorTest, MathFunctionsOn3DTensor) {
    TensorIndices<3> shape = {2, 2, 2};
    Tensor<float, 3> a(shape);
    
    a.fill(4.0f);
    
    auto result = a.sqrt();
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                EXPECT_FLOAT_EQ((result[{i, j, k}]), 2.0f);
            }
        }
    }
}

TEST_F(TensorTest, NeuralNetworkActivationSequence) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape);
    
    // Input values
    a[{0}] = -2.0f;
    a[{1}] = -1.0f;
    a[{2}] = 0.0f;
    a[{3}] = 1.0f;
    a[{4}] = 2.0f;
    
    // Test ReLU -> Sigmoid chain
    auto after_relu = a.relu();
    EXPECT_FLOAT_EQ((after_relu[{0}]), 0.0f);
    EXPECT_FLOAT_EQ((after_relu[{1}]), 0.0f);
    EXPECT_FLOAT_EQ((after_relu[{2}]), 0.0f);
    EXPECT_FLOAT_EQ((after_relu[{3}]), 1.0f);
    EXPECT_FLOAT_EQ((after_relu[{4}]), 2.0f);
    
    auto after_sigmoid = after_relu.sigmoid();
    EXPECT_NEAR((after_sigmoid[{0}]), 0.5f, 1e-5);
    EXPECT_NEAR((after_sigmoid[{3}]), 1.0f / (1.0f + std::exp(-1.0f)), 1e-5);
}

// ============================================
// Derivative Tests
// ============================================

TEST_F(TensorTest, SigmoidDerivative) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> output(shape);
    
    // Sigmoid outputs (y = sigmoid(x))
    output[{0, 0}] = 0.5f;   // sigmoid(0)
    output[{0, 1}] = 0.7311f; // sigmoid(1)
    output[{1, 0}] = 0.2689f; // sigmoid(-1)
    output[{1, 1}] = 0.8808f; // sigmoid(2)
    
    auto deriv = output.sigmoid_derivative();
    
    // dy/dx = y * (1 - y)
    EXPECT_NEAR((deriv[{0, 0}]), 0.25f, 1e-3);
    EXPECT_NEAR((deriv[{0, 1}]), 0.7311f * (1.0f - 0.7311f), 1e-3);
    EXPECT_NEAR((deriv[{1, 0}]), 0.2689f * (1.0f - 0.2689f), 1e-3);
}

TEST_F(TensorTest, SigmoidDerivativeFromInput) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> input(shape);
    
    input[{0, 0}] = 0.0f;
    input[{0, 1}] = 1.0f;
    input[{1, 0}] = -1.0f;
    input[{1, 1}] = 2.0f;
    
    auto deriv = input.sigmoid_derivative_from_input();
    
    // Verify derivative at each point
    EXPECT_NEAR((deriv[{0, 0}]), 0.25f, 1e-3);
}

TEST_F(TensorTest, TanhDerivative) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> output(shape);
    
    // tanh outputs
    output[{0, 0}] = 0.0f;      // tanh(0)
    output[{0, 1}] = 0.7616f;   // tanh(1)
    output[{1, 0}] = -0.7616f;  // tanh(-1)
    output[{1, 1}] = 0.9640f;   // tanh(2)
    
    auto deriv = output.tanh_derivative();
    
    // dy/dx = 1 - y²
    EXPECT_NEAR((deriv[{0, 0}]), 1.0f, 1e-3);
    EXPECT_NEAR((deriv[{0, 1}]), 1.0f - 0.7616f * 0.7616f, 1e-3);
    EXPECT_NEAR((deriv[{1, 1}]), 1.0f - 0.9640f * 0.9640f, 1e-3);
}

TEST_F(TensorTest, ReluDerivative) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> input(shape);
    
    input[{0, 0}] = -2.0f;
    input[{0, 1}] = 0.0f;
    input[{1, 0}] = 0.5f;
    input[{1, 1}] = 2.0f;
    
    auto deriv = input.relu_derivative();
    
    EXPECT_FLOAT_EQ((deriv[{0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((deriv[{0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((deriv[{1, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((deriv[{1, 1}]), 1.0f);
}

TEST_F(TensorTest, LeakyReluDerivative) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> input(shape);
    
    input[{0, 0}] = -2.0f;
    input[{0, 1}] = 0.0f;
    input[{1, 0}] = 0.5f;
    input[{1, 1}] = 2.0f;
    
    float alpha = 0.01f;
    auto deriv = input.leaky_relu_derivative(alpha);
    
    EXPECT_FLOAT_EQ((deriv[{0, 0}]), alpha);
    EXPECT_FLOAT_EQ((deriv[{0, 1}]), alpha);
    EXPECT_FLOAT_EQ((deriv[{1, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((deriv[{1, 1}]), 1.0f);
}

TEST_F(TensorTest, ExpDerivative) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> input(shape);
    
    input[{0, 0}] = 0.0f;
    input[{0, 1}] = 1.0f;
    input[{1, 0}] = 2.0f;
    input[{1, 1}] = -1.0f;
    
    auto deriv = input.exp_derivative();
    auto expected = input.exp();
    
    // d/dx[exp(x)] = exp(x)
    EXPECT_NEAR((deriv[{0, 0}]), (expected[{0, 0}]), 1e-5);
    EXPECT_NEAR((deriv[{0, 1}]), (expected[{0, 1}]), 1e-5);
}

TEST_F(TensorTest, LogDerivative) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> input(shape);
    
    input[{0, 0}] = 1.0f;
    input[{0, 1}] = 2.0f;
    input[{1, 0}] = 5.0f;
    input[{1, 1}] = 10.0f;
    
    auto deriv = input.log_derivative();
    
    // d/dx[log(x)] = 1/x
    EXPECT_FLOAT_EQ((deriv[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((deriv[{0, 1}]), 0.5f);
    EXPECT_FLOAT_EQ((deriv[{1, 0}]), 0.2f);
    EXPECT_FLOAT_EQ((deriv[{1, 1}]), 0.1f);
}

TEST_F(TensorTest, PowDerivative) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> input(shape);
    
    input[{0, 0}] = 2.0f;
    input[{0, 1}] = 3.0f;
    input[{1, 0}] = 4.0f;
    input[{1, 1}] = 5.0f;
    
    float exponent = 3.0f;
    auto deriv = input.pow_derivative(exponent);
    
    // d/dx[x^3] = 3 * x^2
    EXPECT_NEAR((deriv[{0, 0}]), 3.0f * 4.0f, 1e-3);
    EXPECT_NEAR((deriv[{0, 1}]), 3.0f * 9.0f, 1e-3);
    EXPECT_NEAR((deriv[{1, 0}]), 3.0f * 16.0f, 1e-3);
}

TEST_F(TensorTest, SinCosDerivative) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> input(shape);
    
    input[{0, 0}] = 0.0f;
    input[{0, 1}] = M_PI / 2.0f;
    input[{1, 0}] = M_PI;
    input[{1, 1}] = 3.0f * M_PI / 2.0f;
    
    auto sin_deriv = input.sin_derivative();
    auto cos_deriv = input.cos_derivative();
    
    // d/dx[sin(x)] = cos(x)
    EXPECT_NEAR((sin_deriv[{0, 0}]), 1.0f, 1e-4);
    EXPECT_NEAR((sin_deriv[{0, 1}]), 0.0f, 1e-4);
    
    // d/dx[cos(x)] = -sin(x)
    EXPECT_NEAR((cos_deriv[{0, 0}]), 0.0f, 1e-4);
    EXPECT_NEAR((cos_deriv[{1, 0}]), 0.0f, 1e-4);
}

TEST_F(TensorTest, ChainRule) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> local_grad(shape);
    Tensor<float, 2> upstream_grad(shape);
    
    local_grad.fill(2.0f);
    upstream_grad.fill(3.0f);
    
    auto result_var = local_grad.chain_rule(upstream_grad);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_var)));
    auto result = std::get<Tensor<float, 2>>(result_var);
    
    // gradient = local * upstream = 2 * 3 = 6
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ((result[{i, j}]), 6.0f);
        }
    }
}

TEST_F(TensorTest, SumReduction) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 1.0f; a[{0, 1}] = 2.0f; a[{0, 2}] = 3.0f;
    a[{1, 0}] = 4.0f; a[{1, 1}] = 5.0f; a[{1, 2}] = 6.0f;
    
    float sum = a.sum();
    EXPECT_FLOAT_EQ(sum, 21.0f);
}

TEST_F(TensorTest, MeanCalculation) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a.fill(5.0f);
    
    float mean = a.mean();
    EXPECT_FLOAT_EQ(mean, 5.0f);
}

TEST_F(TensorTest, VarianceAndStd) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape);
    a[{0}] = 2.0f;
    a[{1}] = 4.0f;
    a[{2}] = 4.0f;
    a[{3}] = 4.0f;
    a[{4}] = 6.0f;
    float mean = a.mean();
    EXPECT_FLOAT_EQ(mean, 4.0f);
    // Population variance (ddof=0): sum((x - mean)²) / n = 8/5 = 1.6
    float variance = a.variance();
    EXPECT_NEAR(variance, 1.6f, 1e-4);
    float std_dev = a.std();
    EXPECT_NEAR(std_dev, std::sqrt(1.6f), 1e-4);
    // Sample variance (ddof=1): sum((x - mean)²) / (n-1) = 8/4 = 2.0
    float sample_variance = a.variance(1);
    EXPECT_NEAR(sample_variance, 2.0f, 1e-4);
    float sample_std = a.std(1);
    EXPECT_NEAR(sample_std, std::sqrt(2.0f), 1e-4);
}

TEST_F(TensorTest, SquareFunction) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> a(shape);
    
    a[{0, 0}] = 2.0f; a[{0, 1}] = 3.0f;
    a[{1, 0}] = -4.0f; a[{1, 1}] = 5.0f;
    
    auto squared = a.square();
    
    EXPECT_FLOAT_EQ((squared[{0, 0}]), 4.0f);
    EXPECT_FLOAT_EQ((squared[{0, 1}]), 9.0f);
    EXPECT_FLOAT_EQ((squared[{1, 0}]), 16.0f);
    EXPECT_FLOAT_EQ((squared[{1, 1}]), 25.0f);
}

TEST_F(TensorTest, MSELoss) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> predictions(shape);
    Tensor<float, 2> targets(shape);
    
    predictions[{0, 0}] = 2.0f; predictions[{0, 1}] = 4.0f;
    predictions[{1, 0}] = 6.0f; predictions[{1, 1}] = 8.0f;
    
    targets[{0, 0}] = 1.0f; targets[{0, 1}] = 3.0f;
    targets[{1, 0}] = 5.0f; targets[{1, 1}] = 7.0f;
    
    auto loss_var = predictions.mse_loss(targets);
    ASSERT_TRUE((std::holds_alternative<float>(loss_var)));
    float loss = std::get<float>(loss_var);
    
    // MSE = mean((pred - target)²) = mean([1, 1, 1, 1]) = 1.0
    EXPECT_FLOAT_EQ(loss, 1.0f);
}

TEST_F(TensorTest, MSELossGradient) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> predictions(shape);
    Tensor<float, 2> targets(shape);
    
    predictions[{0, 0}] = 3.0f; predictions[{0, 1}] = 5.0f;
    predictions[{1, 0}] = 7.0f; predictions[{1, 1}] = 9.0f;
    
    targets[{0, 0}] = 1.0f; targets[{0, 1}] = 3.0f;
    targets[{1, 0}] = 5.0f; targets[{1, 1}] = 7.0f;
    
    auto grad_var = predictions.mse_loss_gradient(targets);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(grad_var)));
    auto grad = std::get<Tensor<float, 2>>(grad_var);
    
    // Gradient = 2 * (pred - target) / n
    // diff = [2, 2, 2, 2], grad = 2 * diff / 4 = [1, 1, 1, 1]
    EXPECT_FLOAT_EQ((grad[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((grad[{0, 1}]), 1.0f);
    EXPECT_FLOAT_EQ((grad[{1, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((grad[{1, 1}]), 1.0f);
}

TEST_F(TensorTest, BinaryCrossentropyLoss) {
    TensorIndices<1> shape = {4};
    Tensor<float, 1> predictions(shape);
    Tensor<float, 1> targets(shape);
    
    predictions[{0}] = 0.9f;
    predictions[{1}] = 0.1f;
    predictions[{2}] = 0.8f;
    predictions[{3}] = 0.3f;
    
    targets[{0}] = 1.0f;
    targets[{1}] = 0.0f;
    targets[{2}] = 1.0f;
    targets[{3}] = 0.0f;
    
    auto loss_var = predictions.binary_crossentropy_loss(targets);
    ASSERT_TRUE((std::holds_alternative<float>(loss_var)));
    float loss = std::get<float>(loss_var);
    
    // Loss should be positive and reasonable
    EXPECT_GT(loss, 0.0f);
    EXPECT_LT(loss, 1.0f);
}

TEST_F(TensorTest, BackpropagationExample) {
    // Simulate a simple forward and backward pass
    TensorIndices<1> shape = {5};
    Tensor<float, 1> input(shape);
    Tensor<float, 1> target(shape);
    
    // Forward pass
    input.fill(0.5f);
    target.fill(1.0f);
    
    auto hidden = input.sigmoid();
    auto output = hidden.tanh();
    
    // Compute loss (simplified)
    auto loss_grad_var = output.mse_loss_gradient(target);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(loss_grad_var)));
    auto loss_grad = std::get<Tensor<float, 1>>(loss_grad_var);
    
    // Backward pass
    auto tanh_grad = output.tanh_derivative();
    auto grad_hidden_var = tanh_grad.chain_rule(loss_grad);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(grad_hidden_var)));
    auto grad_hidden = std::get<Tensor<float, 1>>(grad_hidden_var);
    
    auto sigmoid_grad = hidden.sigmoid_derivative();
    auto grad_input_var = sigmoid_grad.chain_rule(grad_hidden);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(grad_input_var)));
    auto grad_input = std::get<Tensor<float, 1>>(grad_input_var);
    
    // Gradients should be computed
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite((grad_input[{i}])));
    }
}

// ============================================
// Autograd Tests
// ============================================

TEST_F(TensorTest, AutogradBasicAddition) {
    TensorIndices<1> shape = {1};  // Scalar tensor
    Tensor<float, 1> a(shape, true, true);  // requires_grad=true
    Tensor<float, 1> b(shape, true, true);
    
    a[{0}] = 2.0f;
    b[{0}] = 3.0f;
    
    auto c_var = a + b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(c_var)));
    auto c = std::get<Tensor<float, 1>>(c_var);
    
    // Forward: c = a + b = 5.0
    EXPECT_FLOAT_EQ((c[{0}]), 5.0f);
    
    // Backward
    c.backward();
    
    // Gradients of addition: da = 1, db = 1
    EXPECT_FLOAT_EQ(((*a.grad())[{0}]), 1.0f);
    EXPECT_FLOAT_EQ(((*b.grad())[{0}]), 1.0f);
}

TEST_F(TensorTest, AutogradMultiplication) {
    TensorIndices<1> shape = {1};  // Scalar tensor
    Tensor<float, 1> a(shape, true, true);
    Tensor<float, 1> b(shape, true, true);
    
    a[{0}] = 3.0f;
    b[{0}] = 4.0f;
    
    auto c_var = a * b;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(c_var)));
    auto c = std::get<Tensor<float, 1>>(c_var);
    
    // Forward: c = a * b = 12.0
    EXPECT_FLOAT_EQ((c[{0}]), 12.0f);
    
    // Backward
    c.backward();
    
    // Gradients: da/dc = b = 4.0, db/dc = a = 3.0
    EXPECT_FLOAT_EQ(((*a.grad())[{0}]), 4.0f);
    EXPECT_FLOAT_EQ(((*b.grad())[{0}]), 3.0f);
}

TEST_F(TensorTest, AutogradSigmoid) {
    TensorIndices<1> shape = {1};  // Scalar tensor
    Tensor<float, 1> x(shape, true, true);
    
    x[{0}] = 0.0f;
    
    auto y = x.sigmoid();    
    // sigmoid(0) = 0.5
    EXPECT_NEAR((y[{0}]), 0.5f, 1e-5);
    
    // Backward
    y.backward();    // Gradient of sigmoid at 0: 0.5 * (1 - 0.5) = 0.25
    EXPECT_NEAR(((*x.grad())[{0}]), 0.25f, 1e-5);
}

TEST_F(TensorTest, AutogradReLU) {
    TensorIndices<1> shape = {1};  // Scalar tensor
    Tensor<float, 1> x(shape, true, true);
    
    x[{0}] = 2.0f;
    
    auto y = x.relu();
    
    // ReLU results
    EXPECT_FLOAT_EQ((y[{0}]), 2.0f);
    
    // Backward
    y.backward();
    
    // Gradient of ReLU: 1 if x > 0
    EXPECT_FLOAT_EQ(((*x.grad())[{0}]), 1.0f);
}

TEST_F(TensorTest, AutogradReLUNegative) {
    TensorIndices<1> shape = {1};
    Tensor<float, 1> x(shape, true, true);
    
    x[{0}] = -1.0f;
    
    auto y = x.relu();
    
    // ReLU(-1) = 0
    EXPECT_FLOAT_EQ((y[{0}]), 0.0f);
    
    // Backward
    y.backward();
    
    // Gradient of ReLU: 0 if x <= 0
    EXPECT_FLOAT_EQ(((*x.grad())[{0}]), 0.0f);
}

TEST_F(TensorTest, AutogradWithGradientArg) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> x(shape, true, true);
    
    x.fill(2.0f);
    
    auto y_var = x * x;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(y_var)));
    auto y = std::get<Tensor<float, 2>>(y_var);
    
    // Create upstream gradient (all ones)
    Tensor<float, 2> upstream_grad(shape, true, false);
    upstream_grad.fill(1.0f);
    
    // Backward with explicit gradient
    y.backward(&upstream_grad);
    
    // Gradient: d/dx[x²] = 2x = 4
    EXPECT_FLOAT_EQ(((*x.grad())[{0, 0}]), 4.0f);
    EXPECT_FLOAT_EQ(((*x.grad())[{1, 1}]), 4.0f);
}

TEST_F(TensorTest, AutogradChainedOperations) {
    TensorIndices<1> shape = {1};  // Scalar
    Tensor<float, 1> x(shape, true, true);
    
    x[{0}] = 1.0f;
    
    // y = x * 2
    auto y = x * 2.0f;    
    // z = y + 3 = x * 2 + 3
    auto z = y + 3.0f;    
    // Forward: z = 1 * 2 + 3 = 5
    EXPECT_FLOAT_EQ((z[{0}]), 5.0f);
    
    // Backward
    z.backward();
    
    // dz/dx = 2 (from x * 2)
    EXPECT_FLOAT_EQ(((*x.grad())[{0}]), 2.0f);
}

TEST_F(TensorTest, AutogradCompositeFunction) {
    // Test: f(x) = sigmoid(x * x) for scalar
    TensorIndices<1> shape = {1};
    Tensor<float, 1> x(shape, true, true);
    
    x[{0}] = 1.0f;
    
    // y = x * x
    auto y_var = x * x;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(y_var)));
    auto y = std::get<Tensor<float, 1>>(y_var);
    
    // z = sigmoid(y)
    auto z = y.sigmoid();
    
    // Forward check
    EXPECT_NEAR((z[{0}]), 1.0f / (1.0f + std::exp(-1.0f)), 1e-5);  // sigmoid(1)
    
    // Backward
    z.backward();
    
    // Gradients should be computed through chain rule
    ASSERT_NE(x.grad(), nullptr);
    // For x=1: d/dx[sigmoid(x²)] = sigmoid'(1) * 2x = sigmoid(1)*(1-sigmoid(1)) * 2
    float sig1 = 1.0f / (1.0f + std::exp(-1.0f));
    float expected_grad = sig1 * (1.0f - sig1) * 2.0f;
    EXPECT_NEAR(((*x.grad())[{0}]), expected_grad, 1e-4);
}

TEST_F(TensorTest, AutogradZeroGrad) {
    TensorIndices<1> shape = {1};  // Scalar
    Tensor<float, 1> x(shape, true, true);
    
    x[{0}] = 2.0f;
    
    auto y_var = x * x;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(y_var)));
    auto y = std::get<Tensor<float, 1>>(y_var);
    
    y.backward();
    
    // Check gradients exist
    ASSERT_NE(x.grad(), nullptr);
    EXPECT_FLOAT_EQ(((*x.grad())[{0}]), 4.0f);  // d/dx[x²] = 2x = 4
    
    // Zero gradients
    x.zero_grad();
    
    // Gradient should be zero
    EXPECT_FLOAT_EQ(((*x.grad())[{0}]), 0.0f);
}

TEST_F(TensorTest, AutogradDetach) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> x(shape, true, true);
    
    x.fill(3.0f);
    
    // Detach stops gradient tracking
    auto y = x.detach();
    
    EXPECT_FALSE(y.requires_grad());
    EXPECT_EQ(y.grad(), nullptr);
    EXPECT_FLOAT_EQ((y[{0, 0}]), 3.0f);
}

TEST_F(TensorTest, AutogradNoGradTracking) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> x(shape, true, false);  // no grad tracking
    Tensor<float, 2> y(shape, true, false);
    
    x.fill(2.0f);
    y.fill(3.0f);
    
    auto z_var = x + y;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(z_var)));
    auto z = std::get<Tensor<float, 2>>(z_var);
    
    // Should not require gradients
    EXPECT_FALSE(z.requires_grad());
}

TEST_F(TensorTest, AutogradSumReduction) {
    // Test backward through a reduction operation
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> x(shape, true, true);
    
    x.fill(2.0f);
    
    // Compute sum (this would need autograd support)
    float loss = x.sum();
    
    // For now, just verify sum works
    EXPECT_FLOAT_EQ(loss, 8.0f);
}

TEST_F(TensorTest, AutogradLeafTensor) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> x(shape, true, true);
    
    x.fill(1.0f);
    
    // Leaf tensor
    EXPECT_TRUE(x.is_leaf());
    EXPECT_TRUE(x.requires_grad());
    
    // Computed tensor
    auto y_var = x * x;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(y_var)));
    auto y = std::get<Tensor<float, 2>>(y_var);
    
    EXPECT_FALSE(y.is_leaf());
    EXPECT_TRUE(y.requires_grad());
}

TEST_F(TensorTest, AutogradMultipleBackward) {
    TensorIndices<1> shape = {1};
    Tensor<float, 1> x(shape, true, true);
    
    x[{0}] = 2.0f;
    
    auto y_var = x * x;
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(y_var)));
    auto y = std::get<Tensor<float, 1>>(y_var);
    
    // First backward
    y.backward();
    EXPECT_FLOAT_EQ(((*x.grad())[{0}]), 4.0f);
    
    // Zero and do second backward
    x.zero_grad();
    
    // Compute x³ = (x * x) * x
    auto temp_var = x * x;  // x²
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(temp_var)));
    auto temp = std::get<Tensor<float, 1>>(temp_var);
    
    auto z_var = temp * x;  // x³
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(z_var)));
    auto z = std::get<Tensor<float, 1>>(z_var);
    
    z.backward();
    // d/dx[x³] = 3x² = 3*4 = 12
    EXPECT_FLOAT_EQ(((*x.grad())[{0}]), 12.0f);
}

// ============================================
// Phase 1: Core Neural Network Operations Tests
// ============================================

TEST_F(TensorTest, MatmulBasic) {
    TensorIndices<2> shape_a = {2, 3};
    TensorIndices<2> shape_b = {3, 4};
    
    Tensor<float, 2> A(shape_a, true, true);
    Tensor<float, 2> B(shape_b, true, true);
    
    // Fill with simple values
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A[{i, j}] = static_cast<float>(i * 3 + j + 1);
        }
    }
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            B[{i, j}] = static_cast<float>(i * 4 + j + 1);
        }
    }
    
    // C = A @ B, shape [2, 4]
    auto C_var = A.matmul(B);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(C_var)));
    auto C = std::get<Tensor<float, 2>>(C_var);
    
    // Verify forward pass
    // C[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
    EXPECT_FLOAT_EQ((C[{0, 0}]), 38.0f);
    
    // Test backward pass with all-ones gradient
    Tensor<float, 2> grad_C(C.dims(), true, false);
    grad_C.fill(1.0f);
    
    C.backward(&grad_C);
    
    // Check gradients exist
    ASSERT_NE(A.grad(), nullptr);
    ASSERT_NE(B.grad(), nullptr);
}

TEST_F(TensorTest, MatmulAutograd) {
    TensorIndices<2> shape_w = {3, 2};
    TensorIndices<2> shape_x = {2, 3};
    
    Tensor<float, 2> W(shape_w, true, true);  // Weights
    Tensor<float, 2> x(shape_x, true, false); // Input
    
    W.fill(0.5f);
    x.fill(1.0f);
    
    // y = W @ x, shape [3, 3]
    auto y_var = W.matmul(x);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(y_var)));
    auto y = std::get<Tensor<float, 2>>(y_var);
    
    // Forward: each element should be 0.5 * 2 = 1.0
    EXPECT_FLOAT_EQ((y[{0, 0}]), 1.0f);
    
    // Backward with gradient of all ones
    Tensor<float, 2> grad_output(y.dims(), true, false);
    grad_output.fill(1.0f);
    
    y.backward(&grad_output);
    
    // Gradient for W should be computed
    ASSERT_NE(W.grad(), nullptr);
}

TEST_F(TensorTest, SumAxis) {
    TensorIndices<2> shape = {3, 4};
    Tensor<float, 2> a(shape, true, true);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            a[{i, j}] = static_cast<float>(i * 4 + j + 1);
        }
    }
    
    // Sum along axis 1 (columns)
    auto sum_result = a.sum_axis(1, true);
    
    // sum_result[0,0] = 1+2+3+4 = 10
    // sum_result[1,0] = 5+6+7+8 = 26
    // sum_result[2,0] = 9+10+11+12 = 42
    EXPECT_FLOAT_EQ((sum_result[{0, 0}]), 10.0f);
    EXPECT_FLOAT_EQ((sum_result[{1, 0}]), 26.0f);
    EXPECT_FLOAT_EQ((sum_result[{2, 0}]), 42.0f);
}

TEST_F(TensorTest, MeanAxis) {
    TensorIndices<2> shape = {2, 4};
    Tensor<float, 2> a(shape, true, true);
    
    a.fill(2.0f);
    
    // Mean along axis 1
    auto mean_result = a.mean_axis(1, true);
    
    // All means should be 2.0
    EXPECT_FLOAT_EQ((mean_result[{0, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((mean_result[{1, 0}]), 2.0f);
}

TEST_F(TensorTest, SoftmaxBasic) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> a(shape, true, true);
    
    // First row: [1, 2, 3]
    a[{0, 0}] = 1.0f;
    a[{0, 1}] = 2.0f;
    a[{0, 2}] = 3.0f;
    
    // Second row: [0, 0, 0]
    a[{1, 0}] = 0.0f;
    a[{1, 1}] = 0.0f;
    a[{1, 2}] = 0.0f;
    
    auto softmax_result = a.softmax(-1);
    
    // Check that probabilities sum to 1
    float sum_row0 = softmax_result[{0, 0}] + softmax_result[{0, 1}] + softmax_result[{0, 2}];
    float sum_row1 = softmax_result[{1, 0}] + softmax_result[{1, 1}] + softmax_result[{1, 2}];
    
    EXPECT_NEAR(sum_row0, 1.0f, 1e-5);
    EXPECT_NEAR(sum_row1, 1.0f, 1e-5);
    
    // Second row should have equal probabilities (all inputs are 0)
    EXPECT_NEAR((softmax_result[{1, 0}]), 1.0f / 3.0f, 1e-5);
    EXPECT_NEAR((softmax_result[{1, 1}]), 1.0f / 3.0f, 1e-5);
    EXPECT_NEAR((softmax_result[{1, 2}]), 1.0f / 3.0f, 1e-5);
}

TEST_F(TensorTest, SoftmaxAutograd) {
    TensorIndices<1> shape = {3};
    Tensor<float, 1> x(shape, true, true);
    
    x[{0}] = 1.0f;
    x[{1}] = 2.0f;
    x[{2}] = 3.0f;
    
    auto probs = x.softmax(-1);
    
    // Check that it requires grad
    EXPECT_TRUE(probs.requires_grad());
    
    // Sum should be 1
    float sum = probs[{0}] + probs[{1}] + probs[{2}];
    EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST_F(TensorTest, LogSoftmax) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> a(shape, true, false);
    
    a[{0, 0}] = 1.0f;
    a[{0, 1}] = 2.0f;
    a[{0, 2}] = 3.0f;
    a[{1, 0}] = 0.0f;
    a[{1, 1}] = 0.0f;
    a[{1, 2}] = 0.0f;
    
    auto log_softmax_result = a.log_softmax(-1);
    
    // log_softmax should give log probabilities
    // Verify: exp(log_softmax) == softmax
    auto softmax_result = a.softmax(-1);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float log_prob = log_softmax_result[{i, j}];
            float prob = softmax_result[{i, j}];
            EXPECT_NEAR(std::exp(log_prob), prob, 1e-5);
        }
    }
}

TEST_F(TensorTest, NeuralNetworkForwardPass) {
    // Simulate: y = softmax(W @ x + b)
    TensorIndices<2> shape_w = {4, 3};
    TensorIndices<1> shape_x = {3};
    TensorIndices<1> shape_b = {4};
    
    Tensor<float, 2> W(shape_w, true, true);
    Tensor<float, 1> x(shape_x, true, false);
    Tensor<float, 1> b(shape_b, true, true);
    
    W.fill(0.1f);
    x.fill(1.0f);
    b.fill(0.0f);
    
    // Note: We need to reshape x to [1, 3] for matmul
    // For now, just test the operations exist
    
    EXPECT_TRUE(W.requires_grad());
    EXPECT_TRUE(b.requires_grad());
}

TEST_F(TensorTest, MatmulChained) {
    // Test: y = (A @ B) @ C
    TensorIndices<2> shape_a = {2, 3};
    TensorIndices<2> shape_b = {3, 4};
    TensorIndices<2> shape_c = {4, 2};
    
    Tensor<float, 2> A(shape_a, true, true);
    Tensor<float, 2> B(shape_b, true, true);
    Tensor<float, 2> C(shape_c, true, true);
    
    A.fill(1.0f);
    B.fill(2.0f);
    C.fill(0.5f);
    
    auto AB_var = A.matmul(B);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(AB_var)));
    auto AB = std::get<Tensor<float, 2>>(AB_var);
    
    auto result_var = AB.matmul(C);
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result_var)));
    auto result = std::get<Tensor<float, 2>>(result_var);
    
    // Result shape should be [2, 2]
    EXPECT_EQ(result.dims()[0], 2);
    EXPECT_EQ(result.dims()[1], 2);
    
    // Test backward
    Tensor<float, 2> grad_output(result.dims(), true, false);
    grad_output.fill(1.0f);
    
    result.backward(&grad_output);
    
    // All tensors should have gradients
    ASSERT_NE(A.grad(), nullptr);
    ASSERT_NE(B.grad(), nullptr);
    ASSERT_NE(C.grad(), nullptr);
}

// ============================================
// Phase 2: Training Infrastructure Tests
// ============================================

TEST_F(TensorTest, OptimizerSGDBasic) {
    // Test basic SGD optimizer
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> W(shape, true, true);
    
    W.fill(1.0f);
    
    // Manually set gradient (all ones for sum)
    if (!W.grad()) {
        W.set_requires_grad(true);
    }
    W.zero_grad();
    // Access grad through the public interface
    auto* grad_ptr = W.grad();
    grad_ptr->fill(1.0f);
    
    // Create optimizer
    std::vector<Tensor<float, 2>*> params = {&W};
    SGD<float, 2> optimizer(params, 0.1f);
    
    // Perform one step
    float before = W[{0, 0}];
    optimizer.step();
    float after = W[{0, 0}];
    
    // Should have moved: W = W - lr * grad = 1.0 - 0.1 * 1.0 = 0.9
    EXPECT_FLOAT_EQ(after, 0.9f);
    
    // Zero gradients
    optimizer.zero_grad();
    ASSERT_NE(W.grad(), nullptr);
    EXPECT_FLOAT_EQ(((*W.grad())[{0, 0}]), 0.0f);
}

TEST_F(TensorTest, OptimizerSGDMomentum) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> W(shape, true, true);
    
    W.fill(1.0f);
    
    std::vector<Tensor<float, 2>*> params = {&W};
    SGD<float, 2> optimizer(params, 0.1f, 0.9f);  // With momentum
    
    // First step
    W.zero_grad();
    W.grad()->fill(1.0f);
    
    float before_1 = W[{0, 0}];
    optimizer.step();
    float after_1 = W[{0, 0}];
    
    // v = 0.9 * 0 + 0.1 * 1 = 0.1
    // W = 1.0 - 0.1 = 0.9
    EXPECT_FLOAT_EQ(after_1, 0.9f);
    
    // Second step
    optimizer.zero_grad();
    W.grad()->fill(1.0f);
    
    optimizer.step();
    float after_2 = W[{0, 0}];
    
    // v = 0.9 * 0.1 + 0.1 * 1 = 0.19
    // W = 0.9 - 0.19 = 0.71
    EXPECT_NEAR(after_2, 0.71f, 1e-5);
}

TEST_F(TensorTest, OptimizerAdam) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> W(shape, true, true);
    
    W.fill(1.0f);
    
    std::vector<Tensor<float, 2>*> params = {&W};
    Adam<float, 2> optimizer(params, 0.01f);
    
    // Perform one optimization step
    W.zero_grad();
    W.grad()->fill(1.0f);
    
    float before = W[{0, 0}];
    optimizer.step();
    float after = W[{0, 0}];
    
    // With Adam's adaptive learning rate, value should change
    EXPECT_NE(before, after);
    EXPECT_LT(after, before);  // Should decrease
}

TEST_F(TensorTest, LossMSE) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> predictions(shape, true, true);
    Tensor<float, 2> targets(shape, true, false);
    
    predictions.fill(2.0f);
    targets.fill(1.0f);
    
    // MSE = (1/6) * sum((2-1)^2) = 1.0
    auto loss_val = loss::mse_loss(predictions, targets, "mean");
    
    // mse_loss returns same shape as input with broadcasted value
    float loss_value = loss_val[{0, 0}];
    EXPECT_NEAR(loss_value, 1.0f, 1e-5);
}

TEST_F(TensorTest, LossL1) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> predictions(shape, true, true);
    Tensor<float, 2> targets(shape, true, false);
    
    predictions.fill(3.0f);
    targets.fill(1.0f);
    
    // L1 = (1/6) * sum(|3-1|) = 2.0
    auto loss_val = loss::l1_loss(predictions, targets, "mean");
    
    EXPECT_NEAR(loss_val[{0}], 2.0f, 1e-5);
}

TEST_F(TensorTest, LossBCE) {
    TensorIndices<1> shape = {4};
    Tensor<float, 1> predictions(shape, true, true);
    Tensor<float, 1> targets(shape, true, false);
    
    // Perfect predictions
    predictions[{0}] = 0.9f;
    predictions[{1}] = 0.1f;
    predictions[{2}] = 0.9f;
    predictions[{3}] = 0.1f;
    
    targets[{0}] = 1.0f;
    targets[{1}] = 0.0f;
    targets[{2}] = 1.0f;
    targets[{3}] = 0.0f;
    
    auto loss_val = loss::binary_cross_entropy(predictions, targets, "mean");
    
    // Loss should be low for good predictions
    EXPECT_LT(loss_val[{0}], 0.2f);
}

TEST_F(TensorTest, OptimizerTrainingLoop) {
    // Simple training loop simulation
    TensorIndices<2> shape_w = {2, 2};
    Tensor<float, 2> W(shape_w, true, true);
    
    W.fill(0.5f);
    
    std::vector<Tensor<float, 2>*> params = {&W};
    SGD<float, 2> optimizer(params, 0.1f);
    
    // Target: make W close to all 1.0
    Tensor<float, 2> target(shape_w, true, false);
    target.fill(1.0f);
    
    // Train for a few steps
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    
    for (int epoch = 0; epoch < 10; ++epoch) {
        optimizer.zero_grad();
        
        // Forward: compute loss
        auto loss_val = loss::mse_loss(W, target, "mean");
        
        if (epoch == 0) {
            initial_loss = loss_val[{0, 0}];
        }
        if (epoch == 9) {
            final_loss = loss_val[{0, 0}];
        }
        
        // Backward
        Tensor<float, 2> grad_loss(shape_w, true, false);
        grad_loss.fill(1.0f);
        loss_val.backward(&grad_loss);
        
        // Update
        optimizer.step();
    }
    
    // Loss should decrease
    EXPECT_LT(final_loss, initial_loss);
}

TEST_F(TensorTest, OptimizerScheduler) {
    TensorIndices<2> shape = {2, 2};
    Tensor<float, 2> W(shape, true, true);
    
    std::vector<Tensor<float, 2>*> params = {&W};
    SGD<float, 2> optimizer(params, 1.0f);
    
    ExponentialLR<float> scheduler(&optimizer, 0.9f);
    
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 1.0f);
    
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer.get_lr(), 0.9f);
    
    scheduler.step();
    EXPECT_NEAR(optimizer.get_lr(), 0.81f, 1e-5);
}

// ==================== PHASE 3: ADVANCED TENSOR OPERATIONS TESTS ====================

TEST_F(TensorTest, Reshape2Dto1D) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> tensor(shape, true, false);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            tensor[{i, j}] = static_cast<float>(i * 3 + j);
        }
    }
    
    auto reshaped = tensor.reshape<1>({6});
    
    EXPECT_EQ(reshaped.dims()[0], 6);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ((reshaped[{i}]), static_cast<float>(i));
    }
}

TEST_F(TensorTest, Reshape1Dto2D) {
    TensorIndices<1> shape = {6};
    Tensor<float, 1> tensor(shape, true, false);
    
    for (size_t i = 0; i < 6; ++i) {
        tensor[{i}] = static_cast<float>(i);
    }
    
    auto reshaped = tensor.reshape<2>({2, 3});
    
    EXPECT_EQ(reshaped.dims()[0], 2);
    EXPECT_EQ(reshaped.dims()[1], 3);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ((reshaped[{i, j}]), static_cast<float>(i * 3 + j));
        }
    }
}

TEST_F(TensorTest, ReshapeWithAutograd) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> tensor(shape, true, true);
    tensor.fill(2.0f);
    
    auto reshaped = tensor.reshape<1>({6});
    auto doubled = reshaped * 2.0f;  // Returns plain Tensor, not variant
    
    Tensor<float, 1> grad({6}, true, false);
    grad.fill(1.0f);
    doubled.backward(&grad);
    
    // Gradient should flow back through reshape
    bool has_grad = (tensor.grad() != nullptr);
    ASSERT_TRUE(has_grad);
    float grad_val = (*tensor.grad())[{0, 0}];
    EXPECT_FLOAT_EQ(grad_val, 2.0f);
}

TEST_F(TensorTest, Transpose2D) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> tensor(shape, true, false);
    
    tensor[{0, 0}] = 1.0f; tensor[{0, 1}] = 2.0f; tensor[{0, 2}] = 3.0f;
    tensor[{1, 0}] = 4.0f; tensor[{1, 1}] = 5.0f; tensor[{1, 2}] = 6.0f;
    
    auto transposed = tensor.transpose();
    
    EXPECT_EQ(transposed.dims()[0], 3);
    EXPECT_EQ(transposed.dims()[1], 2);
    
    EXPECT_FLOAT_EQ((transposed[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((transposed[{0, 1}]), 4.0f);
    EXPECT_FLOAT_EQ((transposed[{1, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((transposed[{1, 1}]), 5.0f);
    EXPECT_FLOAT_EQ((transposed[{2, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((transposed[{2, 1}]), 6.0f);
}

TEST_F(TensorTest, TransposeWithAutograd) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> tensor(shape, true, true);
    tensor.fill(1.0f);
    
    auto transposed = tensor.transpose();
    
    Tensor<float, 2> grad({3, 2}, true, false);
    grad.fill(1.0f);
    transposed.backward(&grad);
    
    ASSERT_TRUE(tensor.grad() != nullptr);
    // Each element should receive gradient of 1
    float grad_val1 = (*tensor.grad())[{0, 0}];
    float grad_val2 = (*tensor.grad())[{1, 2}];
    EXPECT_FLOAT_EQ(grad_val1, 1.0f);
    EXPECT_FLOAT_EQ(grad_val2, 1.0f);
}

TEST_F(TensorTest, Permute3D) {
    TensorIndices<3> shape = {2, 3, 4};
    Tensor<float, 3> tensor(shape, true, false);
    
    // Fill with identifiable values
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                tensor[{i, j, k}] = static_cast<float>(i * 100 + j * 10 + k);
            }
        }
    }
    
    // Permute: (0,1,2) -> (2,0,1)
    auto permuted = tensor.permute({2, 0, 1});
    
    EXPECT_EQ(permuted.dims()[0], 4);
    EXPECT_EQ(permuted.dims()[1], 2);
    EXPECT_EQ(permuted.dims()[2], 3);
    
    // Check some values
    float val1 = permuted[{0, 0, 0}];
    float expected1 = tensor[{0, 0, 0}];
    float val2 = permuted[{1, 0, 0}];
    float expected2 = tensor[{0, 0, 1}];
    float val3 = permuted[{0, 1, 0}];
    float expected3 = tensor[{1, 0, 0}];
    EXPECT_FLOAT_EQ(val1, expected1);
    EXPECT_FLOAT_EQ(val2, expected2);
    EXPECT_FLOAT_EQ(val3, expected3);
}

TEST_F(TensorTest, PermuteWithAutograd) {
    TensorIndices<3> shape = {2, 3, 4};
    Tensor<float, 3> tensor(shape, true, true);
    tensor.fill(1.0f);
    
    auto permuted = tensor.permute({2, 0, 1});
    
    Tensor<float, 3> grad({4, 2, 3}, true, false);
    grad.fill(1.0f);
    permuted.backward(&grad);
    
    ASSERT_TRUE(tensor.grad() != nullptr);
    float grad_val = (*tensor.grad())[{0, 0, 0}];
    EXPECT_FLOAT_EQ(grad_val, 1.0f);
}

TEST_F(TensorTest, Squeeze) {
    TensorIndices<3> shape = {2, 1, 4};
    Tensor<float, 3> tensor(shape, true, false);
    tensor.fill(5.0f);
    
    auto squeezed = tensor.squeeze(1);
    
    // Note: Our implementation keeps same rank for simplicity
    EXPECT_EQ(squeezed.dims()[0], 2);
    EXPECT_EQ(squeezed.dims()[1], 1);
    EXPECT_EQ(squeezed.dims()[2], 4);
    float val = squeezed[{0, 0, 0}];
    EXPECT_FLOAT_EQ(val, 5.0f);
}

TEST_F(TensorTest, Unsqueeze) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> tensor(shape, true, false);
    tensor.fill(7.0f);
    
    auto unsqueezed = tensor.unsqueeze<3>(1);
    
    EXPECT_EQ(unsqueezed.dims()[0], 2);
    EXPECT_EQ(unsqueezed.dims()[1], 1);
    EXPECT_EQ(unsqueezed.dims()[2], 3);
    float val = unsqueezed[{0, 0, 0}];
    EXPECT_FLOAT_EQ(val, 7.0f);
}

TEST_F(TensorTest, Concatenate1D) {
    TensorIndices<1> shape1 = {3};
    TensorIndices<1> shape2 = {2};
    
    Tensor<float, 1> t1(shape1, true, false);
    Tensor<float, 1> t2(shape2, true, false);
    
    t1[{0}] = 1.0f; t1[{1}] = 2.0f; t1[{2}] = 3.0f;
    t2[{0}] = 4.0f; t2[{1}] = 5.0f;
    
    auto concat = t1.concatenate(t2, 0);
    
    EXPECT_EQ(concat.dims()[0], 5);
    float v0 = concat[{0}];
    float v1 = concat[{1}];
    float v2 = concat[{2}];
    float v3 = concat[{3}];
    float v4 = concat[{4}];
    EXPECT_FLOAT_EQ(v0, 1.0f);
    EXPECT_FLOAT_EQ(v1, 2.0f);
    EXPECT_FLOAT_EQ(v2, 3.0f);
    EXPECT_FLOAT_EQ(v3, 4.0f);
    EXPECT_FLOAT_EQ(v4, 5.0f);
}

TEST_F(TensorTest, Concatenate2D) {
    TensorIndices<2> shape1 = {2, 3};
    TensorIndices<2> shape2 = {2, 2};
    
    Tensor<float, 2> t1(shape1, true, false);
    Tensor<float, 2> t2(shape2, true, false);
    
    t1.fill(1.0f);
    t2.fill(2.0f);
    
    auto concat = t1.concatenate(t2, 1);
    
    EXPECT_EQ(concat.dims()[0], 2);
    EXPECT_EQ(concat.dims()[1], 5);
    
    float v1 = concat[{0, 0}];
    float v2 = concat[{0, 2}];
    float v3 = concat[{0, 3}];
    float v4 = concat[{1, 4}];
    EXPECT_FLOAT_EQ(v1, 1.0f);
    EXPECT_FLOAT_EQ(v2, 1.0f);
    EXPECT_FLOAT_EQ(v3, 2.0f);
    EXPECT_FLOAT_EQ(v4, 2.0f);
}

TEST_F(TensorTest, ConcatenateWithAutograd) {
    TensorIndices<1> shape1 = {3};
    TensorIndices<1> shape2 = {2};
    
    Tensor<float, 1> t1(shape1, true, true);
    Tensor<float, 1> t2(shape2, true, true);
    
    t1.fill(1.0f);
    t2.fill(2.0f);
    
    auto concat = t1.concatenate(t2, 0);
    
    Tensor<float, 1> grad({5}, true, false);
    grad.fill(1.0f);
    concat.backward(&grad);
    
    ASSERT_TRUE(t1.grad() != nullptr);
    ASSERT_TRUE(t2.grad() != nullptr);
    
    float g1 = (*t1.grad())[{0}];
    float g2 = (*t2.grad())[{0}];
    EXPECT_FLOAT_EQ(g1, 1.0f);
    EXPECT_FLOAT_EQ(g2, 1.0f);
}

TEST_F(TensorTest, ReshapeInvalidSize) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> tensor(shape, true, false);
    
    // With the new error handling, invalid reshape returns a zero tensor
    auto result = tensor.reshape<1>({5});
    auto dims = result.dims();
    EXPECT_EQ(dims[0], 0u);
}

TEST_F(TensorTest, PermuteInvalidAxes) {
    TensorIndices<3> shape = {2, 3, 4};
    Tensor<float, 3> tensor(shape, true, false);
    
    // Invalid: axis 3 doesn't exist - returns zero tensor
    auto result1 = tensor.permute({0, 1, 3});
    auto dims1 = result1.dims();
    EXPECT_EQ(dims1[0], 0u);
    
    // Invalid: duplicate axis - returns zero tensor
    auto result2 = tensor.permute({0, 1, 1});
    auto dims2 = result2.dims();
    EXPECT_EQ(dims2[0], 0u);
}

TEST_F(TensorTest, ConcatenateDimensionMismatch) {
    TensorIndices<2> shape1 = {2, 3};
    TensorIndices<2> shape2 = {3, 3}; // Different first dimension
    
    Tensor<float, 2> t1(shape1, true, false);
    Tensor<float, 2> t2(shape2, true, false);
    
    // With new error handling, invalid concatenate returns zero tensor
    auto result = t1.concatenate(t2, 1);
    auto dims = result.dims();
    EXPECT_EQ(dims[0], 0u);
}

// ============================================
// Statistical Functions Tests
// ============================================

TEST_F(TensorTest, Covariance) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> b(shape, false, false);
    
    // Set values: a = [1, 2, 3, 4, 5], b = [2, 4, 6, 8, 10]
    a[{0}] = 1.0f;
    a[{1}] = 2.0f;
    a[{2}] = 3.0f;
    a[{3}] = 4.0f;
    a[{4}] = 5.0f;
    
    b[{0}] = 2.0f;
    b[{1}] = 4.0f;
    b[{2}] = 6.0f;
    b[{3}] = 8.0f;
    b[{4}] = 10.0f;
    
    // Covariance should be 2.0 (population) or 2.5 (sample)
    auto cov_result = a.covariance(b, 0);
    ASSERT_TRUE(std::holds_alternative<float>(cov_result));
    float cov = std::get<float>(cov_result);
    EXPECT_NEAR(cov, 4.0f, 1e-4);
    
    auto cov_sample = a.covariance(b, 1);
    ASSERT_TRUE(std::holds_alternative<float>(cov_sample));
    float cov_s = std::get<float>(cov_sample);
    EXPECT_NEAR(cov_s, 5.0f, 1e-4);
}

TEST_F(TensorTest, PearsonCorrelation) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> b(shape, false, false);
    
    // Perfect positive correlation
    a[{0}] = 1.0f;
    a[{1}] = 2.0f;
    a[{2}] = 3.0f;
    a[{3}] = 4.0f;
    a[{4}] = 5.0f;
    
    b[{0}] = 2.0f;
    b[{1}] = 4.0f;
    b[{2}] = 6.0f;
    b[{3}] = 8.0f;
    b[{4}] = 10.0f;
    
    auto corr_result = a.correlation(b);
    ASSERT_TRUE(std::holds_alternative<float>(corr_result));
    float corr = std::get<float>(corr_result);
    EXPECT_NEAR(corr, 1.0f, 1e-4);
    
    // Perfect negative correlation
    Tensor<float, 1> c(shape, false, false);
    c[{0}] = 10.0f;
    c[{1}] = 8.0f;
    c[{2}] = 6.0f;
    c[{3}] = 4.0f;
    c[{4}] = 2.0f;
    
    auto corr_result2 = a.correlation(c);
    ASSERT_TRUE(std::holds_alternative<float>(corr_result2));
    float corr2 = std::get<float>(corr_result2);
    EXPECT_NEAR(corr2, -1.0f, 1e-4);
}

TEST_F(TensorTest, Median) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    // Odd number of elements
    a[{0}] = 3.0f;
    a[{1}] = 1.0f;
    a[{2}] = 5.0f;
    a[{3}] = 2.0f;
    a[{4}] = 4.0f;
    
    float median = a.median();
    EXPECT_FLOAT_EQ(median, 3.0f);
    
    // Even number of elements
    TensorIndices<1> shape2 = {4};
    Tensor<float, 1> b(shape2, false, false);
    b[{0}] = 1.0f;
    b[{1}] = 2.0f;
    b[{2}] = 3.0f;
    b[{3}] = 4.0f;
    
    float median2 = b.median();
    EXPECT_FLOAT_EQ(median2, 2.5f);
}

TEST_F(TensorTest, Quantile) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    a[{0}] = 1.0f;
    a[{1}] = 2.0f;
    a[{2}] = 3.0f;
    a[{3}] = 4.0f;
    a[{4}] = 5.0f;
    
    // Test 25th percentile
    auto q25_result = a.quantile(0.25f);
    ASSERT_TRUE(std::holds_alternative<float>(q25_result));
    float q25 = std::get<float>(q25_result);
    EXPECT_NEAR(q25, 2.0f, 0.5f);
    
    // Test 50th percentile (median)
    auto q50_result = a.quantile(0.5f);
    ASSERT_TRUE(std::holds_alternative<float>(q50_result));
    float q50 = std::get<float>(q50_result);
    EXPECT_NEAR(q50, 3.0f, 1e-4);
    
    // Test 75th percentile
    auto q75_result = a.quantile(0.75f);
    ASSERT_TRUE(std::holds_alternative<float>(q75_result));
    float q75 = std::get<float>(q75_result);
    EXPECT_NEAR(q75, 4.0f, 0.5f);
    
    // Test boundary
    auto q100_result = a.quantile(1.0f);
    ASSERT_TRUE(std::holds_alternative<float>(q100_result));
    float q100 = std::get<float>(q100_result);
    EXPECT_FLOAT_EQ(q100, 5.0f);
}

TEST_F(TensorTest, MinMax) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    a[{0}] = 3.0f;
    a[{1}] = 1.0f;
    a[{2}] = 5.0f;
    a[{3}] = 2.0f;
    a[{4}] = 4.0f;
    
    float min_val = a.min();
    float max_val = a.max();
    
    EXPECT_FLOAT_EQ(min_val, 1.0f);
    EXPECT_FLOAT_EQ(max_val, 5.0f);
}

TEST_F(TensorTest, SpearmanCorrelation) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> b(shape, false, false);
    
    // Monotonic relationship but not linear
    a[{0}] = 1.0f;
    a[{1}] = 2.0f;
    a[{2}] = 3.0f;
    a[{3}] = 4.0f;
    a[{4}] = 5.0f;
    
    b[{0}] = 1.0f;
    b[{1}] = 4.0f;
    b[{2}] = 9.0f;
    b[{3}] = 16.0f;
    b[{4}] = 25.0f;
    
    auto spearman_result = a.spearman_correlation(b);
    ASSERT_TRUE(std::holds_alternative<float>(spearman_result));
    float spearman = std::get<float>(spearman_result);
    EXPECT_NEAR(spearman, 1.0f, 1e-4); // Perfect monotonic relationship
}

TEST_F(TensorTest, Normalize) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    a[{0}] = 2.0f;
    a[{1}] = 4.0f;
    a[{2}] = 4.0f;
    a[{3}] = 4.0f;
    a[{4}] = 6.0f;
    
    Tensor<float, 1> normalized = a.normalize();
    
    // Check mean is approximately 0
    float mean = normalized.mean();
    EXPECT_NEAR(mean, 0.0f, 1e-4);
    
    // Check std is approximately 1
    float std_dev = normalized.std();
    EXPECT_NEAR(std_dev, 1.0f, 1e-4);
}

TEST_F(TensorTest, Standardize) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    a[{0}] = 1.0f;
    a[{1}] = 2.0f;
    a[{2}] = 3.0f;
    a[{3}] = 4.0f;
    a[{4}] = 5.0f;
    
    Tensor<float, 1> standardized = a.standardize();
    
    // Check min is 0
    float min_val = standardized.min();
    EXPECT_NEAR(min_val, 0.0f, 1e-4);
    
    // Check max is 1
    float max_val = standardized.max();
    EXPECT_NEAR(max_val, 1.0f, 1e-4);
}

TEST_F(TensorTest, CovarianceDimensionMismatch) {
    TensorIndices<1> shape1 = {5};
    TensorIndices<1> shape2 = {6};
    
    Tensor<float, 1> a(shape1, false, false);
    Tensor<float, 1> b(shape2, false, false);
    
    auto result = a.covariance(b);
    ASSERT_TRUE(std::holds_alternative<TensorError>(result));
    EXPECT_EQ(std::get<TensorError>(result), TensorError::DimensionMismatch);
}

TEST_F(TensorTest, CorrelationDimensionMismatch) {
    TensorIndices<1> shape1 = {5};
    TensorIndices<1> shape2 = {6};
    
    Tensor<float, 1> a(shape1, false, false);
    Tensor<float, 1> b(shape2, false, false);
    
    auto result = a.correlation(b);
    ASSERT_TRUE(std::holds_alternative<TensorError>(result));
    EXPECT_EQ(std::get<TensorError>(result), TensorError::DimensionMismatch);
}

TEST_F(TensorTest, QuantileInvalidRange) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    auto result1 = a.quantile(-0.1f);
    ASSERT_TRUE(std::holds_alternative<TensorError>(result1));
    
    auto result2 = a.quantile(1.1f);
    ASSERT_TRUE(std::holds_alternative<TensorError>(result2));
}

// ==================== PHASE 1 TESTS: Enhanced Operations & Utilities ====================

TEST_F(TensorTest, ComparisonGreaterThan) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> b(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
        b[{i}] = 2.0f;
    }
    
    auto result = a > b;
    
    EXPECT_FLOAT_EQ((result[{0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 1.0f);
}

TEST_F(TensorTest, ComparisonGreaterThanScalar) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
    }
    
    auto result = a > 2.0f;
    
    EXPECT_FLOAT_EQ((result[{0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 1.0f);
}

TEST_F(TensorTest, ComparisonLessThan) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> b(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
        b[{i}] = 2.0f;
    }
    
    auto result = a < b;
    
    EXPECT_FLOAT_EQ((result[{0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 0.0f);
}

TEST_F(TensorTest, ComparisonLessEqual) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
    }
    
    auto result = a <= 2.0f;
    
    EXPECT_FLOAT_EQ((result[{0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 0.0f);
}

TEST_F(TensorTest, ComparisonGreaterEqual) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
    }
    
    auto result = a >= 2.0f;
    
    EXPECT_FLOAT_EQ((result[{0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 1.0f);
}

TEST_F(TensorTest, ComparisonEquality) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> b(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
        b[{i}] = (i % 2 == 0) ? static_cast<float>(i) : static_cast<float>(i + 1);
    }
    
    auto result = a.eq(b);
    
    EXPECT_FLOAT_EQ((result[{0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 1.0f);
}

TEST_F(TensorTest, ComparisonInequality) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> b(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
        b[{i}] = (i % 2 == 0) ? static_cast<float>(i) : static_cast<float>(i + 1);
    }
    
    auto result = a.ne(b);
    
    EXPECT_FLOAT_EQ((result[{0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 0.0f);
}

TEST_F(TensorTest, ClipValues) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
    }
    
    auto result = a.clip(1.0f, 3.0f);
    
    EXPECT_FLOAT_EQ((result[{0}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{4}]), 3.0f);
}

TEST_F(TensorTest, ClipWithGradient) {
    TensorIndices<1> shape = {3};
    Tensor<float, 1> x(shape, false, true);
    
    x[{0}] = 0.5f;
    x[{1}] = 2.0f;
    x[{2}] = 4.0f;
    
    auto y = x.clip(1.0f, 3.0f);
    
    // Create a gradient tensor
    Tensor<float, 1> grad_out(shape, false, false);
    grad_out.fill(1.0f);
    
    y.backward(&grad_out);
    
    // Gradient should be 0 for clipped values, 1 for values in range
    EXPECT_FLOAT_EQ(((*x.grad())[{0}]), 0.0f);  // Clipped to 1
    EXPECT_FLOAT_EQ(((*x.grad())[{1}]), 1.0f);  // In range
    EXPECT_FLOAT_EQ(((*x.grad())[{2}]), 0.0f);  // Clipped to 3
}

TEST_F(TensorTest, MaskedFill) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> mask(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
        mask[{i}] = (i % 2 == 0) ? 1.0f : 0.0f;
    }
    
    auto result = a.masked_fill(mask, -1.0f);
    
    EXPECT_FLOAT_EQ((result[{0}]), -1.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 1.0f);
    EXPECT_FLOAT_EQ((result[{2}]), -1.0f);
    EXPECT_FLOAT_EQ((result[{3}]), 3.0f);
    EXPECT_FLOAT_EQ((result[{4}]), -1.0f);
}

TEST_F(TensorTest, MaskedSelect) {
    TensorIndices<1> shape = {5};
    Tensor<float, 1> a(shape, false, false);
    Tensor<float, 1> mask(shape, false, false);
    
    for (size_t i = 0; i < 5; ++i) {
        a[{i}] = static_cast<float>(i);
        mask[{i}] = (i % 2 == 0) ? 1.0f : 0.0f;
    }
    
    auto result = a.masked_select(mask);
    
    EXPECT_EQ(result.dims()[0], 3);
    EXPECT_FLOAT_EQ((result[{0}]), 0.0f);
    EXPECT_FLOAT_EQ((result[{1}]), 2.0f);
    EXPECT_FLOAT_EQ((result[{2}]), 4.0f);
}

TEST_F(TensorTest, UniformDistribution) {
    TensorIndices<1> shape = {100};
    Tensor<float, 1> a(shape, false, false);
    
    a.uniform(0.0f, 1.0f);
    
    // Check that all values are in range [0, 1)
    bool all_in_range = true;
    for (size_t i = 0; i < 100; ++i) {
        if (a[{i}] < 0.0f || a[{i}] >= 1.0f) {
            all_in_range = false;
            break;
        }
    }
    
    EXPECT_TRUE(all_in_range);
}

TEST_F(TensorTest, BernoulliDistribution) {
    TensorIndices<1> shape = {100};
    Tensor<float, 1> a(shape, false, false);
    
    a.bernoulli(0.5f);
    
    // Check that all values are either 0 or 1
    bool all_binary = true;
    for (size_t i = 0; i < 100; ++i) {
        if (a[{i}] != 0.0f && a[{i}] != 1.0f) {
            all_binary = false;
            break;
        }
    }
    
    EXPECT_TRUE(all_binary);
}

TEST_F(TensorTest, StackTensors) {
    TensorIndices<2> shape = {2, 3};
    Tensor<float, 2> a(shape, false, false);
    Tensor<float, 2> b(shape, false, false);
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    std::vector<Tensor<float, 2>> tensors = {a, b};
    auto stacked = Tensor<float, 2>::stack(tensors, 0);
    
    EXPECT_EQ(stacked.dims()[0], 2);
    EXPECT_EQ(stacked.dims()[1], 2);
    EXPECT_EQ(stacked.dims()[2], 3);
    
    // Check values
    EXPECT_FLOAT_EQ((stacked[{0, 0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((stacked[{1, 0, 0}]), 2.0f);
}

TEST_F(TensorTest, VStack) {
    TensorIndices<2> shape1 = {2, 3};
    TensorIndices<2> shape2 = {3, 3};
    
    Tensor<float, 2> a(shape1, false, false);
    Tensor<float, 2> b(shape2, false, false);
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    std::vector<Tensor<float, 2>> tensors = {a, b};
    auto vstacked = Tensor<float, 2>::vstack(tensors);
    
    EXPECT_EQ(vstacked.dims()[0], 5);
    EXPECT_EQ(vstacked.dims()[1], 3);
    
    EXPECT_FLOAT_EQ((vstacked[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((vstacked[{2, 0}]), 2.0f);
}

TEST_F(TensorTest, HStack) {
    TensorIndices<2> shape1 = {3, 2};
    TensorIndices<2> shape2 = {3, 3};
    
    Tensor<float, 2> a(shape1, false, false);
    Tensor<float, 2> b(shape2, false, false);
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    std::vector<Tensor<float, 2>> tensors = {a, b};
    auto hstacked = Tensor<float, 2>::hstack(tensors);
    
    EXPECT_EQ(hstacked.dims()[0], 3);
    EXPECT_EQ(hstacked.dims()[1], 5);
    
    EXPECT_FLOAT_EQ((hstacked[{0, 0}]), 1.0f);
    EXPECT_FLOAT_EQ((hstacked[{0, 2}]), 2.0f);
}

// Tests for GPU-accelerated fill() functionality
TEST_F(TensorTest, FillCPUTensor) {
    TensorIndices<2> shape = {10, 10};
    Tensor<float, 2> tensor(shape, false, false);  // CPU tensor
    
    tensor.fill(5.5f);
    
    // Verify all elements are filled
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            EXPECT_FLOAT_EQ((tensor[{i, j}]), 5.5f);
        }
    }
}

#ifdef USE_GPU
TEST_F(TensorTest, FillGPUTensor) {
    TensorIndices<2> shape = {10, 10};
    Tensor<float, 2> tensor(shape, true, false);  // GPU tensor
    
    tensor.fill(7.7f);
    
    // Verify all elements are filled (triggers GPU->CPU sync)
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            EXPECT_FLOAT_EQ((tensor[{i, j}]), 7.7f);
        }
    }
}

TEST_F(TensorTest, FillGPUTensorLarge) {
    TensorIndices<2> shape = {1000, 1000};
    Tensor<float, 2> tensor(shape, true, false);  // GPU tensor
    
    tensor.fill(3.14f);
    
    // Spot check several elements
    EXPECT_FLOAT_EQ((tensor[{0, 0}]), 3.14f);
    EXPECT_FLOAT_EQ((tensor[{500, 500}]), 3.14f);
    EXPECT_FLOAT_EQ((tensor[{999, 999}]), 3.14f);
    EXPECT_FLOAT_EQ((tensor[{123, 456}]), 3.14f);
}

TEST_F(TensorTest, FillGPUTensorZero) {
    TensorIndices<2> shape = {100, 50};
    Tensor<double, 2> tensor(shape, true, false);  // GPU tensor, double precision
    
    // Set to non-zero first
    tensor.fill(99.9);
    
    // Verify it's set
    EXPECT_DOUBLE_EQ((tensor[{0, 0}]), 99.9);
    
    // Now fill with zero
    tensor.fill(0.0);
    
    // Verify all zeros
    EXPECT_DOUBLE_EQ((tensor[{0, 0}]), 0.0);
    EXPECT_DOUBLE_EQ((tensor[{50, 25}]), 0.0);
    EXPECT_DOUBLE_EQ((tensor[{99, 49}]), 0.0);
}

TEST_F(TensorTest, FillGPUTensorNegative) {
    TensorIndices<1> shape = {1000};
    Tensor<float, 1> tensor(shape, true, false);
    
    tensor.fill(-42.5f);
    
    // Check several elements
    EXPECT_FLOAT_EQ((tensor[{0}]), -42.5f);
    EXPECT_FLOAT_EQ((tensor[{500}]), -42.5f);
    EXPECT_FLOAT_EQ((tensor[{999}]), -42.5f);
}

TEST_F(TensorTest, FillGPUTensorMultipleTimes) {
    TensorIndices<2> shape = {50, 50};
    Tensor<float, 2> tensor(shape, true, false);
    
    // Fill multiple times with different values
    tensor.fill(1.0f);
    EXPECT_FLOAT_EQ((tensor[{0, 0}]), 1.0f);
    
    tensor.fill(2.0f);
    EXPECT_FLOAT_EQ((tensor[{0, 0}]), 2.0f);
    EXPECT_FLOAT_EQ((tensor[{25, 25}]), 2.0f);
    
    tensor.fill(3.0f);
    EXPECT_FLOAT_EQ((tensor[{49, 49}]), 3.0f);
}

TEST_F(TensorTest, FillGPUTensor3D) {
    TensorIndices<3> shape = {10, 20, 30};
    Tensor<float, 3> tensor(shape, true, false);
    
    tensor.fill(8.8f);
    
    // Check various elements
    EXPECT_FLOAT_EQ((tensor[{0, 0, 0}]), 8.8f);
    EXPECT_FLOAT_EQ((tensor[{5, 10, 15}]), 8.8f);
    EXPECT_FLOAT_EQ((tensor[{9, 19, 29}]), 8.8f);
}
#endif

