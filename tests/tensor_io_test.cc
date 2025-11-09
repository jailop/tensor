/**
 * @file tensor_io_test.cc
 * @brief Unit tests for tensor I/O operations.
 */

#include <gtest/gtest.h>
#include "tensor.h"
#include "tensor_io.h"
#include <cstdio>
#include <fstream>
#include <sstream>

class TensorIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test files
    }
    
    void TearDown() override {
        // Clean up test files
        std::remove("test_binary.tnsr");
        std::remove("test_text.txt");
        std::remove("test_npy.npy");
    }
};

// Binary I/O Tests

TEST_F(TensorIOTest, SaveLoadBinary1D) {
    Tensor<float, 1> original({5});
    original.fill(0.0f);
    original[{0}] = 1.0f;
    original[{1}] = 2.0f;
    original[{2}] = 3.0f;
    original[{3}] = 4.0f;
    original[{4}] = 5.0f;
    
    ASSERT_TRUE(save_binary(original, "test_binary.tnsr"));
    
    auto result = load_binary<float, 1>("test_binary.tnsr");
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 1>>(result)));
    
    auto loaded = std::get<Tensor<float, 1>>(result);
    ASSERT_EQ(loaded.dims()[0], 5);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ((loaded[{i}]), ((original[{i}])));
    }
}

TEST_F(TensorIOTest, SaveLoadBinary2D) {
    Tensor<float, 2> original({3, 4});
    original.fill(0.0f);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            original[{i, j}] = static_cast<float>(i * 4 + j);
        }
    }
    
    ASSERT_TRUE(save_binary(original, "test_binary.tnsr"));
    
    auto result = load_binary<float, 2>("test_binary.tnsr");
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result)));
    
    auto loaded = std::get<Tensor<float, 2>>(result);
    auto shape = loaded.dims();
    ASSERT_EQ(shape[0], 3);
    ASSERT_EQ(shape[1], 4);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ((loaded[{i, j}]), ((original[{i, j}])));
        }
    }
}

TEST_F(TensorIOTest, SaveLoadBinary3D) {
    Tensor<double, 3> original({2, 3, 4});
    original.fill(0.0);
    
    size_t counter = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                original[{i, j, k}] = static_cast<double>(counter++);
            }
        }
    }
    
    ASSERT_TRUE(save_binary(original, "test_binary.tnsr"));
    
    auto result = load_binary<double, 3>("test_binary.tnsr");
    ASSERT_TRUE((std::holds_alternative<Tensor<double, 3>>(result)));
    
    auto loaded = std::get<Tensor<double, 3>>(result);
    auto shape = loaded.dims();
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
    ASSERT_EQ(shape[2], 4);
    
    counter = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                EXPECT_DOUBLE_EQ((loaded[{i, j, k}]), (static_cast<double>(counter++)));
            }
        }
    }
}

TEST_F(TensorIOTest, LoadBinaryFileNotFound) {
    auto result = load_binary<float, 2>("nonexistent_file.tnsr");
    ASSERT_TRUE((std::holds_alternative<TensorIOError>(result)));
    EXPECT_EQ(std::get<TensorIOError>(result), TensorIOError::FILE_OPEN_FAILED);
}

TEST_F(TensorIOTest, LoadBinaryInvalidFormat) {
    // Create a file with invalid magic number
    std::ofstream file("test_binary.tnsr", std::ios::binary);
    const char magic[4] = {'X', 'X', 'X', 'X'};
    file.write(magic, 4);
    file.close();
    
    auto result = load_binary<float, 2>("test_binary.tnsr");
    ASSERT_TRUE((std::holds_alternative<TensorIOError>(result)));
    EXPECT_EQ(std::get<TensorIOError>(result), TensorIOError::INVALID_FORMAT);
}

// Text I/O Tests

TEST_F(TensorIOTest, SaveText1D) {
    Tensor<float, 1> tensor({5});
    tensor.fill(0.0f);
    tensor[{0}] = 1.5f;
    tensor[{1}] = 2.5f;
    tensor[{2}] = 3.5f;
    tensor[{3}] = 4.5f;
    tensor[{4}] = 5.5f;
    
    ASSERT_TRUE(save_text(tensor, "test_text.txt", 2));
    
    std::ifstream file("test_text.txt");
    ASSERT_TRUE(file.is_open());
    
    std::string line;
    std::getline(file, line);  // Skip header
    EXPECT_TRUE(line.find("# Shape: 5") != std::string::npos);
    
    std::getline(file, line);
    EXPECT_TRUE(line.find("1.50") != std::string::npos);
    EXPECT_TRUE(line.find("2.50") != std::string::npos);
    EXPECT_TRUE(line.find("5.50") != std::string::npos);
}

TEST_F(TensorIOTest, SaveText2D) {
    Tensor<float, 2> tensor({2, 3});
    tensor.fill(0.0f);
    tensor[{0, 0}] = 1.0f;
    tensor[{0, 1}] = 2.0f;
    tensor[{0, 2}] = 3.0f;
    tensor[{1, 0}] = 4.0f;
    tensor[{1, 1}] = 5.0f;
    tensor[{1, 2}] = 6.0f;
    
    ASSERT_TRUE(save_text(tensor, "test_text.txt", 1));
    
    std::ifstream file("test_text.txt");
    ASSERT_TRUE(file.is_open());
    
    std::string line;
    std::getline(file, line);  // Skip header
    EXPECT_TRUE(line.find("# Shape: 2x3") != std::string::npos);
    
    // Check data lines
    std::getline(file, line);
    EXPECT_TRUE(line.find("1.0") != std::string::npos);
    EXPECT_TRUE(line.find("2.0") != std::string::npos);
    EXPECT_TRUE(line.find("3.0") != std::string::npos);
    
    std::getline(file, line);
    EXPECT_TRUE(line.find("4.0") != std::string::npos);
    EXPECT_TRUE(line.find("5.0") != std::string::npos);
    EXPECT_TRUE(line.find("6.0") != std::string::npos);
}

// NumPy format tests

TEST_F(TensorIOTest, SaveNPY1D) {
    Tensor<float, 1> tensor({5});
    tensor.fill(0.0f);
    for (size_t i = 0; i < 5; ++i) {
        tensor[{i}] = static_cast<float>(i + 1);
    }
    
    ASSERT_TRUE(save_npy(tensor, "test_npy.npy"));
    
    // Check file exists and has NPY magic number
    std::ifstream file("test_npy.npy", std::ios::binary);
    ASSERT_TRUE(file.is_open());
    
    unsigned char magic[6];
    file.read(reinterpret_cast<char*>(magic), 6);
    EXPECT_EQ(magic[0], 0x93);
    EXPECT_EQ(magic[1], 'N');
    EXPECT_EQ(magic[2], 'U');
    EXPECT_EQ(magic[3], 'M');
    EXPECT_EQ(magic[4], 'P');
    EXPECT_EQ(magic[5], 'Y');
}

TEST_F(TensorIOTest, SaveNPY2D) {
    Tensor<double, 2> tensor({3, 4});
    tensor.fill(0.0);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            tensor[{i, j}] = static_cast<double>(i * 4 + j);
        }
    }
    
    ASSERT_TRUE(save_npy(tensor, "test_npy.npy"));
    
    // Verify file format
    std::ifstream file("test_npy.npy", std::ios::binary);
    ASSERT_TRUE(file.is_open());
    
    unsigned char magic[6];
    file.read(reinterpret_cast<char*>(magic), 6);
    EXPECT_EQ(magic[0], 0x93);
    
    unsigned char version[2];
    file.read(reinterpret_cast<char*>(version), 2);
    EXPECT_EQ(version[0], 1);
    EXPECT_EQ(version[1], 0);
}

// Generic save function tests

TEST_F(TensorIOTest, SaveGenericBinary) {
    Tensor<float, 2> tensor({2, 3});
    tensor.fill(1.0f);
    
    ASSERT_TRUE(save(tensor, "test_binary.tnsr", TensorFileFormat::BINARY));
    
    auto result = load<float, 2>("test_binary.tnsr");
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result)));
}

TEST_F(TensorIOTest, SaveGenericText) {
    Tensor<float, 2> tensor({2, 3});
    tensor.fill(2.0f);
    
    ASSERT_TRUE(save(tensor, "test_text.txt", TensorFileFormat::TEXT, 3));
    
    std::ifstream file("test_text.txt");
    ASSERT_TRUE(file.is_open());
}

TEST_F(TensorIOTest, SaveGenericNPY) {
    Tensor<float, 2> tensor({2, 3});
    tensor.fill(3.0f);
    
    ASSERT_TRUE(save(tensor, "test_npy.npy", TensorFileFormat::NPY));
    
    std::ifstream file("test_npy.npy", std::ios::binary);
    ASSERT_TRUE(file.is_open());
    
    unsigned char magic[6];
    file.read(reinterpret_cast<char*>(magic), 6);
    EXPECT_EQ(magic[0], 0x93);
}

// Print and string conversion tests

TEST_F(TensorIOTest, Print1D) {
    Tensor<float, 1> tensor({5});
    for (size_t i = 0; i < 5; ++i) {
        tensor[{i}] = static_cast<float>(i);
    }
    
    std::ostringstream oss;
    print(tensor, oss, 2, 10, 75);
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("Tensor") != std::string::npos);
    EXPECT_TRUE(output.find("shape=[5]") != std::string::npos);
    EXPECT_TRUE(output.find("0.00") != std::string::npos);
    EXPECT_TRUE(output.find("4.00") != std::string::npos);
}

TEST_F(TensorIOTest, Print2D) {
    Tensor<float, 2> tensor({2, 3});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            tensor[{i, j}] = static_cast<float>(i * 3 + j);
        }
    }
    
    std::ostringstream oss;
    print(tensor, oss, 1, 10, 75);
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("shape=[2, 3]") != std::string::npos);
    EXPECT_TRUE(output.find("0.0") != std::string::npos);
    EXPECT_TRUE(output.find("5.0") != std::string::npos);
}

TEST_F(TensorIOTest, Print1DTruncated) {
    Tensor<float, 1> tensor({20});
    for (size_t i = 0; i < 20; ++i) {
        tensor[{i}] = static_cast<float>(i);
    }
    
    std::ostringstream oss;
    print(tensor, oss, 1, 6, 75);
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("...") != std::string::npos);
    EXPECT_TRUE(output.find("0.0") != std::string::npos);
    EXPECT_TRUE(output.find("19.0") != std::string::npos);
}

TEST_F(TensorIOTest, Print2DTruncated) {
    Tensor<float, 2> tensor({10, 10});
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            tensor[{i, j}] = static_cast<float>(i * 10 + j);
        }
    }
    
    std::ostringstream oss;
    print(tensor, oss, 1, 6, 75);
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("...") != std::string::npos);
}

TEST_F(TensorIOTest, ToString) {
    Tensor<float, 1> tensor({3});
    tensor[{0}] = 1.0f;
    tensor[{1}] = 2.0f;
    tensor[{2}] = 3.0f;
    
    std::string str = to_string(tensor, 2, 10);
    EXPECT_TRUE(str.find("Tensor") != std::string::npos);
    EXPECT_TRUE(str.find("shape=[3]") != std::string::npos);
    EXPECT_TRUE(str.find("1.00") != std::string::npos);
    EXPECT_TRUE(str.find("2.00") != std::string::npos);
    EXPECT_TRUE(str.find("3.00") != std::string::npos);
}

TEST_F(TensorIOTest, LoadAutoDetectBinary) {
    Tensor<float, 2> original({2, 2});
    original.fill(5.0f);
    
    ASSERT_TRUE(save_binary(original, "test_binary.tnsr"));
    
    auto result = load<float, 2>("test_binary.tnsr");
    ASSERT_TRUE((std::holds_alternative<Tensor<float, 2>>(result)));
    
    auto loaded = std::get<Tensor<float, 2>>(result);
    EXPECT_FLOAT_EQ((loaded[{0, 0}]), (5.0f));
}

TEST_F(TensorIOTest, DifferentDtypes) {
    // Test with int32
    Tensor<int32_t, 1> tensor_int({5});
    for (size_t i = 0; i < 5; ++i) {
        tensor_int[{i}] = static_cast<int32_t>(i * 10);
    }
    
    ASSERT_TRUE(save_binary(tensor_int, "test_binary.tnsr"));
    auto result_int = load_binary<int32_t, 1>("test_binary.tnsr");
    ASSERT_TRUE((std::holds_alternative<Tensor<int32_t, 1>>(result_int)));
    
    auto loaded = std::get<Tensor<int32_t, 1>>(result_int);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(loaded[{i}], static_cast<int32_t>(i * 10));
    }
}
