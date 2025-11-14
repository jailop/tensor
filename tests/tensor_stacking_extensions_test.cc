#include <gtest/gtest.h>
#include "../include/tensor.h"

using namespace tensor;

class TensorStackingExtensionsTest : public ::testing::Test {
protected:
};

TEST_F(TensorStackingExtensionsTest, Split1D) {
    Tensor<float, 1> tensor({10});
    for (size_t i = 0; i < 10; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto chunks = split(tensor, 2, 0);
    
    ASSERT_EQ(chunks.size(), 2);
    ASSERT_EQ(chunks[0].dims()[0], 5);
    ASSERT_EQ(chunks[1].dims()[0], 5);
    
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(chunks[0].begin()[i], static_cast<float>(i));
        ASSERT_FLOAT_EQ(chunks[1].begin()[i], static_cast<float>(i + 5));
    }
}

TEST_F(TensorStackingExtensionsTest, SplitUnevenChunks) {
    Tensor<float, 1> tensor({10});
    for (size_t i = 0; i < 10; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto chunks = split(tensor, 3, 0);
    
    ASSERT_EQ(chunks.size(), 3);
    ASSERT_EQ(chunks[0].dims()[0], 4);  // 10/3 = 3 remainder 1, so first chunk gets extra
    ASSERT_EQ(chunks[1].dims()[0], 3);
    ASSERT_EQ(chunks[2].dims()[0], 3);
}

TEST_F(TensorStackingExtensionsTest, Split2DAlongAxis0) {
    Tensor<float, 2> tensor({6, 4});
    for (size_t i = 0; i < tensor.total_size(); ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto chunks = split(tensor, 2, 0);
    
    ASSERT_EQ(chunks.size(), 2);
    ASSERT_EQ(chunks[0].dims()[0], 3);
    ASSERT_EQ(chunks[0].dims()[1], 4);
    ASSERT_EQ(chunks[1].dims()[0], 3);
    ASSERT_EQ(chunks[1].dims()[1], 4);
}

TEST_F(TensorStackingExtensionsTest, SplitInvalidAxis) {
    Tensor<float, 2> tensor({4, 4});
    auto chunks = split(tensor, 2, 5);  // Invalid axis
    
    ASSERT_EQ(chunks.size(), 1);  // Returns original tensor
}

TEST_F(TensorStackingExtensionsTest, SplitZeroChunks) {
    Tensor<float, 1> tensor({10});
    auto chunks = split(tensor, 0, 0);
    
    ASSERT_EQ(chunks.size(), 1);  // Returns original tensor
}

TEST_F(TensorStackingExtensionsTest, Chunk1D) {
    Tensor<float, 1> tensor({10});
    for (size_t i = 0; i < 10; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto chunks = chunk(tensor, 3, 0);
    
    ASSERT_EQ(chunks.size(), 4);  // ceil(10/3) = 4
    ASSERT_EQ(chunks[0].dims()[0], 3);
    ASSERT_EQ(chunks[1].dims()[0], 3);
    ASSERT_EQ(chunks[2].dims()[0], 3);
    ASSERT_EQ(chunks[3].dims()[0], 1);  // Last chunk has remainder
}

TEST_F(TensorStackingExtensionsTest, ChunkExactDivision) {
    Tensor<float, 1> tensor({12});
    for (size_t i = 0; i < 12; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto chunks = chunk(tensor, 4, 0);
    
    ASSERT_EQ(chunks.size(), 3);
    for (auto& c : chunks) {
        ASSERT_EQ(c.dims()[0], 4);
    }
}

TEST_F(TensorStackingExtensionsTest, ChunkInvalidAxis) {
    Tensor<float, 2> tensor({4, 4});
    auto chunks = chunk(tensor, 2, 5);
    
    ASSERT_EQ(chunks.size(), 1);
}

TEST_F(TensorStackingExtensionsTest, ChunkZeroSize) {
    Tensor<float, 1> tensor({10});
    auto chunks = chunk(tensor, 0, 0);
    
    ASSERT_EQ(chunks.size(), 1);
}

TEST_F(TensorStackingExtensionsTest, Tile1D) {
    Tensor<float, 1> tensor({3});
    tensor.begin()[0] = 1.0f;
    tensor.begin()[1] = 2.0f;
    tensor.begin()[2] = 3.0f;
    
    auto tiled = tile(tensor, {3});
    
    ASSERT_EQ(tiled.dims()[0], 9);
    
    // Pattern: 1,2,3,1,2,3,1,2,3
    ASSERT_FLOAT_EQ(tiled.begin()[0], 1.0f);
    ASSERT_FLOAT_EQ(tiled.begin()[1], 2.0f);
    ASSERT_FLOAT_EQ(tiled.begin()[2], 3.0f);
    ASSERT_FLOAT_EQ(tiled.begin()[3], 1.0f);
    ASSERT_FLOAT_EQ(tiled.begin()[4], 2.0f);
    ASSERT_FLOAT_EQ(tiled.begin()[5], 3.0f);
    ASSERT_FLOAT_EQ(tiled.begin()[6], 1.0f);
    ASSERT_FLOAT_EQ(tiled.begin()[7], 2.0f);
    ASSERT_FLOAT_EQ(tiled.begin()[8], 3.0f);
}

TEST_F(TensorStackingExtensionsTest, Tile2D) {
    Tensor<float, 2> tensor({2, 2});
    tensor.begin()[0] = 1.0f;
    tensor.begin()[1] = 2.0f;
    tensor.begin()[2] = 3.0f;
    tensor.begin()[3] = 4.0f;
    
    auto tiled = tile(tensor, {2, 2});
    
    ASSERT_EQ(tiled.dims()[0], 4);
    ASSERT_EQ(tiled.dims()[1], 4);
    
    // Check corners
    ASSERT_FLOAT_EQ((tiled[{0, 0}]), 1.0f);
    ASSERT_FLOAT_EQ((tiled[{0, 2}]), 1.0f);
    ASSERT_FLOAT_EQ((tiled[{2, 0}]), 1.0f);
    ASSERT_FLOAT_EQ((tiled[{2, 2}]), 1.0f);
}

TEST_F(TensorStackingExtensionsTest, TileNoRepetition) {
    Tensor<float, 1> tensor({5});
    for (size_t i = 0; i < 5; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto tiled = tile(tensor, {1});
    
    ASSERT_EQ(tiled.dims()[0], 5);
    
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(tiled.begin()[i], tensor.begin()[i]);
    }
}

TEST_F(TensorStackingExtensionsTest, RepeatAlongAxis) {
    Tensor<float, 2> tensor({2, 3});
    for (size_t i = 0; i < 6; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto repeated = repeat_along_axis(tensor, 2, 0);
    
    ASSERT_EQ(repeated.dims()[0], 4);
    ASSERT_EQ(repeated.dims()[1], 3);
    
    // First row should be repeated
    ASSERT_FLOAT_EQ((repeated[{0, 0}]), 0.0f);
    ASSERT_FLOAT_EQ((repeated[{1, 0}]), 0.0f);  // Repeated
    ASSERT_FLOAT_EQ((repeated[{2, 0}]), 3.0f);
    ASSERT_FLOAT_EQ((repeated[{3, 0}]), 3.0f);  // Repeated
}

TEST_F(TensorStackingExtensionsTest, RepeatAlongAxis1) {
    Tensor<float, 2> tensor({2, 3});
    for (size_t i = 0; i < 6; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto repeated = repeat_along_axis(tensor, 2, 1);
    
    ASSERT_EQ(repeated.dims()[0], 2);
    ASSERT_EQ(repeated.dims()[1], 6);
    
    // Each column should be repeated
    ASSERT_FLOAT_EQ((repeated[{0, 0}]), 0.0f);
    ASSERT_FLOAT_EQ((repeated[{0, 1}]), 0.0f);  // Repeated
    ASSERT_FLOAT_EQ((repeated[{0, 2}]), 1.0f);
    ASSERT_FLOAT_EQ((repeated[{0, 3}]), 1.0f);  // Repeated
}

TEST_F(TensorStackingExtensionsTest, RepeatInvalidAxis) {
    Tensor<float, 2> tensor({2, 2});
    tensor.fill(1.0f);
    
    auto repeated = repeat_along_axis(tensor, 2, 5);
    
    // Should return copy of original
    ASSERT_EQ(repeated.dims()[0], 2);
    ASSERT_EQ(repeated.dims()[1], 2);
}

TEST_F(TensorStackingExtensionsTest, RepeatOnce) {
    Tensor<float, 1> tensor({5});
    for (size_t i = 0; i < 5; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto repeated = repeat_along_axis(tensor, 1, 0);
    
    ASSERT_EQ(repeated.dims()[0], 5);
    
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(repeated.begin()[i], tensor.begin()[i]);
    }
}

TEST_F(TensorStackingExtensionsTest, TileMultipleDimensions) {
    Tensor<float, 3> tensor({2, 2, 2});
    for (size_t i = 0; i < 8; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto tiled = tile(tensor, {2, 1, 3});
    
    ASSERT_EQ(tiled.dims()[0], 4);
    ASSERT_EQ(tiled.dims()[1], 2);
    ASSERT_EQ(tiled.dims()[2], 6);
}

TEST_F(TensorStackingExtensionsTest, SplitSingleChunk) {
    Tensor<float, 1> tensor({10});
    for (size_t i = 0; i < 10; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto chunks = split(tensor, 1, 0);
    
    ASSERT_EQ(chunks.size(), 1);
    ASSERT_EQ(chunks[0].dims()[0], 10);
    
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_FLOAT_EQ(chunks[0].begin()[i], tensor.begin()[i]);
    }
}

TEST_F(TensorStackingExtensionsTest, ChunkLargerThanTensor) {
    Tensor<float, 1> tensor({5});
    for (size_t i = 0; i < 5; ++i) {
        tensor.begin()[i] = static_cast<float>(i);
    }
    
    auto chunks = chunk(tensor, 10, 0);
    
    ASSERT_EQ(chunks.size(), 1);
    ASSERT_EQ(chunks[0].dims()[0], 5);
}
