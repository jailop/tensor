#include <gtest/gtest.h>
#include "../include/tensor.h"
#include <algorithm>

class TensorSortingTest : public ::testing::Test {
protected:
};

TEST_F(TensorSortingTest, SortAscending) {
    Tensor<float, 1> tensor({5});
    tensor.data()[0] = 3.0f;
    tensor.data()[1] = 1.0f;
    tensor.data()[2] = 4.0f;
    tensor.data()[3] = 1.0f;
    tensor.data()[4] = 5.0f;
    
    auto sorted = sort(tensor, true);
    
    ASSERT_FLOAT_EQ(sorted.data()[0], 1.0f);
    ASSERT_FLOAT_EQ(sorted.data()[1], 1.0f);
    ASSERT_FLOAT_EQ(sorted.data()[2], 3.0f);
    ASSERT_FLOAT_EQ(sorted.data()[3], 4.0f);
    ASSERT_FLOAT_EQ(sorted.data()[4], 5.0f);
}

TEST_F(TensorSortingTest, SortDescending) {
    Tensor<float, 1> tensor({5});
    tensor.data()[0] = 3.0f;
    tensor.data()[1] = 1.0f;
    tensor.data()[2] = 4.0f;
    tensor.data()[3] = 1.0f;
    tensor.data()[4] = 5.0f;
    
    auto sorted = sort(tensor, false);
    
    ASSERT_FLOAT_EQ(sorted.data()[0], 5.0f);
    ASSERT_FLOAT_EQ(sorted.data()[1], 4.0f);
    ASSERT_FLOAT_EQ(sorted.data()[2], 3.0f);
    ASSERT_FLOAT_EQ(sorted.data()[3], 1.0f);
    ASSERT_FLOAT_EQ(sorted.data()[4], 1.0f);
}

TEST_F(TensorSortingTest, ArgsortAscending) {
    Tensor<float, 1> tensor({5});
    tensor.data()[0] = 3.0f;
    tensor.data()[1] = 1.0f;
    tensor.data()[2] = 4.0f;
    tensor.data()[3] = 1.0f;
    tensor.data()[4] = 5.0f;
    
    auto indices = argsort(tensor, true);
    
    ASSERT_EQ(indices.data()[0], 1);  // 1.0f at index 1
    ASSERT_EQ(indices.data()[1], 3);  // 1.0f at index 3
    ASSERT_EQ(indices.data()[2], 0);  // 3.0f at index 0
    ASSERT_EQ(indices.data()[3], 2);  // 4.0f at index 2
    ASSERT_EQ(indices.data()[4], 4);  // 5.0f at index 4
}

TEST_F(TensorSortingTest, ArgsortDescending) {
    Tensor<float, 1> tensor({5});
    tensor.data()[0] = 3.0f;
    tensor.data()[1] = 1.0f;
    tensor.data()[2] = 4.0f;
    tensor.data()[3] = 1.0f;
    tensor.data()[4] = 5.0f;
    
    auto indices = argsort(tensor, false);
    
    ASSERT_EQ(indices.data()[0], 4);  // 5.0f at index 4
    ASSERT_EQ(indices.data()[1], 2);  // 4.0f at index 2
    ASSERT_EQ(indices.data()[2], 0);  // 3.0f at index 0
}

TEST_F(TensorSortingTest, TopKLargest) {
    Tensor<float, 1> tensor({10});
    for (size_t i = 0; i < 10; ++i) {
        tensor.data()[i] = static_cast<float>(i);
    }
    
    auto [values, indices] = topk(tensor, 3, true);
    
    ASSERT_EQ(values.dims()[0], 3);
    ASSERT_EQ(indices.dims()[0], 3);
    
    ASSERT_FLOAT_EQ(values.data()[0], 9.0f);
    ASSERT_FLOAT_EQ(values.data()[1], 8.0f);
    ASSERT_FLOAT_EQ(values.data()[2], 7.0f);
    
    ASSERT_EQ(indices.data()[0], 9);
    ASSERT_EQ(indices.data()[1], 8);
    ASSERT_EQ(indices.data()[2], 7);
}

TEST_F(TensorSortingTest, TopKSmallest) {
    Tensor<float, 1> tensor({10});
    for (size_t i = 0; i < 10; ++i) {
        tensor.data()[i] = static_cast<float>(10 - i);
    }
    
    auto [values, indices] = topk(tensor, 3, false);
    
    ASSERT_EQ(values.dims()[0], 3);
    ASSERT_EQ(indices.dims()[0], 3);
    
    ASSERT_FLOAT_EQ(values.data()[0], 1.0f);
    ASSERT_FLOAT_EQ(values.data()[1], 2.0f);
    ASSERT_FLOAT_EQ(values.data()[2], 3.0f);
    
    ASSERT_EQ(indices.data()[0], 9);
    ASSERT_EQ(indices.data()[1], 8);
    ASSERT_EQ(indices.data()[2], 7);
}

TEST_F(TensorSortingTest, TopKExceedsSize) {
    Tensor<float, 1> tensor({5});
    for (size_t i = 0; i < 5; ++i) {
        tensor.data()[i] = static_cast<float>(i);
    }
    
    auto [values, indices] = topk(tensor, 10, true);
    
    // Should return all 5 elements
    ASSERT_EQ(values.dims()[0], 5);
    ASSERT_EQ(indices.dims()[0], 5);
}

TEST_F(TensorSortingTest, UniqueElements) {
    Tensor<float, 1> tensor({10});
    tensor.data()[0] = 1.0f;
    tensor.data()[1] = 3.0f;
    tensor.data()[2] = 1.0f;
    tensor.data()[3] = 5.0f;
    tensor.data()[4] = 3.0f;
    tensor.data()[5] = 3.0f;
    tensor.data()[6] = 7.0f;
    tensor.data()[7] = 1.0f;
    tensor.data()[8] = 5.0f;
    tensor.data()[9] = 9.0f;
    
    auto uniq = unique(tensor);
    
    ASSERT_EQ(uniq.dims()[0], 5);  // 1, 3, 5, 7, 9
    
    ASSERT_FLOAT_EQ(uniq.data()[0], 1.0f);
    ASSERT_FLOAT_EQ(uniq.data()[1], 3.0f);
    ASSERT_FLOAT_EQ(uniq.data()[2], 5.0f);
    ASSERT_FLOAT_EQ(uniq.data()[3], 7.0f);
    ASSERT_FLOAT_EQ(uniq.data()[4], 9.0f);
}

TEST_F(TensorSortingTest, UniqueAllSame) {
    Tensor<float, 1> tensor({5});
    tensor.fill(2.0f);
    
    auto uniq = unique(tensor);
    
    ASSERT_EQ(uniq.dims()[0], 1);
    ASSERT_FLOAT_EQ(uniq.data()[0], 2.0f);
}

TEST_F(TensorSortingTest, UniqueAllDifferent) {
    Tensor<float, 1> tensor({5});
    for (size_t i = 0; i < 5; ++i) {
        tensor.data()[i] = static_cast<float>(i);
    }
    
    auto uniq = unique(tensor);
    
    ASSERT_EQ(uniq.dims()[0], 5);
}

TEST_F(TensorSortingTest, SearchSorted) {
    // Create sorted values: [1, 3, 5, 7, 9]
    Tensor<float, 1> values({5});
    for (size_t i = 0; i < 5; ++i) {
        values.data()[i] = static_cast<float>(2 * i + 1);
    }
    
    // Search for: [0, 2, 4, 6, 8, 10]
    Tensor<float, 1> search_vals({6});
    search_vals.data()[0] = 0.0f;
    search_vals.data()[1] = 2.0f;
    search_vals.data()[2] = 4.0f;
    search_vals.data()[3] = 6.0f;
    search_vals.data()[4] = 8.0f;
    search_vals.data()[5] = 10.0f;
    
    auto indices = searchsorted(values, search_vals);
    
    ASSERT_EQ(indices.data()[0], 0);  // 0 goes before 1
    ASSERT_EQ(indices.data()[1], 1);  // 2 goes between 1 and 3
    ASSERT_EQ(indices.data()[2], 2);  // 4 goes between 3 and 5
    ASSERT_EQ(indices.data()[3], 3);  // 6 goes between 5 and 7
    ASSERT_EQ(indices.data()[4], 4);  // 8 goes between 7 and 9
    ASSERT_EQ(indices.data()[5], 5);  // 10 goes after 9
}

TEST_F(TensorSortingTest, SearchSortedExactMatches) {
    Tensor<float, 1> values({5});
    for (size_t i = 0; i < 5; ++i) {
        values.data()[i] = static_cast<float>(i * 2);
    }
    
    Tensor<float, 1> search_vals({3});
    search_vals.data()[0] = 0.0f;
    search_vals.data()[1] = 4.0f;
    search_vals.data()[2] = 8.0f;
    
    auto indices = searchsorted(values, search_vals);
    
    ASSERT_EQ(indices.data()[0], 0);
    ASSERT_EQ(indices.data()[1], 2);
    ASSERT_EQ(indices.data()[2], 4);
}

TEST_F(TensorSortingTest, SortEmpty) {
    Tensor<float, 1> tensor({0});
    auto sorted = sort(tensor, true);
    ASSERT_EQ(sorted.dims()[0], 0);
}

TEST_F(TensorSortingTest, SortSingleElement) {
    Tensor<float, 1> tensor({1});
    tensor.data()[0] = 5.0f;
    
    auto sorted = sort(tensor, true);
    
    ASSERT_EQ(sorted.dims()[0], 1);
    ASSERT_FLOAT_EQ(sorted.data()[0], 5.0f);
}

TEST_F(TensorSortingTest, ArgsortPreservesOriginal) {
    Tensor<float, 1> tensor({5});
    for (size_t i = 0; i < 5; ++i) {
        tensor.data()[i] = static_cast<float>(5 - i);
    }
    
    float original[5];
    for (size_t i = 0; i < 5; ++i) {
        original[i] = tensor.data()[i];
    }
    
    auto indices = argsort(tensor, true);
    
    // Verify original is unchanged
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(tensor.data()[i], original[i]);
    }
}
