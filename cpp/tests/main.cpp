#include "gtest/gtest.h"
#include "tensor.h"

TEST(TensorTest, InitialRefCount) {
    auto t = axe::Tensor({2, 2}, axe::DType::Float32);
    ASSERT_EQ(t.ref_count(), 1);
}

TEST(TensorTest, CopyConstructorRefCount) {
    auto t1 = axe::Tensor({2, 2}, axe::DType::Float32);
    ASSERT_EQ(t1.ref_count(), 1);

    axe::Tensor t2 = t1;
    ASSERT_EQ(t1.ref_count(), 2);
    ASSERT_EQ(t2.ref_count(), 2);
}

TEST(TensorTest, AssignmentOperatorRefCount) {
    auto t1 = axe::Tensor({2, 2}, axe::DType::Float32);
    auto t2 = axe::Tensor({3, 3}, axe::DType::Float32);

    ASSERT_EQ(t1.ref_count(), 1);
    ASSERT_EQ(t2.ref_count(), 1);

    t2 = t1;
    ASSERT_EQ(t1.ref_count(), 2);
    ASSERT_EQ(t2.ref_count(), 2);
}