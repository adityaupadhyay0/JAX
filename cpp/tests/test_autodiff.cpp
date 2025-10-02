#include "gtest/gtest.h"
#include "variable.h"
#include "op.h"
#include <numeric>
#include <memory>

class AutodiffTest : public ::testing::Test {
protected:
    void TearDown() override {
        // No need to clear a global tape anymore
    }
};

TEST_F(AutodiffTest, AddGrad) {
    auto a_tensor = axe::Tensor::ones({2, 2}, axe::DType::Float32);
    auto b_tensor = axe::Tensor::ones({2, 2}, axe::DType::Float32);

    auto a = std::make_shared<axe::Variable>(a_tensor, true);
    auto b = std::make_shared<axe::Variable>(b_tensor, true);

    auto c = add(a, b);
    c->backward();

    ASSERT_TRUE(a->grad);
    ASSERT_TRUE(b->grad);

    const float* grad_a_ptr = static_cast<const float*>(a->grad->data());
    const float* grad_b_ptr = static_cast<const float*>(b->grad->data());

    for (size_t i = 0; i < a->grad->nelement(); ++i) {
        EXPECT_FLOAT_EQ(grad_a_ptr[i], 1.0f);
        EXPECT_FLOAT_EQ(grad_b_ptr[i], 1.0f);
    }
}

TEST_F(AutodiffTest, ChainRule) {
    auto x_tensor = axe::Tensor({1}, axe::DType::Float32);
    static_cast<float*>(x_tensor.data())[0] = 3.0f;

    auto x = std::make_shared<axe::Variable>(x_tensor, true);

    // y = x * x
    auto y = mul(x, x);

    // z = y * 2
    auto two_tensor = axe::Tensor({1}, axe::DType::Float32);
    static_cast<float*>(two_tensor.data())[0] = 2.0f;
    auto two = std::make_shared<axe::Variable>(two_tensor, true);
    auto z = mul(y, two);

    z->backward();

    ASSERT_TRUE(x->grad);
    const float* grad_x_ptr = static_cast<const float*>(x->grad->data());

    // dz/dx = dz/dy * dy/dx = 2 * (2*x) = 4*x = 12
    EXPECT_FLOAT_EQ(grad_x_ptr[0], 12.0f);
}