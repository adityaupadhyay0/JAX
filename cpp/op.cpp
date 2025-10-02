#include "include/op.h"
#include "include/autodiff.h"
#include <algorithm> // For std::fill

namespace axe {

// --- AddOp ---
AddOp::AddOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    inputs = {a, b};
}

void AddOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    auto b = inputs[1];

    if (a->requires_grad) {
        if (!a->grad) a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), a->data.dtype(), a->data.device()));
        Tensor grad_a = grad_output;
        if (a->data.shape() != grad_output.shape()) {
            grad_a = grad_output.sum();
        }
        *a->grad = a->grad->add(grad_a);
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = std::make_shared<Tensor>(Tensor::zeros(b->data.shape(), b->data.dtype(), b->data.device()));
        Tensor grad_b = grad_output;
        if (b->data.shape() != grad_output.shape()) {
            grad_b = grad_output.sum();
        }
        *b->grad = b->grad->add(grad_b);
    }
}

// --- MulOp ---
MulOp::MulOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    inputs = {a, b};
}

void MulOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    auto b = inputs[1];

    if (a->requires_grad) {
        if (!a->grad) a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), a->data.dtype(), a->data.device()));
        *a->grad = a->grad->add(grad_output.mul(b->data));
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = std::make_shared<Tensor>(Tensor::zeros(b->data.shape(), b->data.dtype(), b->data.device()));
        *b->grad = b->grad->add(grad_output.mul(a->data));
    }
}

// --- SumOp ---
SumOp::SumOp(std::shared_ptr<Variable> a) {
    inputs = {a};
}

void SumOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), a->data.dtype(), a->data.device()));
        }
        // Create a tensor of same shape as input, filled with grad_output
        auto grad_output_scalar = static_cast<const float*>(grad_output.data())[0];
        Tensor broadcasted_grad(a->data.shape(), a->data.dtype(), a->data.device());
        float* ptr = static_cast<float*>(broadcasted_grad.data());
        std::fill(ptr, ptr + broadcasted_grad.nelement(), grad_output_scalar);

        *a->grad = a->grad->add(broadcasted_grad);
    }
}

// --- SubOp ---
SubOp::SubOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    inputs = {a, b};
}

void SubOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    auto b = inputs[1];

    if (a->requires_grad) {
        if (!a->grad) a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), a->data.dtype(), a->data.device()));
        Tensor grad_a = grad_output;
        if (a->data.shape() != grad_output.shape()) {
            grad_a = grad_output.sum();
        }
        *a->grad = a->grad->add(grad_a);
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = std::make_shared<Tensor>(Tensor::zeros(b->data.shape(), b->data.dtype(), b->data.device()));
        Tensor grad_b = grad_output;
        if (b->data.shape() != grad_output.shape()) {
            grad_b = grad_output.sum();
        }
        *b->grad = b->grad->sub(grad_b);
    }
}

// --- MatMulOp ---
MatMulOp::MatMulOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    inputs = {a, b};
}

void MatMulOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    auto b = inputs[1];

    if (a->requires_grad) {
        if (!a->grad) a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), a->data.dtype(), a->data.device()));
        *a->grad = a->grad->add(grad_output.matmul(b->data.transpose()));
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = std::make_shared<Tensor>(Tensor::zeros(b->data.shape(), b->data.dtype(), b->data.device()));
        *b->grad = b->grad->add(a->data.transpose().matmul(grad_output));
    }
}


// --- Operator Functions ---
std::shared_ptr<Variable> add(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    Tensor result_data = a->data.add(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<AddOp>(a, b);
    }
    return result;
}

std::shared_ptr<Variable> mul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    Tensor result_data = a->data.mul(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<MulOp>(a, b);
    }
    return result;
}

std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    Tensor result_data = a->data.matmul(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<MatMulOp>(a, b);
    }
    return result;
}

std::shared_ptr<Variable> sum(std::shared_ptr<Variable> a) {
    Tensor result_data = a->data.sum();
    bool requires_grad = a->requires_grad && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<SumOp>(a);
    }
    return result;
}

std::shared_ptr<Variable> sub(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    Tensor result_data = a->data.sub(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<SubOp>(a, b);
    }
    return result;
}

} // namespace axe