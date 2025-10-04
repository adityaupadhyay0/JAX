#include "include/op.h"
#include "include/autodiff.h"
#include "include/jit.h"
#include <algorithm> // For std::fill

namespace axe {

// --- AddOp ---
AddOp::AddOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) : Operation(file, line) {
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
MulOp::MulOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) : Operation(file, line) {
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
SumOp::SumOp(std::shared_ptr<Variable> a, const std::string& file, int line) : Operation(file, line) {
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
SubOp::SubOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) : Operation(file, line) {
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
MatMulOp::MatMulOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) : Operation(file, line) {
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

// Helper to get or create a JIT node for a variable
static jit::Node get_or_create_jit_node(std::shared_ptr<Variable>& var) {
    // If a variable doesn't have a node ID at this point, it's a constant
    // created inside the function, not a placeholder input.
    if (!var->jit_node_id.has_value()) {
        jit::Node new_node = jit::active_graph->add_constant(var->data);
        var->jit_node_id = new_node.id;
        return new_node;
    }
    // It's a known variable (either a registered input or the output of a previous op).
    return {var->jit_node_id.value(), var};
}

std::shared_ptr<Variable> add(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) {
    Tensor result_data = a->data.add(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<AddOp>(a, b, file, line);
    }

    if (jit::is_tracing) {
        jit::Node node_a = get_or_create_jit_node(a);
        jit::Node node_b = get_or_create_jit_node(b);
        jit::Node output_node = {jit::active_graph->get_next_node_id()};
        result->jit_node_id = output_node.id;
        jit::active_graph->add_op({jit::OpType::Add, {node_a, node_b}, output_node});
    }

    return result;
}

// --- SliceOp ---
SliceOp::SliceOp(std::shared_ptr<Variable> a, size_t dim, size_t index) : dim(dim), index(index) {
    inputs = {a};
}

void SliceOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), a->data.dtype(), a->data.device()));
        }
        // Add the gradient output to the correct slice of the input's gradient tensor.
        a->grad->add_from_slice(grad_output, this->dim, this->index);
    }
}

std::shared_ptr<Variable> slice(std::shared_ptr<Variable> a, size_t dim, size_t index) {
    Tensor result_data = a->data.slice(dim, index);
    bool requires_grad = a->requires_grad && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);
    if (requires_grad) {
        result->creator = std::make_shared<SliceOp>(a, dim, index);
    }
    return result;
}


// --- StackOp ---
StackOp::StackOp(const std::vector<std::shared_ptr<Variable>>& inputs, size_t dim) : dim(dim) {
    this->inputs = inputs;
}

void StackOp::backward(const Tensor& grad_output) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input = inputs[i];
        if (input->requires_grad) {
            if (!input->grad) {
                input->grad = std::make_shared<Tensor>(Tensor::zeros(input->data.shape(), input->data.dtype(), input->data.device()));
            }
            Tensor grad_slice = grad_output.slice(dim, i);
            *input->grad = input->grad->add(grad_slice);
        }
    }
}

std::shared_ptr<Variable> stack(const std::vector<std::shared_ptr<Variable>>& inputs, size_t dim) {
    std::vector<Tensor> tensors;
    tensors.reserve(inputs.size());
    bool requires_grad = false;
    for (const auto& var : inputs) {
        tensors.push_back(var->data);
        if (var->requires_grad) {
            requires_grad = true;
        }
    }
    requires_grad = requires_grad && grad_enabled;

    Tensor result_data = Tensor::stack(tensors, dim);
    auto result = std::make_shared<Variable>(result_data, requires_grad);
    if (requires_grad) {
        result->creator = std::make_shared<StackOp>(inputs, dim);
    }
    return result;
}

std::shared_ptr<Variable> mul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) {
    Tensor result_data = a->data.mul(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<MulOp>(a, b, file, line);
    }

    if (jit::is_tracing) {
        jit::Node node_a = get_or_create_jit_node(a);
        jit::Node node_b = get_or_create_jit_node(b);
        jit::Node output_node = {jit::active_graph->get_next_node_id()};
        result->jit_node_id = output_node.id;
        jit::active_graph->add_op({jit::OpType::Mul, {node_a, node_b}, output_node});
    }

    return result;
}

std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) {
    Tensor result_data = a->data.matmul(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<MatMulOp>(a, b, file, line);
    }

    if (jit::is_tracing) {
        jit::Node node_a = get_or_create_jit_node(a);
        jit::Node node_b = get_or_create_jit_node(b);
        jit::Node output_node = {jit::active_graph->get_next_node_id()};
        result->jit_node_id = output_node.id;
        jit::active_graph->add_op({jit::OpType::MatMul, {node_a, node_b}, output_node});
    }

    return result;
}

std::shared_ptr<Variable> sum(std::shared_ptr<Variable> a, const std::string& file, int line) {
    Tensor result_data = a->data.sum();
    bool requires_grad = a->requires_grad && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<SumOp>(a, file, line);
    }

    if (jit::is_tracing) {
        jit::Node node_a = get_or_create_jit_node(a);
        jit::Node output_node = {jit::active_graph->get_next_node_id()};
        result->jit_node_id = output_node.id;
        jit::active_graph->add_op({jit::OpType::Sum, {node_a}, output_node});
    }

    return result;
}

std::shared_ptr<Variable> sub(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) {
    Tensor result_data = a->data.sub(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<SubOp>(a, b, file, line);
    }

    if (jit::is_tracing) {
        jit::Node node_a = get_or_create_jit_node(a);
        jit::Node node_b = get_or_create_jit_node(b);
        jit::Node output_node = {jit::active_graph->get_next_node_id()};
        result->jit_node_id = output_node.id;
        jit::active_graph->add_op({jit::OpType::Sub, {node_a, node_b}, output_node});
    }

    return result;
}

} // namespace axe