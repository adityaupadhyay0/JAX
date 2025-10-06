#include "include/op.h"
#include "include/autodiff.h"
#include "include/jit.h"
#include <algorithm> // For std::fill
#include <stdexcept>

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
SumOp::SumOp(std::shared_ptr<Variable> a, std::optional<std::vector<int>> axis, const std::string& file, int line)
    : Operation(file, line), input_shape(a->data.shape()), axis(axis) {
    inputs = {a};
}

void SumOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = std::make_shared<Tensor>(Tensor::zeros(input_shape, a->data.dtype(), a->data.device()));
        }

        Tensor grad_to_add = grad_output;

        // If grad_output shape doesn't match input_shape, it needs to be reshaped for broadcasting
        if (grad_output.shape() != input_shape) {
            std::vector<size_t> new_shape;
            if (axis.has_value() && !axis->empty()) {
                auto grad_dims = grad_output.shape();
                int grad_idx = 0;
                std::vector<bool> reduced_axes(input_shape.size(), false);
                for (int ax : *axis) {
                    int a_ax = ax < 0 ? ax + input_shape.size() : ax;
                    if (a_ax >= 0 && (size_t)a_ax < input_shape.size()) {
                        reduced_axes[a_ax] = true;
                    }
                }

                for (size_t i = 0; i < input_shape.size(); ++i) {
                    if (reduced_axes[i]) {
                        new_shape.push_back(1);
                    } else {
                        new_shape.push_back(grad_dims[grad_idx++]);
                    }
                }
            } else { // Full reduction
                new_shape.assign(input_shape.size(), 1);
            }
            grad_to_add = grad_output.reshape(new_shape);
        }

        *a->grad = a->grad->add(grad_to_add);
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
        *a->grad = a->grad->add(grad_output.matmul(b->data.transpose(1, 0)));
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = std::make_shared<Tensor>(Tensor::zeros(b->data.shape(), b->data.dtype(), b->data.device()));
        *b->grad = b->grad->add(a->data.transpose(1, 0).matmul(grad_output));
    }
}

// --- BMmOp ---
BMmOp::BMmOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) : Operation(file, line) {
    inputs = {a, b};
}

void BMmOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    auto b = inputs[1];

    if (a->requires_grad) {
        if (!a->grad) a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), a->data.dtype(), a->data.device()));
        *a->grad = a->grad->add(grad_output.bmm(b->data.transpose(1, 2)));
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = std::make_shared<Tensor>(Tensor::zeros(b->data.shape(), b->data.dtype(), b->data.device()));
        *b->grad = b->grad->add(a->data.transpose(1, 2).bmm(grad_output));
    }
}

// --- TransposeOp ---
TransposeOp::TransposeOp(std::shared_ptr<Variable> a, int dim0, int dim1, const std::string& file, int line) : dim0(dim0), dim1(dim1), Operation(file, line) {
    inputs = {a};
}

void TransposeOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    if (a->requires_grad) {
        if (!a->grad) a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), a->data.dtype(), a->data.device()));
        *a->grad = a->grad->add(grad_output.transpose(dim0, dim1));
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

// SqrtOp
SqrtOp::SqrtOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> output, const std::string& file, int line)
    : Operation(file, line), output(output) {
    inputs = {a};
}

void SqrtOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    if (a->requires_grad) {
        // grad = grad_output * 1 / (2 * sqrt(input))
        // which is grad_output * 1 / (2 * output)
        Tensor two = Tensor::ones(output->data.shape(), output->data.dtype(), output->data.device());
        float* two_ptr = static_cast<float*>(two.data());
        std::fill(two_ptr, two_ptr + two.nelement(), 2.0f);

        Tensor term = output->data.mul(two);
        Tensor one = Tensor::ones(term.shape(), term.dtype(), term.device());
        Tensor inv_term = one.div(term);

        Tensor grad = grad_output.mul(inv_term);

        if (a->grad) {
            *a->grad = a->grad->add(grad);
        } else {
            a->grad = std::make_shared<Tensor>(grad);
        }
    }
}

std::shared_ptr<Variable> sqrt(std::shared_ptr<Variable> a, const std::string& file, int line) {
    Tensor result_data = a->data.sqrt();
    bool requires_grad = a->requires_grad && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<SqrtOp>(a, result, file, line);
    }
    return result;
}

// CastOp
CastOp::CastOp(std::shared_ptr<Variable> a, DType from_dtype, DType to_dtype, const std::string& file, int line)
    : Operation(file, line), from_dtype(from_dtype), to_dtype(to_dtype) {
    inputs = {a};
}

void CastOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = std::make_shared<Tensor>(Tensor::zeros(a->data.shape(), from_dtype, a->data.device()));
        }
        Tensor grad_to_add = grad_output.cast(from_dtype);
        *a->grad = a->grad->add(grad_to_add);
    }
}

std::shared_ptr<Variable> cast(std::shared_ptr<Variable> a, DType new_dtype, const std::string& file, int line) {
    Tensor result_data = a->data.cast(new_dtype);
    bool requires_grad = a->requires_grad && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<CastOp>(a, a->data.dtype(), new_dtype, file, line);
    }
    return result;
}

// ExpOp
ExpOp::ExpOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> output, const std::string& file, int line)
    : Operation(file, line), output(output) {
    inputs = {a};
}

void ExpOp::backward(const Tensor& grad_output) {
    auto a = inputs[0];
    if (a->requires_grad) {
        // grad = grad_output * exp(input)
        // which is grad_output * output
        Tensor grad = grad_output.mul(output->data);
        if (a->grad) {
            *a->grad = a->grad->add(grad);
        } else {
            a->grad = std::make_shared<Tensor>(grad);
        }
    }
}

std::shared_ptr<Variable> exp(std::shared_ptr<Variable> a, const std::string& file, int line) {
    Tensor result_data = a->data.exp();
    bool requires_grad = a->requires_grad && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<ExpOp>(a, result, file, line);
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

std::shared_ptr<Variable> bmm(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line) {
    Tensor result_data = a->data.bmm(b->data);
    bool requires_grad = (a->requires_grad || b->requires_grad) && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<BMmOp>(a, b, file, line);
    }
    return result;
}

std::shared_ptr<Variable> transpose(std::shared_ptr<Variable> a, int dim0, int dim1, const std::string& file, int line) {
    Tensor result_data = a->data.transpose(dim0, dim1);
    bool requires_grad = a->requires_grad && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<TransposeOp>(a, dim0, dim1, file, line);
    }
    return result;
}

std::shared_ptr<Variable> sum(std::shared_ptr<Variable> a, const std::optional<std::vector<int>>& axis, bool keepdims, const std::string& file, int line) {
    Tensor result_data = a->data.sum(axis, keepdims);
    bool requires_grad = a->requires_grad && grad_enabled;
    auto result = std::make_shared<Variable>(result_data, requires_grad);

    if (requires_grad) {
        result->creator = std::make_shared<SumOp>(a, axis, file, line);
    }

    // JIT tracing for sum is not updated for axis/keepdims yet.
    if (jit::is_tracing && !axis.has_value()) {
        jit::Node node_a = get_or_create_jit_node(a);
        jit::Node output_node = {jit::active_graph->get_next_node_id()};
        result->jit_node_id = output_node.id;
        jit::active_graph->add_op({jit::OpType::Sum, {node_a}, output_node});
    }

    return result;
}

// MeanOp
MeanOp::MeanOp(std::shared_ptr<Variable> a, std::optional<std::vector<int>> axis, const std::string& file, int line)
    : Operation(file, line), input_shape(a->data.shape()), axis(axis) {
    inputs = {a};
}

void MeanOp::backward(const Tensor& grad_output) {
    auto input_var = inputs[0];
    if (input_var->requires_grad) {
        if (!input_var->grad) {
            input_var->grad = std::make_shared<Tensor>(Tensor::zeros(input_shape, input_var->data.dtype(), input_var->data.device()));
        }

        size_t n = 1;
        if (axis.has_value() && !axis->empty()) {
            for (int ax : *axis) {
                int a_ax = ax < 0 ? ax + input_shape.size() : ax;
                n *= input_shape[a_ax];
            }
        } else {
            n = input_var->data.nelement();
        }

        // Create a scalar tensor for the divisor
        Tensor divisor_tensor({1}, grad_output.dtype(), grad_output.device());
        switch(grad_output.dtype()) {
            case DType::Float32: *static_cast<float*>(divisor_tensor.data()) = static_cast<float>(n); break;
            case DType::Float64: *static_cast<double*>(divisor_tensor.data()) = static_cast<double>(n); break;
            case DType::Int32: *static_cast<int32_t*>(divisor_tensor.data()) = static_cast<int32_t>(n); break;
            case DType::Int64: *static_cast<int64_t*>(divisor_tensor.data()) = static_cast<int64_t>(n); break;
            default: throw std::runtime_error("Unsupported dtype for mean division");
        }

        Tensor grad_divided = grad_output.div(divisor_tensor);

        // Reshape grad_output to be broadcastable to the input shape
        if (grad_divided.shape() != input_shape) {
            std::vector<size_t> new_shape;
            if (axis.has_value() && !axis->empty()) {
                auto grad_dims = grad_divided.shape();
                int grad_idx = 0;
                std::vector<bool> reduced_axes(input_shape.size(), false);
                for (int ax : *axis) {
                    int a_ax = ax < 0 ? ax + input_shape.size() : ax;
                    if (a_ax >= 0 && (size_t)a_ax < input_shape.size()) {
                        reduced_axes[a_ax] = true;
                    }
                }

                for (size_t i = 0; i < input_shape.size(); ++i) {
                    if (reduced_axes[i]) {
                        new_shape.push_back(1);
                    } else {
                        new_shape.push_back(grad_dims[grad_idx++]);
                    }
                }
            } else { // Full reduction
                new_shape.assign(input_shape.size(), 1);
            }
            grad_divided = grad_divided.reshape(new_shape);
        }

        *input_var->grad = input_var->grad->add(grad_divided);
    }
}

std::shared_ptr<Variable> mean(std::shared_ptr<Variable> a, const std::optional<std::vector<int>>& axis, bool keepdims, const std::string& file, int line) {
    auto output_tensor = a->data.mean(axis, keepdims);
    auto output_variable = std::make_shared<Variable>(output_tensor, a->requires_grad);
    if (a->requires_grad) {
        output_variable->creator = std::make_shared<MeanOp>(a, axis, file, line);
    }
    return output_variable;
}

// Conv2dOp
Conv2dOp::Conv2dOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> weight, std::optional<std::shared_ptr<Variable>> bias, int stride, int padding, const std::string& file, int line)
    : Operation(file, line), stride(stride), padding(padding) {
    inputs = {a, weight};
    if (bias) {
        inputs.push_back(*bias);
    }
}

void Conv2dOp::backward(const Tensor& grad_output) {
    auto input_var = inputs[0];
    auto weight_var = inputs[1];
    std::optional<std::shared_ptr<Variable>> bias_var;
    if (inputs.size() > 2) {
        bias_var = inputs[2];
    }

    const Tensor& input = input_var->data;
    const Tensor& weight = weight_var->data;

    const size_t N = input.shape()[0];
    const size_t C_in = input.shape()[1];
    const size_t H_in = input.shape()[2];
    const size_t W_in = input.shape()[3];

    const size_t C_out = weight.shape()[0];
    const size_t K_h = weight.shape()[2];
    const size_t K_w = weight.shape()[3];

    const size_t H_out = grad_output.shape()[2];
    const size_t W_out = grad_output.shape()[3];

    const float* grad_output_data = static_cast<const float*>(grad_output.data());
    const float* input_data = static_cast<const float*>(input.data());

    // Calculate bias gradient
    if (bias_var && (*bias_var)->requires_grad) {
        if (!(*bias_var)->grad) {
            (*bias_var)->grad = std::make_shared<Tensor>(Tensor::zeros((*bias_var)->data.shape(), (*bias_var)->data.dtype(), (*bias_var)->data.device()));
        }
        float* grad_bias_data = static_cast<float*>((*bias_var)->grad->data());

        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            float grad_sum = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                for (size_t h_out = 0; h_out < H_out; ++h_out) {
                    for (size_t w_out = 0; w_out < W_out; ++w_out) {
                        grad_sum += grad_output_data[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out];
                    }
                }
            }
            grad_bias_data[c_out] += grad_sum;
        }
    }

    // Calculate weight gradient
    if (weight_var->requires_grad) {
        if (!weight_var->grad) {
            weight_var->grad = std::make_shared<Tensor>(Tensor::zeros(weight.shape(), weight.dtype(), weight.device()));
        }
        float* grad_weight_data = static_cast<float*>(weight_var->grad->data());

        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            for (size_t c_in = 0; c_in < C_in; ++c_in) {
                for (size_t kh = 0; kh < K_h; ++kh) {
                    for (size_t kw = 0; kw < K_w; ++kw) {
                        float grad_sum = 0.0f;
                        for (size_t n = 0; n < N; ++n) {
                            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                                    long h_in_pos = h_out * stride - padding + kh;
                                    long w_in_pos = w_out * stride - padding + kw;

                                    if (h_in_pos >= 0 && h_in_pos < (long)H_in && w_in_pos >= 0 && w_in_pos < (long)W_in) {
                                        size_t input_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in_pos * W_in + w_in_pos;
                                        size_t grad_output_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
                                        grad_sum += input_data[input_idx] * grad_output_data[grad_output_idx];
                                    }
                                }
                            }
                        }
                        size_t grad_weight_idx = c_out * C_in * K_h * K_w + c_in * K_h * K_w + kh * K_w + kw;
                        grad_weight_data[grad_weight_idx] += grad_sum;
                    }
                }
            }
        }
    }

    // Calculate input gradient (Transposed Convolution)
    if (input_var->requires_grad) {
         if (!input_var->grad) {
            input_var->grad = std::make_shared<Tensor>(Tensor::zeros(input.shape(), input.dtype(), input.device()));
        }
        float* grad_input_data = static_cast<float*>(input_var->grad->data());
        const float* weight_data = static_cast<const float*>(weight.data());

        for (size_t n = 0; n < N; ++n) {
            for (size_t c_in = 0; c_in < C_in; ++c_in) {
                for (size_t h_in = 0; h_in < H_in; ++h_in) {
                    for (size_t w_in = 0; w_in < W_in; ++w_in) {
                        float grad_sum = 0.0f;
                        for (size_t c_out = 0; c_out < C_out; ++c_out) {
                            for (size_t kh = 0; kh < K_h; ++kh) {
                                for (size_t kw = 0; kw < K_w; ++kw) {
                                    long h_out_num = h_in + padding - kh;
                                    long w_out_num = w_in + padding - kw;

                                    if (h_out_num >= 0 && h_out_num % stride == 0 && w_out_num >=0 && w_out_num % stride == 0) {
                                        long h_out = h_out_num / stride;
                                        long w_out = w_out_num / stride;
                                        if (h_out >= 0 && h_out < (long)H_out && w_out >= 0 && w_out < (long)W_out) {
                                            size_t grad_output_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
                                            size_t weight_idx = c_out * C_in * K_h * K_w + c_in * K_h * K_w + (K_h - 1 - kh) * K_w + (K_w - 1 - kw);
                                            grad_sum += grad_output_data[grad_output_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                        size_t grad_input_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                        grad_input_data[grad_input_idx] += grad_sum;
                    }
                }
            }
        }
    }
}

std::shared_ptr<Variable> conv2d(std::shared_ptr<Variable> a, std::shared_ptr<Variable> weight, std::optional<std::shared_ptr<Variable>> bias, int stride, int padding, const std::string& file, int line) {
    std::optional<Tensor> bias_tensor;
    if (bias) {
        bias_tensor = (*bias)->data;
    }

    auto output_tensor = a->data.conv2d(weight->data, bias_tensor, stride, padding);

    bool requires_grad = a->requires_grad || weight->requires_grad || (bias && (*bias)->requires_grad);
    auto output_variable = std::make_shared<Variable>(output_tensor, requires_grad);

    if (requires_grad) {
        output_variable->creator = std::make_shared<Conv2dOp>(a, weight, bias, stride, padding, file, line);
    }

    return output_variable;
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