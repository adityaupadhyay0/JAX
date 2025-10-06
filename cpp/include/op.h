#pragma once

#include "variable.h"
#include <vector>
#include <memory>
#include <string>

namespace axe {

class Operation {
public:
    Operation(const std::string& file = "", int line = 0) : file_(file), line_(line) {}
    virtual ~Operation() = default;
    virtual void backward(const Tensor& grad_output) = 0;
    std::vector<std::shared_ptr<Variable>> inputs;

    std::string file_;
    int line_;
};

class AddOp : public Operation {
public:
    AddOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

class MulOp : public Operation {
public:
    MulOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

class MatMulOp : public Operation {
public:
    MatMulOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

class BMmOp : public Operation {
public:
    BMmOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

class TransposeOp : public Operation {
public:
    int dim0;
    int dim1;
    TransposeOp(std::shared_ptr<Variable> a, int dim0, int dim1, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

class SumOp : public Operation {
public:
    std::vector<size_t> input_shape;
    std::optional<std::vector<int>> axis;
    SumOp(std::shared_ptr<Variable> a, std::optional<std::vector<int>> axis, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

class MeanOp : public Operation {
public:
    std::vector<size_t> input_shape;
    std::optional<std::vector<int>> axis;
    MeanOp(std::shared_ptr<Variable> a, std::optional<std::vector<int>> axis, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

class SubOp : public Operation {
public:
    SubOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

// Functions to create the ops and link them to the output variable
std::shared_ptr<Variable> add(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file = "", int line = 0);
std::shared_ptr<Variable> mul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file = "", int line = 0);
std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file = "", int line = 0);
std::shared_ptr<Variable> bmm(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file = "", int line = 0);
std::shared_ptr<Variable> transpose(std::shared_ptr<Variable> a, int dim0, int dim1, const std::string& file = "", int line = 0);
std::shared_ptr<Variable> sum(std::shared_ptr<Variable> a, const std::optional<std::vector<int>>& axis, bool keepdims, const std::string& file = "", int line = 0);
std::shared_ptr<Variable> mean(std::shared_ptr<Variable> a, const std::optional<std::vector<int>>& axis, bool keepdims, const std::string& file = "", int line = 0);
std::shared_ptr<Variable> sub(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file = "", int line = 0);

class SqrtOp : public Operation {
public:
    std::shared_ptr<Variable> output; // Need to store output for backward pass
    SqrtOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> output, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

std::shared_ptr<Variable> sqrt(std::shared_ptr<Variable> a, const std::string& file = "", int line = 0);

class CastOp : public Operation {
public:
    DType from_dtype;
    DType to_dtype;
    CastOp(std::shared_ptr<Variable> a, DType from_dtype, DType to_dtype, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

std::shared_ptr<Variable> cast(std::shared_ptr<Variable> a, DType new_dtype, const std::string& file = "", int line = 0);

class ExpOp : public Operation {
public:
    std::shared_ptr<Variable> output; // Need to store output for backward pass
    ExpOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> output, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

std::shared_ptr<Variable> exp(std::shared_ptr<Variable> a, const std::string& file = "", int line = 0);

class Conv2dOp : public Operation {
public:
    int stride;
    int padding;
    Conv2dOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> weight, std::optional<std::shared_ptr<Variable>> bias, int stride, int padding, const std::string& file, int line);
    void backward(const Tensor& grad_output) override;
};

std::shared_ptr<Variable> conv2d(std::shared_ptr<Variable> a, std::shared_ptr<Variable> weight, std::optional<std::shared_ptr<Variable>> bias, int stride, int padding, const std::string& file = "", int line = 0);

class SliceOp : public Operation {
public:
    size_t dim;
    size_t index;
    SliceOp(std::shared_ptr<Variable> a, size_t dim, size_t index);
    void backward(const Tensor& grad_output) override;
};

class StackOp : public Operation {
public:
    size_t dim;
    StackOp(const std::vector<std::shared_ptr<Variable>>& inputs, size_t dim);
    void backward(const Tensor& grad_output) override;
};

std::shared_ptr<Variable> slice(std::shared_ptr<Variable> a, size_t dim, size_t index);
std::shared_ptr<Variable> stack(const std::vector<std::shared_ptr<Variable>>& inputs, size_t dim);

} // namespace axe