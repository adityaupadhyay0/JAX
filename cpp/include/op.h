#pragma once

#include "variable.h"
#include <vector>
#include <memory>

namespace axe {

class Operation {
public:
    virtual ~Operation() = default;
    virtual void backward(const Tensor& grad_output) = 0;
    std::vector<std::shared_ptr<Variable>> inputs;
};

class AddOp : public Operation {
public:
    AddOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
    void backward(const Tensor& grad_output) override;
};

class MulOp : public Operation {
public:
    MulOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
    void backward(const Tensor& grad_output) override;
};

class MatMulOp : public Operation {
public:
    MatMulOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
    void backward(const Tensor& grad_output) override;
};

class SumOp : public Operation {
public:
    SumOp(std::shared_ptr<Variable> a);
    void backward(const Tensor& grad_output) override;
};

// Functions to create the ops and link them to the output variable
std::shared_ptr<Variable> add(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
std::shared_ptr<Variable> mul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
class SubOp : public Operation {
public:
    SubOp(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
    void backward(const Tensor& grad_output) override;
};

std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
std::shared_ptr<Variable> sum(std::shared_ptr<Variable> a);
std::shared_ptr<Variable> sub(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);

} // namespace axe