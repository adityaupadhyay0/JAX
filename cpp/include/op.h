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

class SumOp : public Operation {
public:
    SumOp(std::shared_ptr<Variable> a, const std::string& file, int line);
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
std::shared_ptr<Variable> sum(std::shared_ptr<Variable> a, const std::string& file = "", int line = 0);
std::shared_ptr<Variable> sub(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b, const std::string& file = "", int line = 0);

} // namespace axe