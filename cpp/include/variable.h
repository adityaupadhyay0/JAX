#pragma once

#include <vector>
#include <functional>
#include <memory>
#include "tensor.h"

namespace axe {

class Operation; // Forward declaration

class Variable : public std::enable_shared_from_this<Variable> {
public:
    Tensor data;
    std::shared_ptr<Tensor> grad;
    bool requires_grad;
    std::shared_ptr<Operation> creator;

    Variable(const Tensor& data, bool requires_grad = false);

    void backward();
};

} // namespace axe