#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <optional>
#include "tensor.h"

namespace axe {

class Operation; // Forward declaration

namespace jit { struct Node; } // Forward declaration for JIT node

class Variable : public std::enable_shared_from_this<Variable> {
public:
    Tensor data;
    std::shared_ptr<Tensor> grad;
    bool requires_grad;
    std::shared_ptr<Operation> creator;
    std::optional<size_t> jit_node_id;

    Variable(const Tensor& data, bool requires_grad = false);

    void backward();
};

} // namespace axe