#include "include/variable.h"
#include "include/op.h"
#include "include/tensor.h"
#include <vector>
#include <unordered_set>
#include <algorithm>

namespace axe {

Variable::Variable(const Tensor& data, bool requires_grad)
    : data(data), grad(nullptr), requires_grad(requires_grad), creator(nullptr) {}

void Variable::backward() {
    if (!requires_grad) {
        return;
    }

    // Topological sort of the graph
    std::vector<std::shared_ptr<Variable>> sorted_graph;
    std::unordered_set<Variable*> visited;

    std::function<void(std::shared_ptr<Variable>)> visit =
        [&](std::shared_ptr<Variable> var) {
        if (visited.count(var.get())) {
            return;
        }
        visited.insert(var.get());
        if (var->creator) {
            for (auto& input : var->creator->inputs) {
                visit(input);
            }
        }
        sorted_graph.push_back(var);
    };

    visit(shared_from_this());

    // Backward pass
    this->grad = std::make_shared<Tensor>(Tensor::ones(this->data.shape(), this->data.dtype(), this->data.device()));

    for (auto it = sorted_graph.rbegin(); it != sorted_graph.rend(); ++it) {
        auto& var = *it;
        if (var->creator) {
            var->creator->backward(*var->grad);
        }
    }
}

} // namespace axe