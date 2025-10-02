#include "include/jit.h"
#include <stdexcept>
#include <memory>

namespace axe {
namespace jit {

// --- Global JIT State ---
std::unique_ptr<JitGraph> active_graph = nullptr;
bool is_tracing = false;

#include <unordered_map>

// --- JitGraph Methods ---
Node JitGraph::add_input(std::shared_ptr<Variable> var) {
    Node input_node{next_node_id_++, var};
    placeholder_inputs_.push_back(input_node);
    return input_node;
}

Node JitGraph::add_constant(const Tensor& tensor) {
    Node constant_node{get_next_node_id()};
    constant_inputs_.emplace(constant_node.id, tensor);
    return constant_node;
}

void JitGraph::add_op(const TraceableOp& op) {
    ops_.push_back(op);
}

void JitGraph::set_output_node(const Node& node) {
    output_node_ = node;
}

Tensor JitGraph::execute(const std::vector<Tensor>& inputs) {
    if (inputs.size() != placeholder_inputs_.size()) {
        throw std::runtime_error("Mismatched number of inputs for JIT execution.");
    }

    std::unordered_map<size_t, Tensor> node_values;

    // Map placeholder input nodes to provided tensor values
    for (size_t i = 0; i < inputs.size(); ++i) {
        node_values.emplace(placeholder_inputs_[i].id, inputs[i]);
    }

    // Add constant values to the map
    node_values.insert(constant_inputs_.begin(), constant_inputs_.end());

    // Execute operations in sequence
    for (const auto& op : ops_) {
        std::vector<Tensor> op_inputs;
        for (const auto& input_node : op.inputs) {
            op_inputs.push_back(node_values.at(input_node.id));
        }

        switch (op.type) {
            case OpType::Add:
                node_values.emplace(op.output.id, op_inputs[0].add(op_inputs[1]));
                break;
            case OpType::Sub:
                node_values.emplace(op.output.id, op_inputs[0].sub(op_inputs[1]));
                break;
            case OpType::Mul:
                node_values.emplace(op.output.id, op_inputs[0].mul(op_inputs[1]));
                break;
            case OpType::Div:
                node_values.emplace(op.output.id, op_inputs[0].div(op_inputs[1]));
                break;
            case OpType::MatMul:
                node_values.emplace(op.output.id, op_inputs[0].matmul(op_inputs[1]));
                break;
            case OpType::Sum:
                node_values.emplace(op.output.id, op_inputs[0].sum());
                break;
        }
    }

    return node_values.at(output_node_.id);
}


// --- JIT Tracer Control ---

void start_tracing() {
    if (is_tracing) {
        throw std::runtime_error("Tracing is already active.");
    }
    is_tracing = true;
    active_graph = std::make_unique<JitGraph>();
}

std::shared_ptr<JitGraph> stop_tracing() {
    if (!is_tracing) {
        // Return a null pointer if tracing wasn't active, to be handled by the caller.
        return nullptr;
    }
    is_tracing = false;

    // Automatically set the output of the last operation as the graph's output node.
    if (active_graph && !active_graph->get_ops().empty()) {
        active_graph->set_output_node(active_graph->get_ops().back().output);
    }

    // Transfer ownership from the unique_ptr to a shared_ptr for the caller.
    return std::move(active_graph);
}

void register_input(std::shared_ptr<Variable> var) {
    if (!is_tracing || !active_graph) {
        // It's not an error to JIT a function that doesn't use its inputs
        return;
    }
    // Do not re-register an input
    if (var->jit_node_id.has_value()) {
        return;
    }
    jit::Node new_node = active_graph->add_input(var);
    var->jit_node_id = new_node.id;
}


} // namespace jit
} // namespace axe