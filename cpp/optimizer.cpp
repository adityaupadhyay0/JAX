#include "include/optimizer.h"
#include "include/jit.h"
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility> // For std::move

namespace axe {
namespace jit {

void Optimizer::add_pass(std::unique_ptr<OptimizationPass> pass) {
    passes_.emplace_back(std::move(pass));
}

void Optimizer::run(JitGraph& graph) {
    for (const auto& pass : passes_) {
        pass->run(graph);
    }
}

// --- StrengthReduction Implementation ---

namespace {
// Helper to check if a tensor is a scalar with a specific value.
bool is_scalar_value(const JitGraph& graph, const Node& node, float value) {
    if (!graph.is_constant(node)) {
        return false;
    }
    const auto tensor = graph.get_constant_value(node);
    if (tensor.nelement() != 1 || tensor.dtype() != DType::Float32 || tensor.device() != Device::CPU) {
        return false;
    }
    const auto* data_ptr = static_cast<const float*>(tensor.data());
    return *data_ptr == value;
}
} // anonymous namespace

void StrengthReduction::run(JitGraph& graph) {
    auto& ops = graph.get_ops_mut();

    for (auto& op : ops) {
        if (op.type == OpType::Mul) {
            // x * 2 -> x + x
            if (is_scalar_value(graph, op.inputs[1], 2.0f)) {
                op.type = OpType::Add;
                op.inputs[1] = op.inputs[0]; // Change to x + x
            }
            // 2 * x -> x + x
            else if (is_scalar_value(graph, op.inputs[0], 2.0f)) {
                op.type = OpType::Add;
                op.inputs[0] = op.inputs[1]; // Change to x + x
            }
        }
    }
}

// --- AlgebraicSimplification Implementation ---

void AlgebraicSimplification::run(JitGraph& graph) {
    std::unordered_map<size_t, Node> node_replacements;
    auto& ops = graph.get_ops_mut();

    bool changed = true;
    while (changed) {
        changed = false;
        auto it = ops.begin();
        while (it != ops.end()) {
            bool erased = false;
            // Apply existing replacements to the current op's inputs first
            for (auto& input : it->inputs) {
                if (node_replacements.count(input.id)) {
                    input = node_replacements.at(input.id);
                }
            }

            if (it->type == OpType::Add) {
                // x + 0 -> x
                if (is_scalar_value(graph, it->inputs[1], 0.0f)) {
                    node_replacements[it->output.id] = it->inputs[0];
                    it = ops.erase(it);
                    erased = true;
                    changed = true;
                }
                // 0 + x -> x
                else if (is_scalar_value(graph, it->inputs[0], 0.0f)) {
                    node_replacements[it->output.id] = it->inputs[1];
                    it = ops.erase(it);
                    erased = true;
                    changed = true;
                }
            } else if (it->type == OpType::Mul) {
                // x * 1 -> x
                if (is_scalar_value(graph, it->inputs[1], 1.0f)) {
                    node_replacements[it->output.id] = it->inputs[0];
                    it = ops.erase(it);
                    erased = true;
                    changed = true;
                }
                // 1 * x -> x
                else if (is_scalar_value(graph, it->inputs[0], 1.0f)) {
                    node_replacements[it->output.id] = it->inputs[1];
                    it = ops.erase(it);
                    erased = true;
                    changed = true;
                }
            }

            if (!erased) {
                ++it;
            }
        }
    }

    // Final pass to apply all replacements
    for (auto& op : ops) {
        for (auto& input : op.inputs) {
            if (node_replacements.count(input.id)) {
                input = node_replacements.at(input.id);
            }
        }
    }

    auto output_node = graph.get_output_node();
    if (node_replacements.count(output_node.id)) {
        graph.set_output_node(node_replacements.at(output_node.id));
    }
}


// --- ConstantFolding Implementation ---

void ConstantFolding::run(JitGraph& graph) {
    auto& ops = graph.get_ops_mut();
    auto it = ops.begin();

    while (it != ops.end()) {
        bool all_inputs_are_constant = true;
        for (const auto& input_node : it->inputs) {
            if (!graph.is_constant(input_node)) {
                all_inputs_are_constant = false;
                break;
            }
        }

        if (all_inputs_are_constant && !it->inputs.empty()) {
            std::vector<Tensor> input_values;
            for (const auto& input_node : it->inputs) {
                input_values.push_back(graph.get_constant_value(input_node));
            }

            Node output_node = it->output;
            bool op_folded = true;

            switch (it->type) {
                case OpType::Add:
                    graph.convert_to_constant(output_node, input_values[0].add(input_values[1]));
                    break;
                case OpType::Sub:
                    graph.convert_to_constant(output_node, input_values[0].sub(input_values[1]));
                    break;
                case OpType::Mul:
                    graph.convert_to_constant(output_node, input_values[0].mul(input_values[1]));
                    break;
                case OpType::Div:
                    graph.convert_to_constant(output_node, input_values[0].div(input_values[1]));
                    break;
                case OpType::MatMul:
                    graph.convert_to_constant(output_node, input_values[0].matmul(input_values[1]));
                    break;
                case OpType::Sum:
                    graph.convert_to_constant(output_node, input_values[0].sum());
                    break;
                default:
                    op_folded = false;
                    break;
            }

            if (op_folded) {
                it = ops.erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}


// --- DeadCodeElimination Implementation ---

void DeadCodeElimination::run(JitGraph& graph) {
    std::unordered_set<size_t> live_nodes;
    live_nodes.insert(graph.get_output_node().id);

    // Iterate backwards from the last op to the first
    for (auto it = graph.get_ops().rbegin(); it != graph.get_ops().rend(); ++it) {
        if (live_nodes.count(it->output.id)) {
            for (const auto& input_node : it->inputs) {
                live_nodes.insert(input_node.id);
            }
        }
    }

    auto& ops = graph.get_ops_mut();
    ops.erase(std::remove_if(ops.begin(), ops.end(),
        [&](const TraceableOp& op) {
            return live_nodes.find(op.output.id) == live_nodes.end();
        }),
        ops.end());
}


// --- CommonSubexpressionElimination Implementation ---

void CommonSubexpressionElimination::run(JitGraph& graph) {
    // Maps an expression signature to the node ID that computes it
    std::unordered_map<std::string, size_t> seen_expressions;
    // Maps node IDs that have been replaced to their replacement
    std::unordered_map<size_t, size_t> node_replacements;

    auto& ops = graph.get_ops_mut();
    auto it = ops.begin();

    while (it != ops.end()) {
        // Remap inputs if they were replaced by a previous CSE pass
        for (auto& input : it->inputs) {
            if (node_replacements.count(input.id)) {
                input.id = node_replacements.at(input.id);
            }
        }

        // Create a signature for the operation
        std::string signature = std::to_string(static_cast<int>(it->type));
        for (const auto& input : it->inputs) {
            signature += ":" + std::to_string(input.id);
        }

        if (seen_expressions.count(signature)) {
            // This is a common subexpression
            size_t original_node_id = seen_expressions.at(signature);
            node_replacements[it->output.id] = original_node_id;
            it = ops.erase(it); // Remove the redundant op
        } else {
            // First time seeing this expression
            seen_expressions[signature] = it->output.id;
            ++it;
        }
    }

    // Final pass to update any remaining references, including the output node
    for (auto& op : ops) {
        for (auto& input : op.inputs) {
            if (node_replacements.count(input.id)) {
                input.id = node_replacements.at(input.id);
            }
        }
    }

    auto output_node = graph.get_output_node();
    if (node_replacements.count(output_node.id)) {
        output_node.id = node_replacements.at(output_node.id);
        graph.set_output_node(output_node);
    }
}


// --- OperationFusion Implementation ---

void OperationFusion::run(JitGraph& graph) {
    auto& ops = graph.get_ops_mut();
    if (ops.size() < 2) {
        return;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        std::unordered_set<size_t> nodes_to_remove;

        for (int i = ops.size() - 1; i >= 0; --i) {
            auto& op = ops[i];
            if (op.type != OpType::Add) {
                continue;
            }

            for (size_t j = 0; j < op.inputs.size(); ++j) {
                const auto& input_node = op.inputs[j];
                if (nodes_to_remove.count(input_node.id)) continue;

                // Find the producer op for this input
                int producer_idx = -1;
                for (int k = i - 1; k >= 0; --k) {
                    if (ops[k].output.id == input_node.id) {
                        producer_idx = k;
                        break;
                    }
                }

                if (producer_idx != -1 && ops[producer_idx].type == OpType::Mul) {
                    // Check if the output of the producer is only used by the current 'Add' op
                    int use_count = 0;
                    for (const auto& user_op : ops) {
                         if (nodes_to_remove.count(user_op.output.id)) continue;
                        for (const auto& in_node : user_op.inputs) {
                            if (in_node.id == input_node.id) {
                                use_count++;
                            }
                        }
                    }

                    if (use_count == 1) {
                        // Fuse Mul and Add
                        op.type = OpType::FusedMulAdd;
                        auto& producer_op = ops[producer_idx];
                        op.inputs = {producer_op.inputs[0], producer_op.inputs[1], op.inputs[1 - j]};
                        nodes_to_remove.insert(producer_op.output.id);
                        changed = true;
                        goto next_op_in_loop;
                    }
                }
            }
            next_op_in_loop:;
        }

        if (changed) {
            ops.erase(std::remove_if(ops.begin(), ops.end(),
                [&](const TraceableOp& op) {
                    return nodes_to_remove.count(op.output.id);
                }),
                ops.end());
        }
    }
}

} // namespace jit
} // namespace axe