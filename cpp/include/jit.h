#pragma once

#include "variable.h"
#include <memory>
#include <string>
#include <vector>
#include <variant>

#include "optimizer.h"

namespace axe {

// Forward declare Tensor
class Tensor;

namespace jit {

// Enum to represent different operation types
enum class OpType {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Sum,
    // --- Fused Ops ---
    FusedMulAdd,
};

// Represents a node in the computation graph. It can be an input variable or the output of another op.
struct Node {
    // A unique identifier for this node within the graph
    size_t id;
    // The actual value, only present for input nodes at trace time
    std::shared_ptr<Variable> value;

    bool operator==(const Node& other) const {
        return id == other.id;
    }
};

// Represents a single operation recorded in the trace
struct TraceableOp {
    OpType type;
    std::vector<Node> inputs;
    Node output;
};

// Represents the entire traced computation graph
class JitGraph {
public:
    JitGraph() = default;

    Node add_input(std::shared_ptr<Variable> var);
    Node add_constant(const Tensor& tensor);
    void add_op(const TraceableOp& op);
    void set_output_node(const Node& node);
    void optimize();
    Tensor execute(const std::vector<Tensor>& inputs);

    const std::vector<Node>& get_inputs() const { return placeholder_inputs_; }
    const std::vector<TraceableOp>& get_ops() const { return ops_; }
    std::vector<TraceableOp>& get_ops_mut() { return ops_; }
    const Node& get_output_node() const { return output_node_; }
    size_t get_next_node_id() { return next_node_id_++; }

    // Methods for optimizer passes
    bool is_constant(const Node& node) const;
    Tensor get_constant_value(const Node& node) const;
    void convert_to_constant(const Node& node, Tensor value);
    const std::unordered_map<size_t, Tensor>& get_constants() const { return constant_inputs_; }

private:
    size_t next_node_id_ = 0;
    std::vector<Node> placeholder_inputs_;
    std::unordered_map<size_t, Tensor> constant_inputs_;
    std::vector<TraceableOp> ops_;
    Node output_node_;
};


#include <memory>

// --- Global JIT Tracer Context ---

// The currently active tracer, managed by a unique_ptr for safety
extern std::unique_ptr<JitGraph> active_graph;
// Whether tracing is currently enabled
extern bool is_tracing;

void start_tracing();
std::shared_ptr<JitGraph> stop_tracing();
void register_input(std::shared_ptr<Variable> var);


} // namespace jit
} // namespace axe