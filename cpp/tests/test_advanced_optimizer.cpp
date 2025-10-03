#include <gtest/gtest.h>
#include "jit.h"
#include "optimizer.h"
#include "tensor.h"
#include "variable.h"
#include <memory>

using namespace axe;
using namespace axe::jit;

// Helper function to create a scalar tensor
Tensor create_scalar(float value) {
    Tensor t({1}, DType::Float32);
    *static_cast<float*>(t.data()) = value;
    return t;
}

TEST(AdvancedOptimizer, AlgebraicSimplification_AddZero) {
    JitGraph graph;
    auto x = graph.add_input(std::make_shared<Variable>(Tensor({1}, DType::Float32), true));
    auto zero = graph.add_constant(create_scalar(0.0f));

    graph.add_op({OpType::Add, {x, zero}, {graph.get_next_node_id()}});
    graph.set_output_node(graph.get_ops().back().output);

    EXPECT_EQ(graph.get_ops().size(), 1);

    // Run the optimization
    AlgebraicSimplification pass;
    pass.run(graph);

    // The add operation should be removed, and the output should be the original input `x`
    EXPECT_EQ(graph.get_ops().size(), 0);
    EXPECT_EQ(graph.get_output_node().id, x.id);
}

TEST(AdvancedOptimizer, AlgebraicSimplification_MulOne) {
    JitGraph graph;
    auto x = graph.add_input(std::make_shared<Variable>(Tensor({1}, DType::Float32), true));
    auto one = graph.add_constant(create_scalar(1.0f));

    graph.add_op({OpType::Mul, {x, one}, {graph.get_next_node_id()}});
    graph.set_output_node(graph.get_ops().back().output);

    EXPECT_EQ(graph.get_ops().size(), 1);

    // Run the optimization
    AlgebraicSimplification pass;
    pass.run(graph);

    // The mul operation should be removed
    EXPECT_EQ(graph.get_ops().size(), 0);
    EXPECT_EQ(graph.get_output_node().id, x.id);
}

TEST(AdvancedOptimizer, StrengthReduction_MulTwo) {
    JitGraph graph;
    auto x = graph.add_input(std::make_shared<Variable>(Tensor({1}, DType::Float32), true));
    auto two = graph.add_constant(create_scalar(2.0f));

    graph.add_op({OpType::Mul, {x, two}, {graph.get_next_node_id()}});
    graph.set_output_node(graph.get_ops().back().output);

    EXPECT_EQ(graph.get_ops().size(), 1);
    EXPECT_EQ(graph.get_ops()[0].type, OpType::Mul);

    // Run the optimization
    StrengthReduction pass;
    pass.run(graph);

    // The mul operation should be replaced by an add
    EXPECT_EQ(graph.get_ops().size(), 1);
    const auto& new_op = graph.get_ops()[0];
    EXPECT_EQ(new_op.type, OpType::Add);
    // It should now be x + x
    EXPECT_EQ(new_op.inputs[0].id, x.id);
    EXPECT_EQ(new_op.inputs[1].id, x.id);
}