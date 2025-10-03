#include <gtest/gtest.h>
#include "jit.h"
#include "codegen.h"
#include "tensor.h"
#include "variable.h"
#include <string>
#include <memory>

using namespace axe;
using namespace axe::jit;

// Helper to check if a string contains a substring
bool contains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

TEST(CodeGen, GeneratesValidCppSource) {
    JitGraph graph;
    auto a = graph.add_input(std::make_shared<Variable>(Tensor({1}, DType::Float32), false));
    auto b = graph.add_input(std::make_shared<Variable>(Tensor({1}, DType::Float32), false));

    graph.add_op({OpType::Add, {a, b}, {graph.get_next_node_id()}});
    graph.set_output_node(graph.get_ops().back().output);

    CodeGenerator codegen;
    std::string code = codegen.generate(graph, "test_func");

    // Check for key components of the generated code
    EXPECT_TRUE(contains(code, "extern \"C\" axe::Tensor test_func(const std::vector<axe::Tensor>& inputs)"));
    EXPECT_TRUE(contains(code, "node_values.emplace(0, inputs[0]);"));
    EXPECT_TRUE(contains(code, "node_values.emplace(1, inputs[1]);"));
    EXPECT_TRUE(contains(code, "auto& lhs = node_values.at(0);"));
    EXPECT_TRUE(contains(code, "auto& rhs = node_values.at(1);"));
    EXPECT_TRUE(contains(code, "node_values.emplace(2, lhs.add(rhs));"));
    EXPECT_TRUE(contains(code, "return node_values.at(2);"));
}