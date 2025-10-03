#include "include/codegen.h"
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <cstring>

namespace axe {
namespace jit {

namespace {

// Helper to convert OpType enum to its string representation for code generation
std::string op_type_to_string(OpType type) {
    switch (type) {
        case OpType::Add: return "add";
        case OpType::Sub: return "sub";
        case OpType::Mul: return "mul";
        case OpType::Div: return "div";
        case OpType::MatMul: return "matmul";
        case OpType::Sum: return "sum";
        case OpType::FusedMulAdd: return "fused_mul_add"; // Special handling
        default: throw std::runtime_error("Unsupported OpType for code generation");
    }
}

// Helper to generate code for creating a tensor from raw data
void generate_tensor_constructor(std::stringstream& ss, size_t node_id, const Tensor& tensor) {
    ss << "  {\n";
    ss << "    const std::vector<size_t> shape = {";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        ss << tensor.shape()[i] << (i == tensor.shape().size() - 1 ? "" : ", ");
    }
    ss << "};\n";

    // For now, we only support Float32 for codegen simplicity
    if (tensor.dtype() != DType::Float32) {
        throw std::runtime_error("Code generation only supports Float32 tensors for constants.");
    }

    const auto* data_ptr = static_cast<const float*>(tensor.data());
    size_t num_elements = tensor.nelement();

    ss << "    const float data[] = {";
    for (size_t i = 0; i < num_elements; ++i) {
        ss << data_ptr[i] << (i == num_elements - 1 ? "" : ", ");
    }
    ss << "};\n";

    ss << "    axe::Tensor t(shape, axe::DType::Float32, axe::Device::CPU);\n";
    ss << "    memcpy(t.data(), data, " << num_elements * sizeof(float) << ");\n";
    ss << "    node_values.emplace(" << node_id << ", t);\n";
    ss << "  }\n";
}

} // anonymous namespace


std::string CodeGenerator::generate(const JitGraph& graph, const std::string& function_name) {
    std::stringstream ss;

    // 1. Preamble and Includes
    ss << "#include \"tensor.h\"\n";
    ss << "#include <vector>\n";
    ss << "#include <unordered_map>\n";
    ss << "#include <memory>\n";
    ss << "#include <cstring> // for memcpy\n\n";

    // 2. Function Definition
    ss << "extern \"C\" axe::Tensor " << function_name << "(const std::vector<axe::Tensor>& inputs) {\n";

    // 3. Node values map
    ss << "  std::unordered_map<size_t, axe::Tensor> node_values;\n\n";

    // 4. Input Mapping
    const auto& placeholder_inputs = graph.get_inputs();
    for (size_t i = 0; i < placeholder_inputs.size(); ++i) {
        ss << "  node_values.emplace(" << placeholder_inputs[i].id << ", inputs[" << i << "]);\n";
    }
    ss << "\n";

    // 5. Constant Reconstruction
    ss << "  // Reconstruct constant tensors\n";
    const auto& constants = graph.get_constants();
    for (const auto& pair : constants) {
        generate_tensor_constructor(ss, pair.first, pair.second);
    }
    ss << "\n";

    // 6. Operations
    for (const auto& op : graph.get_ops()) {
        ss << "  // Op for node " << op.output.id << "\n";
        ss << "  {\n";

        // FusedMulAdd is a special case
        if (op.type == OpType::FusedMulAdd) {
            ss << "    auto& a = node_values.at(" << op.inputs[0].id << ");\n";
            ss << "    auto& b = node_values.at(" << op.inputs[1].id << ");\n";
            ss << "    auto& c = node_values.at(" << op.inputs[2].id << ");\n";
            ss << "    node_values.emplace(" << op.output.id << ", a.mul(b).add(c));\n";
        } else {
            std::string op_name = op_type_to_string(op.type);
            ss << "    auto& lhs = node_values.at(" << op.inputs[0].id << ");\n";
            if (op.inputs.size() > 1) {
                ss << "    auto& rhs = node_values.at(" << op.inputs[1].id << ");\n";
                ss << "    node_values.emplace(" << op.output.id << ", lhs." << op_name << "(rhs));\n";
            } else {
                 ss << "    node_values.emplace(" << op.output.id << ", lhs." << op_name << "());\n";
            }
        }
        ss << "  }\n";
    }
    ss << "\n";

    // 7. Return statement
    ss << "  return node_values.at(" << graph.get_output_node().id << ");\n";
    ss << "}\n";

    return ss.str();
}

} // namespace jit
} // namespace axe