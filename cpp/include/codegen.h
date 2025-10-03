#pragma once

#include "jit.h"
#include <string>

namespace axe {
namespace jit {

/**
 * @class CodeGenerator
 * @brief Generates C++ source code from a JitGraph.
 *
 * This class takes an optimized JitGraph and translates it into a compilable
 * C++ function. The generated code will perform the same computation as the
 * graph but as a native, statically-typed C++ function.
 */
class CodeGenerator {
public:
    CodeGenerator() = default;

    /**
     * @brief Generates a C++ source string for the given graph.
     * @param graph The JitGraph to translate.
     * @param function_name The name to give the generated C++ function.
     * @return A string containing the full C++ source code.
     */
    std::string generate(const JitGraph& graph, const std::string& function_name);
};

} // namespace jit
} // namespace axe