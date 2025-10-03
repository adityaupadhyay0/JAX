#include "include/jit_engine.h"
#include <iostream>

namespace axe {
namespace jit {

// --- Global JIT Engine ---
std::unique_ptr<JitEngine> global_engine = nullptr;

void initialize_jit_engine() {
    if (!global_engine) {
        global_engine = std::make_unique<JitEngine>();
    }
}

JitEngine::JitEngine() {
    // Constructor for the JIT engine
}

CompiledFunction JitEngine::get_or_compile(const JitGraph& graph) {
    // For now, use a simple counter to generate a unique function name for each compilation.
    // A more robust implementation would hash the graph structure to get a stable key.
    std::string function_name = "axe_jit_func_" + std::to_string(function_counter_++);

    if (cache_.count(function_name)) {
        return cache_.at(function_name);
    }

    // 1. Generate C++ source code from the graph
    std::string source_code = codegen_.generate(graph, function_name);
    // std::cout << "--- Generated C++ Code ---\n" << source_code << "\n--------------------------\n";

    // 2. Compile the source code and load the function
    CompiledFunction compiled_func = compiler_.compile_and_load(source_code, function_name);

    if (compiled_func) {
        // 3. Cache the compiled function handle
        cache_[function_name] = compiled_func;
    }

    return compiled_func;
}

} // namespace jit
} // namespace axe