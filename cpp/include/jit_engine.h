#pragma once

#include "codegen.h"
#include "dynamic_compiler.h"
#include "jit.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace axe {
namespace jit {

/**
 * @class JitEngine
 * @brief Manages the full JIT compilation process and caches compiled functions.
 *
 * This class orchestrates the tracing, optimization, code generation,
 * dynamic compilation, and caching of JIT-compiled functions. It serves as
 * the main C++ backend for the Python @axe.jit decorator.
 */
class JitEngine {
public:
    JitEngine();

    /**
     * @brief Gets a compiled function for a given graph.
     *
     * If the function for this graph has already been compiled, it returns
     * the cached function handle. Otherwise, it triggers the full compilation
     * pipeline: optimize, generate code, compile, load, and cache.
     *
     * @param graph The JitGraph to be compiled.
     * @return A handle to the executable compiled function.
     */
    CompiledFunction get_or_compile(const JitGraph& graph);

private:
    CodeGenerator codegen_;
    DynamicCompiler compiler_;

    // Caches compiled functions. The key could be a hash of the graph structure.
    // For simplicity, we'll use a counter-based function name as the key for now.
    std::unordered_map<std::string, CompiledFunction> cache_;
    int function_counter_ = 0;
};

// --- Global JIT Engine ---
// A single, global instance of the JIT engine.
extern std::unique_ptr<JitEngine> global_engine;
void initialize_jit_engine();

} // namespace jit
} // namespace axe