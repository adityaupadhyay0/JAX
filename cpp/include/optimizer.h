#pragma once

#include <memory>
#include <vector>

namespace axe {
namespace jit {

// Forward declare JitGraph to avoid circular dependencies
class JitGraph;

/**
 * @class OptimizationPass
 * @brief Abstract base class for all optimization passes.
 *
 * Each concrete optimization pass will implement the `run` method, which
 * takes a JitGraph and modifies it in place to apply the optimization.
 */
class OptimizationPass {
public:
    virtual ~OptimizationPass() = default;
    virtual void run(JitGraph& graph) = 0;
    virtual const char* name() const = 0;
};


/**
 * @class Optimizer
 * @brief Manages a sequence of optimization passes to be applied to a JitGraph.
 */
class Optimizer {
public:
    Optimizer() = default;

    void add_pass(std::unique_ptr<OptimizationPass> pass);
    void run(JitGraph& graph);

private:
    std::vector<std::unique_ptr<OptimizationPass>> passes_;
};

// --- Core Optimization Passes ---

/**
 * @class ConstantFolding
 * @brief Folds operations where all inputs are constants.
 */
class ConstantFolding : public OptimizationPass {
public:
    void run(JitGraph& graph) override;
    const char* name() const override { return "ConstantFolding"; }
};

/**
 * @class CommonSubexpressionElimination
 * @brief Eliminates redundant computations of the same operation with the same inputs.
 */
class CommonSubexpressionElimination : public OptimizationPass {
public:
    void run(JitGraph& graph) override;
    const char* name() const override { return "CommonSubexpressionElimination"; }
};

/**
 * @class DeadCodeElimination
 * @brief Removes operations whose results are not used by any other operation or the final output.
 */
class DeadCodeElimination : public OptimizationPass {
public:
    void run(JitGraph& graph) override;
    const char* name() const override { return "DeadCodeElimination"; }
};

/**
 * @class OperationFusion
 * @brief Fuses sequential operations into a single, more efficient operation.
 */
class OperationFusion : public OptimizationPass {
public:
    void run(JitGraph& graph) override;
    const char* name() const override { return "OperationFusion"; }
};

/**
 * @class AlgebraicSimplification
 * @brief Applies algebraic rules to simplify expressions (e.g., x + 0 = x).
 */
class AlgebraicSimplification : public OptimizationPass {
public:
    void run(JitGraph& graph) override;
    const char* name() const override { return "AlgebraicSimplification"; }
};

/**
 * @class StrengthReduction
 * @brief Replaces expensive operations with cheaper equivalents (e.g., x * 2 -> x + x).
 */
class StrengthReduction : public OptimizationPass {
public:
    void run(JitGraph& graph) override;
    const char* name() const override { return "StrengthReduction"; }
};


} // namespace jit
} // namespace axe