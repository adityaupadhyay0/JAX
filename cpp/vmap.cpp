#include "include/vmap.h"
#include "include/tensor.h"
#include "include/variable.h"
#include "include/op.h"
#include "include/exception.h"
#include <vector>
#include <iostream>
#include <optional>

namespace axe {
namespace vmap_internal {

// Helper to convert py::object to a vector of optional integers
std::vector<std::optional<int>> parse_axes(const py::object& axes, size_t num_args) {
    std::vector<std::optional<int>> result(num_args);
    if (py::isinstance<py::int_>(axes)) {
        int axis = axes.cast<int>();
        for (size_t i = 0; i < num_args; ++i) {
            result[i] = axis;
        }
    } else if (py::isinstance<py::tuple>(axes) || py::isinstance<py::list>(axes)) {
        py::sequence seq = axes.cast<py::sequence>();
        if (seq.size() != num_args) {
            throw AxeException("The number of in_axes must match the number of arguments.");
        }
        for (size_t i = 0; i < num_args; ++i) {
            if (seq[i].is_none()) {
                result[i] = std::nullopt;
            } else {
                result[i] = seq[i].cast<int>();
            }
        }
    } else {
        throw AxeException("in_axes must be an int, a tuple, or a list.");
    }
    return result;
}

// Helper to extract a Variable from a Python object.
std::shared_ptr<Variable> get_variable_from_py_object(const py::object& obj) {
    try {
        return obj.cast<std::shared_ptr<Variable>>();
    } catch (const py::cast_error& e) {
        throw AxeException("vmap arguments must be Variables, but got an unsupported type.");
    }
}

py::object vmap_impl(const py::function& fn, const py::tuple& args, const py::object& in_axes_obj, const py::object& out_axes_obj) {
    size_t num_args = args.size();
    auto in_axes = parse_axes(in_axes_obj, num_args);

    // Determine batch size
    size_t batch_size = 0;
    bool batch_size_set = false;
    for (size_t i = 0; i < num_args; ++i) {
        if (in_axes[i].has_value()) {
            auto arg_var = get_variable_from_py_object(args[i]);
            int axis = in_axes[i].value();
            if (axis < 0) axis += arg_var->data.shape().size();

            if (static_cast<size_t>(axis) >= arg_var->data.shape().size()) {
                throw AxeException("in_axes value is out of bounds for tensor shape.");
            }

            if (!batch_size_set) {
                batch_size = arg_var->data.shape()[axis];
                batch_size_set = true;
            } else if (arg_var->data.shape()[axis] != batch_size) {
                throw AxeException("Inconsistent batch sizes across vmapped arguments.");
            }
        }
    }

    if (!batch_size_set) { // No axes were mapped
        return fn(*args);
    }

    // Main loop
    std::vector<py::object> results;
    results.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        py::tuple sliced_args(num_args);
        for (size_t j = 0; j < num_args; ++j) {
            if (in_axes[j].has_value()) {
                auto arg_var = get_variable_from_py_object(args[j]);
                int axis = in_axes[j].value();
                if (axis < 0) axis += arg_var->data.shape().size();
                sliced_args[j] = py::cast(slice(arg_var, axis, i));
            } else {
                sliced_args[j] = args[j];
            }
        }
        results.push_back(fn(*sliced_args));
    }

    if (results.empty()) {
        return py::none();
    }

    // Stack results
    int out_axis = out_axes_obj.cast<int>();

    try {
        // Try to cast the first result to see if it's a single Variable
        auto first_res_var = py::cast<std::shared_ptr<Variable>>(results[0]);

        std::vector<std::shared_ptr<Variable>> var_results;
        var_results.reserve(results.size());
        var_results.push_back(first_res_var);
        for (size_t i = 1; i < results.size(); ++i) {
            var_results.push_back(py::cast<std::shared_ptr<Variable>>(results[i]));
        }
        return py::cast(stack(var_results, out_axis));

    } catch (const py::cast_error& e) {
        // If it's not a single Variable, check if it's a tuple
        try {
            auto first_res_tuple = py::cast<py::tuple>(results[0]);
            size_t num_outputs = first_res_tuple.size();
            py::tuple final_results(num_outputs);

            for (size_t i = 0; i < num_outputs; ++i) {
                std::vector<std::shared_ptr<Variable>> var_results;
                var_results.reserve(results.size());
                for (const auto& res_tuple : results) {
                    var_results.push_back(py::cast<std::shared_ptr<Variable>>(res_tuple.cast<py::tuple>()[i]));
                }
                final_results[i] = py::cast(stack(var_results, out_axis));
            }
            return final_results;
        } catch (const py::cast_error& e2) {
             // If it's not a tuple either, throw the error.
             throw AxeException("vmap output must be a Variable or a tuple of Variables.");
        }
    }
}

} // namespace vmap_internal
} // namespace axe