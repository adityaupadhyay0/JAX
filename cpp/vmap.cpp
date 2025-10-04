#include "include/vmap.h"
#include "include/tensor.h"
#include "include/variable.h"
#include "include/exception.h"
#include <vector>
#include <iostream>

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


// Helper to extract a Tensor from a Python object, whether it's a Tensor or Variable.
Tensor get_tensor_from_py_object(const py::object& obj) {
    if (py::isinstance<Tensor>(obj)) {
        return obj.cast<Tensor>();
    }
    if (py::isinstance<std::shared_ptr<Variable>>(obj)) {
        return obj.cast<std::shared_ptr<Variable>>()->data;
    }
    // Attempt to cast to a variable to provide a better error message.
    try {
        obj.cast<std::shared_ptr<Variable>>();
    } catch (const py::cast_error& e) {
        throw AxeException("vmap arguments must be Tensors or Variables, but got an unsupported type.");
    }
    // This part should not be reached if the above try-catch works as expected.
    throw AxeException("vmap arguments must be Tensors or Variables.");
}


py::object vmap_impl(const py::function& fn, const py::tuple& args, const py::object& in_axes_obj, const py::object& out_axes_obj) {
    size_t num_args = args.size();
    auto in_axes = parse_axes(in_axes_obj, num_args);

    // Determine batch size
    size_t batch_size = 0;
    bool batch_size_set = false;
    for (size_t i = 0; i < num_args; ++i) {
        if (in_axes[i].has_value()) {
            Tensor arg_tensor = get_tensor_from_py_object(args[i]);
            int axis = in_axes[i].value();
            if (axis < 0) axis += arg_tensor.shape().size(); // Handle negative axis

            if (static_cast<size_t>(axis) >= arg_tensor.shape().size()) {
                throw AxeException("in_axes value is out of bounds for tensor shape.");
            }

            if (!batch_size_set) {
                batch_size = arg_tensor.shape()[axis];
                batch_size_set = true;
            } else if (arg_tensor.shape()[axis] != batch_size) {
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
                Tensor arg_tensor = get_tensor_from_py_object(args[j]);
                int axis = in_axes[j].value();
                if (axis < 0) axis += arg_tensor.shape().size();
                // We always slice the underlying tensor.
                // The python function `fn` will receive a Tensor, not a Variable.
                // This is sufficient for `grad` to work, as the VJP is handled by python.
                sliced_args[j] = py::cast(arg_tensor.slice(axis, i));
            } else {
                sliced_args[j] = args[j]; // Pass through non-mapped args
            }
        }
        results.push_back(fn(*sliced_args));
    }

    if (results.empty()) {
        return py::none();
    }

    // Stack results
    int out_axis = out_axes_obj.cast<int>();
    if (py::isinstance<Tensor>(results[0])) {
        std::vector<Tensor> tensor_results;
        tensor_results.reserve(results.size());
        for (const auto& res : results) {
            tensor_results.push_back(res.cast<Tensor>());
        }
        return py::cast(Tensor::stack(tensor_results, out_axis));
    } else if (py::isinstance<py::tuple>(results[0])) {
        size_t num_outputs = py::len(results[0]);
        py::tuple final_results(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            std::vector<Tensor> tensor_results;
            tensor_results.reserve(results.size());
            for (const auto& res_tuple : results) {
                tensor_results.push_back(res_tuple.cast<py::tuple>()[i].cast<Tensor>());
            }
            final_results[i] = Tensor::stack(tensor_results, out_axis);
        }
        return final_results;
    }

    throw AxeException("vmap output must be a Tensor or a tuple of Tensors.");
}

} // namespace vmap_internal
} // namespace axe