#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "include/tensor.h"
#include "include/variable.h"
#include "include/op.h"
#include "include/autodiff.h"
#include "include/jit.h"

namespace py = pybind11;
using namespace axe;

size_t dtype_size(DType dtype); // Forward declaration
std::string format_descriptor(DType dtype); // Forward declaration
PYBIND11_MODULE(_axe, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);
    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int32", DType::Int32)
        .value("Int64", DType::Int64);

    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init<const std::vector<size_t>&, DType, Device>(),
             py::arg("shape"), py::arg("dtype"), py::arg("device") = Device::CPU)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("device", &Tensor::device)
        .def_buffer([](Tensor &t) -> py::buffer_info {
            py::ssize_t itemsize = dtype_size(t.dtype());
            const auto& shape_size_t = t.shape();
            std::vector<py::ssize_t> shape(shape_size_t.begin(), shape_size_t.end());

            std::vector<py::ssize_t> strides(shape.size());
            if (!shape.empty()) {
                strides.back() = itemsize;
                for (int i = shape.size() - 2; i >= 0; --i) {
                    strides[i] = strides[i+1] * shape[i+1];
                }
            }

            return py::buffer_info(
                t.data(),                               /* Pointer to buffer */
                itemsize,                               /* Size of one scalar */
                format_descriptor(t.dtype()),           /* Python struct-style format descriptor */
                shape.size(),                           /* Number of dimensions */
                shape,                                  /* Buffer dimensions */
                strides                                 /* Strides */
            );
        })
        .def_static("zeros", &Tensor::zeros)
        .def_static("ones", &Tensor::ones)
        .def_static("arange", &Tensor::arange, py::arg("start"), py::arg("end"), py::arg("dtype"), py::arg("device") = Device::CPU)
        .def("__add__", [](const Tensor& a, py::object b_obj) {
            if (py::isinstance<Tensor>(b_obj)) {
                return py::cast(a.add(b_obj.cast<Tensor>()));
            } else if (py::isinstance<Variable>(b_obj)) {
                auto a_var = std::make_shared<Variable>(a, false);
                auto b_var = b_obj.cast<std::shared_ptr<Variable>>();
                return py::cast(add(a_var, b_var));
            }
            throw py::type_error("Unsupported type for __add__");
        })
        .def("__sub__", [](const Tensor& a, py::object b_obj) {
            if (py::isinstance<Tensor>(b_obj)) {
                return py::cast(a.sub(b_obj.cast<Tensor>()));
            } else if (py::isinstance<Variable>(b_obj)) {
                auto a_var = std::make_shared<Variable>(a, false);
                auto b_var = b_obj.cast<std::shared_ptr<Variable>>();
                return py::cast(sub(a_var, b_var));
            }
            throw py::type_error("Unsupported type for __sub__");
        })
        .def("__mul__", [](const Tensor& a, py::object b_obj) {
            if (py::isinstance<Tensor>(b_obj)) {
                return py::cast(a.mul(b_obj.cast<Tensor>()));
            } else if (py::isinstance<Variable>(b_obj)) {
                auto a_var = std::make_shared<Variable>(a, false);
                auto b_var = b_obj.cast<std::shared_ptr<Variable>>();
                return py::cast(mul(a_var, b_var));
            }
            throw py::type_error("Unsupported type for __mul__");
        })
        .def("__truediv__", &Tensor::div)
        .def("__matmul__", [](const Tensor& a, py::object b_obj) {
            if (py::isinstance<Tensor>(b_obj)) {
                return py::cast(a.matmul(b_obj.cast<Tensor>()));
            } else if (py::isinstance<Variable>(b_obj)) {
                auto a_var = std::make_shared<Variable>(a, false);
                auto b_var = b_obj.cast<std::shared_ptr<Variable>>();
                return py::cast(matmul(a_var, b_var));
            }
            throw py::type_error("Unsupported type for __matmul__");
        })
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("max", &Tensor::max)
        .def("numpy", [](Tensor& t) -> py::array {
            const auto& shape_size_t = t.shape();
            std::vector<py::ssize_t> shape(shape_size_t.begin(), shape_size_t.end());
            py::ssize_t itemsize = dtype_size(t.dtype());

            std::vector<py::ssize_t> strides(shape.size());
            if (!shape.empty()) {
                strides.back() = itemsize;
                for (int i = shape.size() - 2; i >= 0; --i) {
                    strides[i] = strides[i+1] * shape[i+1];
                }
            }

            return py::array(py::buffer_info(
                t.data(),
                itemsize,
                format_descriptor(t.dtype()),
                shape.size(),
                shape,
                strides
            ));
        });

    py::class_<Variable, std::shared_ptr<Variable>>(m, "Variable")
        .def(py::init<const Tensor&, bool>(), py::arg("data"), py::arg("requires_grad") = false)
        .def_readwrite("data", &Variable::data)
        .def_readwrite("requires_grad", &Variable::requires_grad)
        .def_property("grad",
            [](Variable &v) -> py::object {
                if (v.grad) {
                    return py::cast(*v.grad);
                }
                return py::none();
            },
            [](Variable &v, const Tensor &t) {
                v.grad = std::make_shared<Tensor>(t);
            })
        .def("backward", &Variable::backward)
        .def("__add__", [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) -> py::object {
            if (!axe::grad_enabled) {
                return py::cast(a->data.add(b->data));
            }
            return py::cast(add(a, b));
        })
        .def("__mul__", [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) -> py::object {
            if (!axe::grad_enabled) {
                return py::cast(a->data.mul(b->data));
            }
            return py::cast(mul(a, b));
        })
        .def("__matmul__", [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) -> py::object {
            if (!axe::grad_enabled) { return py::cast(a->data.matmul(b->data)); }
            return py::cast(matmul(a, b));
        })
        .def("__sub__", [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) -> py::object {
            if (!axe::grad_enabled) { return py::cast(a->data.sub(b->data)); }
            return py::cast(sub(a, b));
        })
        // Mixed-type operations: Variable op Tensor
        .def("__add__", [](std::shared_ptr<Variable> a, const Tensor& b) -> py::object {
            auto b_var = std::make_shared<Variable>(b, false);
            if (!axe::grad_enabled || !a->requires_grad) { return py::cast(a->data.add(b)); }
            return py::cast(add(a, b_var));
        })
        .def("__sub__", [](std::shared_ptr<Variable> a, const Tensor& b) -> py::object {
            auto b_var = std::make_shared<Variable>(b, false);
            if (!axe::grad_enabled || !a->requires_grad) { return py::cast(a->data.sub(b)); }
            return py::cast(sub(a, b_var));
        })
        .def("__mul__", [](std::shared_ptr<Variable> a, const Tensor& b) -> py::object {
            auto b_var = std::make_shared<Variable>(b, false);
            if (!axe::grad_enabled || !a->requires_grad) { return py::cast(a->data.mul(b)); }
            return py::cast(mul(a, b_var));
        })
        .def("__matmul__", [](std::shared_ptr<Variable> a, const Tensor& b) -> py::object {
            auto b_var = std::make_shared<Variable>(b, false);
            if (!axe::grad_enabled || !a->requires_grad) { return py::cast(a->data.matmul(b)); }
            return py::cast(matmul(a, b_var));
        })
        // Mixed-type operations: Tensor op Variable
        .def("__radd__", [](std::shared_ptr<Variable> a, const Tensor& b) -> py::object {
            auto b_var = std::make_shared<Variable>(b, false);
            if (!axe::grad_enabled || !a->requires_grad) { return py::cast(b.add(a->data)); }
            return py::cast(add(b_var, a));
        })
        .def("__rsub__", [](std::shared_ptr<Variable> a, const Tensor& b) -> py::object {
            auto b_var = std::make_shared<Variable>(b, false);
            if (!axe::grad_enabled || !a->requires_grad) { return py::cast(b.sub(a->data)); }
            return py::cast(sub(b_var, a));
        })
        .def("__rmul__", [](std::shared_ptr<Variable> a, const Tensor& b) -> py::object {
            auto b_var = std::make_shared<Variable>(b, false);
            if (!axe::grad_enabled || !a->requires_grad) { return py::cast(b.mul(a->data)); }
            return py::cast(mul(b_var, a));
        })
        .def("__rmatmul__", [](std::shared_ptr<Variable> a, const Tensor& b) -> py::object {
            auto b_var = std::make_shared<Variable>(b, false);
            if (!axe::grad_enabled || !a->requires_grad) { return py::cast(b.matmul(a->data)); }
            return py::cast(matmul(b_var, a));
        })
        .def("sum", &sum);

    m.def("_value_and_grad_cpp_multi", [](py::function fn, py::args py_args) {
        bool any_requires_grad = false;
        for (const auto& arg_handle : py_args) {
            try {
                auto var = arg_handle.cast<std::shared_ptr<Variable>>();
                if (var->requires_grad) {
                    any_requires_grad = true;
                    break;
                }
            } catch (py::cast_error& e) {}
        }

        py::object result = fn(*py_args);
        auto result_var = result.cast<std::shared_ptr<Variable>>();

        if (any_requires_grad && result_var->requires_grad) {
            result_var->backward();
        }

        py::list all_grads;
        for (const auto& arg_handle : py_args) {
            try {
                auto var = arg_handle.cast<std::shared_ptr<Variable>>();
                if (var->grad) {
                    all_grads.append(py::cast(*var->grad));
                } else {
                    all_grads.append(py::none());
                }
            } catch (py::cast_error &e) {
                all_grads.append(py::none());
            }
        }

        return std::make_pair(result_var->data, all_grads);
    });

    m.def("is_grad_enabled", []() { return axe::grad_enabled; });
    m.def("set_grad_enabled", [](bool enabled) { axe::grad_enabled = enabled; });

    // --- JIT Bindings ---
    auto jit_module = m.def_submodule("jit", "JIT compiler functionality");

    py::enum_<jit::OpType>(jit_module, "OpType")
        .value("Add", jit::OpType::Add)
        .value("Sub", jit::OpType::Sub)
        .value("Mul", jit::OpType::Mul)
        .value("Div", jit::OpType::Div)
        .value("MatMul", jit::OpType::MatMul)
        .value("Sum", jit::OpType::Sum);

    py::class_<jit::Node>(jit_module, "Node")
        .def_readonly("id", &jit::Node::id);

    py::class_<jit::TraceableOp>(jit_module, "TraceableOp")
        .def_readonly("type", &jit::TraceableOp::type)
        .def_readonly("inputs", &jit::TraceableOp::inputs)
        .def_readonly("output", &jit::TraceableOp::output);

    py::class_<jit::JitGraph, std::shared_ptr<jit::JitGraph>>(jit_module, "JitGraph")
        .def("get_inputs", &jit::JitGraph::get_inputs, py::return_value_policy::reference)
        .def("get_ops", &jit::JitGraph::get_ops, py::return_value_policy::reference)
        .def("get_output_node", &jit::JitGraph::get_output_node, py::return_value_policy::reference)
        .def("execute", &jit::JitGraph::execute, "Executes the compiled graph with given inputs.")
        .def("__repr__", [](const jit::JitGraph &g) {
            std::stringstream ss;
            ss << "<JitGraph with " << g.get_inputs().size() << " inputs, "
               << g.get_ops().size() << " ops>";
            return ss.str();
        });

    jit_module.def("start_tracing", &jit::start_tracing, "Starts JIT tracing.");
    jit_module.def("stop_tracing", &jit::stop_tracing, "Stops JIT tracing and returns the graph.");
    jit_module.def("register_input", &jit::register_input, "Registers a variable as a placeholder input for the trace.");
}

// Helper to get pybind11 format descriptor string for a DType
std::string format_descriptor(DType dtype) {
    switch (dtype) {
        case DType::Float32: return py::format_descriptor<float>::format();
        case DType::Float64: return py::format_descriptor<double>::format();
        case DType::Int32:   return py::format_descriptor<int32_t>::format();
        case DType::Int64:   return py::format_descriptor<int64_t>::format();
    }
    throw std::runtime_error("Unsupported dtype");
}

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Int32:   return sizeof(int32_t);
        case DType::Int64:   return sizeof(int64_t);
    }
    throw std::runtime_error("Unsupported dtype");
}
