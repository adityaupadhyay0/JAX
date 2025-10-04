#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "include/tensor.h"
#include "include/variable.h"
#include "include/op.h"
#include "include/autodiff.h"
#include "include/jit.h"
#include "include/jit_engine.h"
#include "include/exception.h"
#include "include/vmap.h"

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

    py::register_exception<AxeException>(m, "AxeError");

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
            std::vector<py::ssize_t> strides;
            if (!shape.empty()) {
                strides.resize(shape.size());
                strides.back() = itemsize;
                for (int i = shape.size() - 2; i >= 0; --i) {
                    strides[i] = strides[i+1] * shape[i+1];
                }
            }
            return py::buffer_info(t.data(), itemsize, format_descriptor(t.dtype()), shape.size(), shape, strides);
        })
        .def_static("zeros", &Tensor::zeros)
        .def_static("ones", &Tensor::ones)
        .def_static("arange", &Tensor::arange, py::arg("start"), py::arg("end"), py::arg("dtype"), py::arg("device") = Device::CPU)
        .def_static("stack", &Tensor::stack, py::arg("tensors"), py::arg("dim") = 0)
        .def("__add__", &Tensor::add)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", &Tensor::mul)
        .def("__truediv__", &Tensor::div)
        .def("__matmul__", &Tensor::matmul)
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("max", &Tensor::max)
        .def("transpose", &Tensor::transpose)
        .def("reshape", &Tensor::reshape, py::arg("new_shape"))
        .def("slice", &Tensor::slice, py::arg("dim"), py::arg("index"))
        .def("numpy", [](Tensor& t) -> py::array {
            const auto& shape_size_t = t.shape();
            std::vector<py::ssize_t> shape(shape_size_t.begin(), shape_size_t.end());
            py::ssize_t itemsize = dtype_size(t.dtype());
            std::vector<py::ssize_t> strides;
            if (!shape.empty()) {
                strides.resize(shape.size());
                strides.back() = itemsize;
                for (int i = shape.size() - 2; i >= 0; --i) {
                    strides[i] = strides[i+1] * shape[i+1];
                }
            }
            return py::array(py::buffer_info(t.data(), itemsize, format_descriptor(t.dtype()), shape.size(), shape, strides));
        });

    py::class_<Variable, std::shared_ptr<Variable>>(m, "Variable")
        .def(py::init<const Tensor&, bool>(), py::arg("data"), py::arg("requires_grad") = false)
        .def_readwrite("data", &Variable::data)
        .def_readwrite("requires_grad", &Variable::requires_grad)
        .def_property("grad",
            [](Variable &v) -> py::object {
                if (v.grad) return py::cast(*v.grad);
                return py::none();
            },
            [](Variable &v, py::object grad_obj) {
                if (grad_obj.is_none()) {
                    v.grad = nullptr;
                } else {
                    v.grad = std::make_shared<Tensor>(grad_obj.cast<Tensor>());
                }
            })
        .def("backward", &Variable::backward);

    m.def("_add", &add, py::arg("a"), py::arg("b"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_sub", &sub, py::arg("a"), py::arg("b"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_mul", &mul, py::arg("a"), py::arg("b"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_matmul", &matmul, py::arg("a"), py::arg("b"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_sum", &sum, py::arg("a"), py::arg("file") = "", py::arg("line") = 0);

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

    m.def("_vmap_impl", &axe::vmap_internal::vmap_impl,
          py::arg("fn"), py::arg("args"), py::arg("in_axes"), py::arg("out_axes"),
          "C++ implementation of vmap");

    auto jit_module = m.def_submodule("jit", "JIT compiler functionality");
    jit_module.def("is_tracing", [](){ return jit::is_tracing; });
    jit_module.def("start_tracing", &jit::start_tracing, "Starts JIT tracing.");
    jit_module.def("stop_tracing", &jit::stop_tracing, "Stops JIT tracing and returns the graph.");
    jit_module.def("register_input", &jit::register_input, "Registers a variable as a placeholder input for the trace.");
    jit_module.def("jit_execute_with_engine", [](std::shared_ptr<jit::JitGraph> graph, const std::vector<Tensor>& inputs) {
        jit::initialize_jit_engine();
        graph->optimize();
        jit::CompiledFunction func = jit::global_engine->get_or_compile(*graph);
        if (!func) throw std::runtime_error("JIT engine failed to compile the function.");
        return func(inputs);
    }, "Compiles and executes a JitGraph using the dynamic C++ backend.");

    py::class_<jit::JitGraph, std::shared_ptr<jit::JitGraph>>(jit_module, "JitGraph")
        .def(py::init<>())
        .def("get_inputs", &jit::JitGraph::get_inputs, py::return_value_policy::reference)
        .def("get_ops", &jit::JitGraph::get_ops, py::return_value_policy::reference)
        .def("get_output_node", &jit::JitGraph::get_output_node, py::return_value_policy::reference)
        .def("optimize", &jit::JitGraph::optimize, "Runs the optimization pipeline on the graph.")
        .def("execute", &jit::JitGraph::execute, "Executes the compiled graph with given inputs.")
        .def("__repr__", [](const jit::JitGraph &g) {
            std::stringstream ss;
            ss << "<JitGraph with " << g.get_inputs().size() << " inputs, "
               << g.get_ops().size() << " ops>";
            return ss.str();
        });
}

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