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
#include "include/allocator.h"

namespace py = pybind11;
using namespace axe;


// --- Checkpointing Implementation ---
// This is defined here, in the binding layer, to avoid making the core C++
// library dependent on Python headers.

class CheckpointOp : public Operation {
public:
    py::function func;

    CheckpointOp(const py::function& fn, const std::vector<std::shared_ptr<Variable>>& inputs) : func(fn) {
        this->inputs = inputs;
    }

    void backward(const Tensor& grad_output) override {
        bool original_grad_state = grad_enabled;
        grad_enabled = true;

        py::tuple py_args(inputs.size());
        for(size_t i = 0; i < inputs.size(); ++i) {
            inputs[i]->requires_grad = true;
            py_args[i] = py::cast(inputs[i]);
        }

        py::object result_obj = func(*py_args);
        auto recomputed_output = py::cast<std::shared_ptr<Variable>>(result_obj);

        recomputed_output->grad = std::make_shared<Tensor>(grad_output);
        recomputed_output->backward();

        grad_enabled = original_grad_state;
    }
};

std::shared_ptr<Variable> checkpoint(const py::function& fn, const std::vector<std::shared_ptr<Variable>>& inputs) {
    bool original_grad_state = grad_enabled;
    grad_enabled = false;

    py::tuple py_args(inputs.size());
    for(size_t i = 0; i < inputs.size(); ++i) {
        py_args[i] = py::cast(inputs[i]);
    }
    py::object result_obj = fn(*py_args);
    auto output_var_no_grad = py::cast<std::shared_ptr<Variable>>(result_obj);

    grad_enabled = original_grad_state;

    bool any_input_requires_grad = false;
    for (const auto& var : inputs) {
        if (var->requires_grad) {
            any_input_requires_grad = true;
            break;
        }
    }

    bool result_requires_grad = grad_enabled && any_input_requires_grad;
    auto result = std::make_shared<Variable>(output_var_no_grad->data, result_requires_grad);

    if (result_requires_grad) {
        result->creator = std::make_shared<CheckpointOp>(fn, inputs);
    }

    return result;
}


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
    py::register_exception<OOMError>(m, "OOMError", m.attr("AxeError"));

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

    py::class_<Variable, std::shared_ptr<Variable>>(m, "Variable", py::buffer_protocol())
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
        .def("backward", &Variable::backward)
        // Delegate tensor properties and methods to make Variable behave like a Tensor
        .def_property_readonly("shape", [](const Variable& v) { return v.data.shape(); })
        .def_property_readonly("dtype", [](const Variable& v) { return v.data.dtype(); })
        .def_property_readonly("device", [](const Variable& v) { return v.data.device(); })
        .def("reshape", [](const Variable& v, const std::vector<size_t>& new_shape) {
            return Variable(v.data.reshape(new_shape));
         })
        .def("transpose", [](const Variable& v) {
            return Variable(v.data.transpose());
        })
        .def("slice", [](const Variable& v, size_t dim, size_t index) {
            return Variable(v.data.slice(dim, index));
        })
        .def("numpy", [](Variable& v) -> py::array {
            const auto& shape_size_t = v.data.shape();
            std::vector<py::ssize_t> shape(shape_size_t.begin(), shape_size_t.end());
            py::ssize_t itemsize = dtype_size(v.data.dtype());
            std::vector<py::ssize_t> strides;
            if (!shape.empty()) {
                strides.resize(shape.size());
                strides.back() = itemsize;
                for (int i = shape.size() - 2; i >= 0; --i) {
                    strides[i] = strides[i+1] * shape[i+1];
                }
            }
            return py::array(py::buffer_info(v.data.data(), itemsize, format_descriptor(v.data.dtype()), shape.size(), shape, strides));
        })
        .def("__repr__", [](const Variable& v) {
            std::stringstream ss;
            ss << "Variable(shape=";
            const auto& shape = v.data.shape();
            ss << "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
            }
            ss << "], requires_grad=" << (v.requires_grad ? "True" : "False") << ")";
            return ss.str();
        })
        .def_buffer([](Variable &v) -> py::buffer_info {
            return py::buffer_info(
                v.data.data(),
                dtype_size(v.data.dtype()),
                format_descriptor(v.data.dtype()),
                v.data.shape().size(),
                v.data.shape(),
                [&]{
                    std::vector<py::ssize_t> strides(v.data.shape().size());
                    if (!v.data.shape().empty()) {
                        strides.back() = dtype_size(v.data.dtype());
                        for (int i = v.data.shape().size() - 2; i >= 0; --i) {
                            strides[i] = strides[i+1] * v.data.shape()[i+1];
                        }
                    }
                    return strides;
                }()
            );
        });

    m.def("_add", &add, py::arg("a"), py::arg("b"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_sub", &sub, py::arg("a"), py::arg("b"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_mul", &mul, py::arg("a"), py::arg("b"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_matmul", &matmul, py::arg("a"), py::arg("b"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_sum", &sum, py::arg("a"), py::arg("file") = "", py::arg("line") = 0);
    m.def("_stack", &stack, py::arg("inputs"), py::arg("dim") = 0, "Create a stack operation.");

    m.def("_checkpoint", &checkpoint, py::arg("fn"), py::arg("inputs"), "Create a gradient checkpoint.");

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

    auto mem_module = m.def_submodule("memory", "Memory management utilities");
    py::class_<memory::AllocatorStats>(mem_module, "AllocatorStats")
        .def_readonly("allocated_bytes", &memory::AllocatorStats::allocated_bytes)
        .def_readonly("peak_bytes", &memory::AllocatorStats::peak_bytes)
        .def_readonly("cached_bytes", &memory::AllocatorStats::cached_bytes)
        .def("__repr__", [](const memory::AllocatorStats &stats) {
            return "<AllocatorStats allocated=" + std::to_string(stats.allocated_bytes) +
                   " peak=" + std::to_string(stats.peak_bytes) +
                   " cached=" + std::to_string(stats.cached_bytes) + ">";
        });

    py::enum_<memory::MemoryEventType>(mem_module, "MemoryEventType")
        .value("ALLOCATE", memory::MemoryEventType::ALLOCATE)
        .value("DEALLOCATE", memory::MemoryEventType::DEALLOCATE)
        .value("FREE_CACHE", memory::MemoryEventType::FREE_CACHE);

    py::class_<memory::MemoryEvent>(mem_module, "MemoryEvent")
        .def_readonly("type", &memory::MemoryEvent::type)
        .def_readonly("size_bytes", &memory::MemoryEvent::size_bytes)
        .def_readonly("timestamp", &memory::MemoryEvent::timestamp)
        .def_readonly("allocated_bytes_after", &memory::MemoryEvent::allocated_bytes_after)
        .def_readonly("cached_bytes_after", &memory::MemoryEvent::cached_bytes_after)
        .def("__repr__", [](const memory::MemoryEvent &event) {
            return "<MemoryEvent type=" + std::string(py::str(py::cast(event.type))) +
                   " size=" + std::to_string(event.size_bytes) + ">";
        });

    mem_module.def("get_memory_timeline", [](Device dev) {
        return memory::Allocator::get_instance().get_memory_timeline(dev);
    }, py::arg("device") = Device::CPU, "Get the timeline of memory events.");

    mem_module.def("clear_memory_timeline", [](Device dev) {
        memory::Allocator::get_instance().clear_memory_timeline(dev);
    }, py::arg("device") = Device::CPU, "Clear the memory event timeline.");

    mem_module.def("get_stats", [](Device dev) {
        return memory::Allocator::get_instance().get_stats(dev);
    }, py::arg("device") = Device::CPU, "Get memory stats for a device.");

    mem_module.def("reset_peak_bytes", [](Device dev) {
        memory::Allocator::get_instance().reset_peak_bytes(dev);
    }, py::arg("device") = Device::CPU, "Reset peak memory counter for a device.");

    mem_module.def("allocated_bytes", [](Device dev) {
        return memory::Allocator::get_instance().get_stats(dev).allocated_bytes;
    }, py::arg("device") = Device::CPU, "Get current allocated bytes for a device.");

    mem_module.def("peak_bytes", [](Device dev) {
        return memory::Allocator::get_instance().get_stats(dev).peak_bytes;
    }, py::arg("device") = Device::CPU, "Get peak allocated bytes for a device.");

    mem_module.def("cached_bytes", [](Device dev) {
        return memory::Allocator::get_instance().get_stats(dev).cached_bytes;
    }, py::arg("device") = Device::CPU, "Get current cached bytes for a device.");

    mem_module.def("debug_clear_everything", []() {
        memory::Allocator::get_instance().debug_clear_everything();
    }, "FOR TESTING ONLY: Clears all cached memory and resets stats.");
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