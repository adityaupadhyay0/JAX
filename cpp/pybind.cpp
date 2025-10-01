#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tensor.h"

namespace py = pybind11;
using namespace axe;

PYBIND11_MODULE(_axe, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);
    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int32", DType::Int32)
        .value("Int64", DType::Int64);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&, DType, Device>(),
             py::arg("shape"), py::arg("dtype"), py::arg("device") = Device::CPU)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly("data", [](Tensor &t) { return reinterpret_cast<uintptr_t>(t.data()); })
        .def("__array__", [](const Tensor& t) {
            py::dtype dt;
            if (t.dtype() == DType::Float32) dt = py::dtype::of<float>();
            else if (t.dtype() == DType::Float64) dt = py::dtype::of<double>();
            else if (t.dtype() == DType::Int32) dt = py::dtype::of<int32_t>();
            else if (t.dtype() == DType::Int64) dt = py::dtype::of<int64_t>();
            else throw std::runtime_error("Unsupported dtype");
            return py::array(dt, t.shape(), t.data());
        })
        .def("numpy", [](const Tensor& t) {
            py::dtype dt;
            if (t.dtype() == DType::Float32) dt = py::dtype::of<float>();
            else if (t.dtype() == DType::Float64) dt = py::dtype::of<double>();
            else if (t.dtype() == DType::Int32) dt = py::dtype::of<int32_t>();
            else if (t.dtype() == DType::Int64) dt = py::dtype::of<int64_t>();
            else throw std::runtime_error("Unsupported dtype");
            return py::array(dt, t.shape(), t.data()).attr("copy")();
        })
        .def_static("zeros", &Tensor::zeros)
        .def_static("ones", &Tensor::ones)
        .def_static("arange", &Tensor::arange)
        .def("__add__", &Tensor::add)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", &Tensor::mul)
        .def("__truediv__", &Tensor::div);
}
