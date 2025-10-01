#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tensor.h"

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
        .def("__add__", &Tensor::add)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", &Tensor::mul)
        .def("__truediv__", &Tensor::div)
        .def("__matmul__", &Tensor::matmul)
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

            // Create a py::array from the buffer_info. This copies the data.
            return py::array(py::buffer_info(
                t.data(),
                itemsize,
                format_descriptor(t.dtype()),
                shape.size(),
                shape,
                strides
            ));
        });
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