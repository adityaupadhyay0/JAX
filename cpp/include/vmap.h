#pragma once
#include <pybind11/pybind11.h>
#include "tensor.h"
#include "variable.h"

namespace py = pybind11;

namespace axe {
namespace vmap_internal {

// Main C++ implementation for vmap
py::object vmap_impl(
    const py::function& fn,
    const py::tuple& args,
    const py::object& in_axes,
    const py::object& out_axes
);

} // namespace vmap_internal
} // namespace axe