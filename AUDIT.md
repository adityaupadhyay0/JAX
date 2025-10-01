# AXE Project Audit: Phase 1 & 2

This document contains the audit of the AXE project, focusing on the completion status of tasks outlined in Phase 1 and Phase 2 of the `todo.md` file.

---

##  PHASE 1: Basic Tensor Ops

### Setup
- **Create GitHub repo**: ❓ (Cannot verify from the codebase)
- **Setup CMake build**: ✅ (Verified in `CMakeLists.txt` and `cpp/CMakeLists.txt`)
- **Add pybind11 dependency**: ✅ (Verified in `cpp/CMakeLists.txt` using `FetchContent`)
- **Configure GitHub Actions CI**: ❌ (No `.github` directory found, indicating CI is not set up)
- **Add pytest + Google Test**: ⚠️ (`pytest` is in `requirements.txt`, but no Google Test integration was found in the C++ build configuration)

### C++ Core
- **Create `Tensor` class with shape/dtype**: ✅ (Implemented in `cpp/include/tensor.h`)
- **Implement reference counting**: ✅ (A basic implementation using a pointer to a count is present in `cpp/tensor.cpp`)
- **Add CPU/GPU device enum**: ✅ (Implemented in `cpp/include/tensor.h`, but GPU functionality is not supported)
- **Write basic memory allocator**: ⚠️ (The implementation uses `malloc` and `free` directly. This is functional but not a custom or optimized allocator)
- **Add `zeros()`, `ones()`, `arange()`**: ✅ (Implemented in `cpp/tensor.cpp`)

### Python API
- **Wrap Tensor class with pybind11**: ✅ (The `Tensor` class is wrapped in `cpp/pybind.cpp`)
- **Support NumPy array protocol**: ✅ (The buffer protocol is implemented in `cpp/pybind.cpp`, allowing for NumPy interoperability)
- **Add `axe.array()` function**: ✅ (Implemented in `python/axe.py`)
- **Enable `.numpy()` conversion**: ⚠️ (Conversion is possible via `np.array(t)`, but a dedicated `.numpy()` method is not present)

### Test
- **Can create tensors on CPU**: ⏳ (To be verified by running tests)
- **Can create tensors on GPU**: ❌ (The constructor in `cpp/tensor.cpp` explicitly throws an error for GPU devices)
- **NumPy interop works**: ⏳ (To be verified by running tests)

---

## PHASE 2: Basic Math Ops

### C++ Implementation
- **Add `add`, `sub`, `mul`, `div`**: ✅ (Basic element-wise implementations are present in `cpp/tensor.cpp`)
- **Add `matmul` using Eigen**: ❌ (Not implemented, and the Eigen dependency is not included)
- **Add `sum`, `mean`, `max`**: ❌ (Not implemented)
- **Add broadcasting logic**: ❌ (Element-wise operations require exact shape matches)
- **Write CUDA kernels for GPU**: ❌ (No GPU support or CUDA code found)

### Python Operators
- **Override `__add__`, `__mul__`, etc**: ✅ (The `__add__`, `__sub__`, `__mul__`, and `__truediv__` operators are bound in `cpp/pybind.cpp`)
- **Support `@` for matmul**: ❌ (The `__matmul__` operator is not implemented or bound)
- **Add `.sum()`, `.mean()` methods**: ❌ (Not implemented)

### Test
- **All ops work on CPU**: ⏳ (To be verified by running tests)
- **All ops work on GPU**: ❌ (No GPU support)
- **Broadcasting works correctly**: ❌ (No broadcasting logic implemented)
- **Benchmark vs NumPy**: ❌ (No benchmark tests found in the repository)