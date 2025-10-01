#include "tensor.h"
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <functional>

namespace axe {

// Helper to get size of a DType in bytes
size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
    }
    throw std::runtime_error("Unsupported dtype");
}

Tensor::Tensor(const std::vector<size_t>& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device), data_(nullptr), ref_count_ptr_(new size_t(1)) {
    if (device_ == Device::GPU) {
        throw std::runtime_error("GPU device support not yet implemented");
    }
    data_ = malloc(nbytes());
    if (!data_) throw std::bad_alloc();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_), data_(other.data_), ref_count_ptr_(other.ref_count_ptr_) {
    ++(*ref_count_ptr_);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (--(*ref_count_ptr_) == 0) {
            if (data_) free(data_);
            delete ref_count_ptr_;
        }
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = other.data_;
        ref_count_ptr_ = other.ref_count_ptr_;
        ++(*ref_count_ptr_);
    }
    return *this;
}

Tensor::~Tensor() {
    if (--(*ref_count_ptr_) == 0) {
        if (data_) free(data_);
        delete ref_count_ptr_;
    }
}

size_t Tensor::nbytes() const {
    size_t n = 1;
    for (auto d : shape_) n *= d;
    return n * dtype_size(dtype_);
}

size_t Tensor::nelement() const {
    size_t n = 1;
    for (auto d : shape_) n *= d;
    return n;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    std::memset(t.data(), 0, t.nbytes());
    return t;
}

Tensor Tensor::ones(const std::vector<size_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    size_t n = t.nelement();

    switch(dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(t.data());
            for (size_t i = 0; i < n; ++i) ptr[i] = 1.0f;
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(t.data());
            for (size_t i = 0; i < n; ++i) ptr[i] = 1.0;
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(t.data());
            for (size_t i = 0; i < n; ++i) ptr[i] = 1;
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(t.data());
            for (size_t i = 0; i < n; ++i) ptr[i] = 1;
            break;
        }
    }
    return t;
}

Tensor Tensor::arange(size_t start, size_t end, DType dtype, Device device) {
    size_t n = end > start ? end - start : 0;
    Tensor t({n}, dtype, device);
    switch(dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(t.data());
            for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<float>(start + i);
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(t.data());
            for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<double>(start + i);
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(t.data());
            for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<int32_t>(start + i);
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(t.data());
            for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<int64_t>(start + i);
            break;
        }
    }
    return t;
}

// Basic element-wise operations
template<typename T>
Tensor element_wise_op(const Tensor& a, const Tensor& b, std::function<T(T, T)> op) {
    if (a.shape() != b.shape() || a.dtype() != b.dtype()) {
        throw std::runtime_error("Mismatched shapes or dtypes for element-wise op");
    }
    Tensor result(a.shape(), a.dtype(), a.device());
    const T* a_ptr = static_cast<const T*>(a.data());
    const T* b_ptr = static_cast<const T*>(b.data());
    T* res_ptr = static_cast<T*>(result.data());
    size_t n = a.nelement();
    for (size_t i = 0; i < n; ++i) {
        res_ptr[i] = op(a_ptr[i], b_ptr[i]);
    }
    return result;
}

#define DISPATCH_DTYPE(dtype, op_name, op) \
    switch (dtype) { \
        case DType::Float32: return element_wise_op<float>(*this, other, op<float>); \
        case DType::Float64: return element_wise_op<double>(*this, other, op<double>); \
        case DType::Int32: return element_wise_op<int32_t>(*this, other, op<int32_t>); \
        case DType::Int64: return element_wise_op<int64_t>(*this, other, op<int64_t>); \
        default: throw std::runtime_error("Unsupported dtype for " #op_name); \
    }

template<typename T> T add_op(T a, T b) { return a + b; }
Tensor Tensor::add(const Tensor& other) const {
    DISPATCH_DTYPE(dtype_, "add", add_op);
}

template<typename T> T sub_op(T a, T b) { return a - b; }
Tensor Tensor::sub(const Tensor& other) const {
    DISPATCH_DTYPE(dtype_, "sub", sub_op);
}

template<typename T> T mul_op(T a, T b) { return a * b; }
Tensor Tensor::mul(const Tensor& other) const {
    DISPATCH_DTYPE(dtype_, "mul", mul_op);
}

template<typename T> T div_op(T a, T b) { return a / b; }
Tensor Tensor::div(const Tensor& other) const {
    DISPATCH_DTYPE(dtype_, "div", div_op);
}

} // namespace axe