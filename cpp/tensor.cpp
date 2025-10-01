size_t Tensor::nbytes() const {
    size_t n = 1;
    for (auto d : shape_) n *= d;
    switch (dtype_) {
        case DType::Float32: n *= 4; break;
        case DType::Float64: n *= 8; break;
        case DType::Int32:   n *= 4; break;
        case DType::Int64:   n *= 8; break;
        default: break;
    }
    return n;
}
#include "tensor.h"
#include <cstring>
#include <stdexcept>

namespace axe {

Tensor::Tensor(const std::vector<size_t>& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device), data_(nullptr), ref_count_(1) {
        : shape_(shape), dtype_(dtype), device_(device), data_(nullptr), ref_count_ptr_(new size_t(1)) {
    // Simple CPU allocator for now
    size_t nbytes = 1;
    for (auto d : shape) nbytes *= d;
    switch (dtype) {
        case DType::Float32: nbytes *= 4; break;
        case DType::Float64: nbytes *= 8; break;
        case DType::Int32:   nbytes *= 4; break;
        case DType::Int64:   nbytes *= 8; break;
        default: throw std::runtime_error("Unknown dtype");
    }
    if (device_ == Device::GPU) {
        // GPU allocation stub (to be implemented)
        throw std::runtime_error("GPU device support not yet implemented");
    }
    data_ = malloc(nbytes);
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
    if (data_) free(data_);
        if (--(*ref_count_ptr_) == 0) {
            if (data_) free(data_);
            delete ref_count_ptr_;
        }
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    std::memset(t.data(), 0, t.shape().size() * sizeof(float)); // crude
    return t;
}

Tensor Tensor::ones(const std::vector<size_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    size_t n = 1;
    for (auto d : shape) n *= d;
    if (dtype == DType::Float32) {
        float* ptr = static_cast<float*>(t.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = 1.0f;
    } else if (dtype == DType::Float64) {
        double* ptr = static_cast<double*>(t.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = 1.0;
    } else if (dtype == DType::Int32) {
        int32_t* ptr = static_cast<int32_t*>(t.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = 1;
    } else if (dtype == DType::Int64) {
        int64_t* ptr = static_cast<int64_t*>(t.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = 1;
    }
    return t;
}

Tensor Tensor::arange(size_t start, size_t end, DType dtype, Device device) {
    size_t n = end - start;
    Tensor t({n}, dtype, device);
    if (dtype == DType::Float32) {
        float* ptr = static_cast<float*>(t.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<float>(start + i);
    } else if (dtype == DType::Float64) {
        double* ptr = static_cast<double*>(t.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<double>(start + i);
    } else if (dtype == DType::Int32) {
        int32_t* ptr = static_cast<int32_t*>(t.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<int32_t>(start + i);
    } else if (dtype == DType::Int64) {
        int64_t* ptr = static_cast<int64_t*>(t.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<int64_t>(start + i);
    }
    return t;
}

} // namespace axe
