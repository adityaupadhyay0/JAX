#include "tensor.h"
#include "include/exception.h"
#include "include/allocator.h"
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <sstream>
#include <functional>
#include <Eigen/Dense>

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
    size_t size_in_bytes = nbytes();
    if (size_in_bytes > 0) {
        data_ = memory::Allocator::get_instance().allocate(size_in_bytes, device_);
        if (!data_) throw std::bad_alloc();
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_), data_(other.data_), ref_count_ptr_(other.ref_count_ptr_) {
    ++(*ref_count_ptr_);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (--(*ref_count_ptr_) == 0) {
            if (data_) {
                memory::Allocator::get_instance().deallocate(data_, nbytes(), device_);
            }
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
        if (data_) {
            memory::Allocator::get_instance().deallocate(data_, nbytes(), device_);
        }
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

// Broadcasting and element-wise operations

// Calculate strides from shape
std::vector<size_t> calculate_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    if (!shape.empty()) {
        strides.back() = 1;
        for (int i = shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return strides;
}

// Calculate the output shape for broadcasting
std::vector<size_t> calculate_broadcast_shape(const std::vector<size_t>& a_shape, const std::vector<size_t>& b_shape) {
    size_t a_dims = a_shape.size();
    size_t b_dims = b_shape.size();
    size_t max_dims = std::max(a_dims, b_dims);
    std::vector<size_t> result_shape(max_dims);

    for (size_t i = 0; i < max_dims; ++i) {
        size_t a_dim = (i < a_dims) ? a_shape[a_dims - 1 - i] : 1;
        size_t b_dim = (i < b_dims) ? b_shape[b_dims - 1 - i] : 1;

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::runtime_error("Operands could not be broadcast together");
        }
        result_shape[max_dims - 1 - i] = std::max(a_dim, b_dim);
    }
    return result_shape;
}


template<typename T>
Tensor broadcast_op(const Tensor& a, const Tensor& b, std::function<T(T, T)> op) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Mismatched dtypes for element-wise op");
    }

    auto result_shape = calculate_broadcast_shape(a.shape(), b.shape());
    Tensor result(result_shape, a.dtype(), a.device());
    T* res_ptr = static_cast<T*>(result.data());
    const T* a_ptr = static_cast<const T*>(a.data());
    const T* b_ptr = static_cast<const T*>(b.data());

    auto a_strides = calculate_strides(a.shape());
    auto b_strides = calculate_strides(b.shape());

    std::vector<size_t> a_bcast_strides(result_shape.size(), 0);
    int a_dim_diff = result_shape.size() - a.shape().size();
    for(size_t i = 0; i < a.shape().size(); i++) {
        if (a.shape()[i] != result_shape[a_dim_diff + i]) {
            a_bcast_strides[a_dim_diff + i] = 0;
        } else {
            a_bcast_strides[a_dim_diff + i] = a_strides[i];
        }
    }

    std::vector<size_t> b_bcast_strides(result_shape.size(), 0);
    int b_dim_diff = result_shape.size() - b.shape().size();
    for(size_t i = 0; i < b.shape().size(); i++) {
        if (b.shape()[i] != result_shape[b_dim_diff + i]) {
            b_bcast_strides[b_dim_diff + i] = 0;
        } else {
            b_bcast_strides[b_dim_diff + i] = b_strides[i];
        }
    }

    std::vector<size_t> index(result_shape.size(), 0);
    for (size_t i = 0; i < result.nelement(); ++i) {
        size_t a_offset = 0;
        size_t b_offset = 0;
        for (size_t j = 0; j < result_shape.size(); ++j) {
            a_offset += index[j] * a_bcast_strides[j];
            b_offset += index[j] * b_bcast_strides[j];
        }
        res_ptr[i] = op(a_ptr[a_offset], b_ptr[b_offset]);

        for (int j = result_shape.size() - 1; j >= 0; --j) {
            index[j]++;
            if (index[j] < result_shape[j]) {
                break;
            }
            index[j] = 0;
        }
    }

    return result;
}

template<typename T>
void transpose_impl(const Tensor& a, Tensor& result) {
    using ConstEigenMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using EigenMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

    ConstEigenMap eigen_a(static_cast<const T*>(a.data()), a.shape()[0], a.shape()[1]);
    EigenMap eigen_result(static_cast<T*>(result.data()), result.shape()[0], result.shape()[1]);

    eigen_result = eigen_a.transpose();
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("transpose expects a 2D tensor");
    }

    std::vector<size_t> result_shape = {shape_[1], shape_[0]};
    Tensor result(result_shape, dtype_, device_);

    switch (dtype_) {
        case DType::Float32:
            transpose_impl<float>(*this, result);
            break;
        case DType::Float64:
            transpose_impl<double>(*this, result);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for transpose. Only Float32 and Float64 are supported.");
    }

    return result;
}

#define DISPATCH_DTYPE_BCAST(op_name, op) \
    switch (this->dtype()) { \
        case DType::Float32: return broadcast_op<float>(*this, other, op<float>); \
        case DType::Float64: return broadcast_op<double>(*this, other, op<double>); \
        case DType::Int32: return broadcast_op<int32_t>(*this, other, op<int32_t>); \
        case DType::Int64: return broadcast_op<int64_t>(*this, other, op<int64_t>); \
        default: throw std::runtime_error("Unsupported dtype for " #op_name); \
    }

template<typename T> T add_op(T a, T b) { return a + b; }
Tensor Tensor::add(const Tensor& other) const {
    DISPATCH_DTYPE_BCAST("add", add_op);
}

template<typename T> T sub_op(T a, T b) { return a - b; }
Tensor Tensor::sub(const Tensor& other) const {
    DISPATCH_DTYPE_BCAST("sub", sub_op);
}

template<typename T> T mul_op(T a, T b) { return a * b; }
Tensor Tensor::mul(const Tensor& other) const {
    DISPATCH_DTYPE_BCAST("mul", mul_op);
}

template<typename T> T div_op(T a, T b) { return a / b; }
Tensor Tensor::div(const Tensor& other) const {
    DISPATCH_DTYPE_BCAST("div", div_op);
}

// Matrix-Matrix multiplication
template<typename T>
void matmul_mat_impl(const Tensor& a, const Tensor& b, Tensor& result) {
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstEigenMap = Eigen::Map<const EigenMatrix>;
    using EigenMap = Eigen::Map<EigenMatrix>;

    ConstEigenMap eigen_a(static_cast<const T*>(a.data()), a.shape()[0], a.shape()[1]);
    ConstEigenMap eigen_b(static_cast<const T*>(b.data()), b.shape()[0], b.shape()[1]);
    EigenMap eigen_result(static_cast<T*>(result.data()), result.shape()[0], result.shape()[1]);

    eigen_result = eigen_a * eigen_b;
}

// Matrix-Vector multiplication
template<typename T>
void matmul_vec_impl(const Tensor& a, const Tensor& b, Tensor& result) {
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using ConstEigenMatrixMap = Eigen::Map<const EigenMatrix>;
    using ConstEigenVectorMap = Eigen::Map<const EigenVector>;
    using EigenVectorMap = Eigen::Map<EigenVector>;

    ConstEigenMatrixMap eigen_a(static_cast<const T*>(a.data()), a.shape()[0], a.shape()[1]);
    ConstEigenVectorMap eigen_b(static_cast<const T*>(b.data()), b.shape()[0]);
    EigenVectorMap eigen_result(static_cast<T*>(result.data()), result.shape()[0]);

    eigen_result = eigen_a * eigen_b;
}

Tensor Tensor::matmul(const Tensor& other) const {
    auto format_shape = [](const std::vector<size_t>& shape) {
        std::stringstream ss;
        ss << "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        ss << ")";
        return ss.str();
    };

    // Matrix-Vector Product
    if (shape_.size() == 2 && other.shape().size() == 1) {
        if (shape_[1] != other.shape()[0]) {
            throw AxeException("Incompatible dimensions for matmul (matrix-vector product)");
        }
        std::vector<size_t> result_shape = {shape_[0]};
        Tensor result(result_shape, dtype_, device_);
        switch (dtype_) {
            case DType::Float32: matmul_vec_impl<float>(*this, other, result); break;
            case DType::Float64: matmul_vec_impl<double>(*this, other, result); break;
            default: throw AxeException("Unsupported dtype for matmul. Only Float32 and Float64 are supported.");
        }
        return result;
    }

    if (shape_.size() != 2 || other.shape().size() != 2) {
        std::stringstream ss;
        ss << "Unsupported matmul dimensions: " << format_shape(shape_) << " and " << format_shape(other.shape());
        throw AxeException(ss.str());
    }
    if (shape_[1] != other.shape()[0]) {
        std::stringstream ss;
        ss << "Incompatible dimensions for matmul: shapes "
           << format_shape(shape_) << " and " << format_shape(other.shape())
           << " cannot be multiplied.";
        throw AxeException(ss.str());
    }
    if (dtype_ != other.dtype()) {
        throw AxeException("Mismatched dtypes for matmul");
    }

    std::vector<size_t> result_shape = {shape_[0], other.shape()[1]};
    Tensor result(result_shape, dtype_, device_);

    switch (dtype_) {
        case DType::Float32:
            matmul_mat_impl<float>(*this, other, result);
            break;
        case DType::Float64:
            matmul_mat_impl<double>(*this, other, result);
            break;
        default:
            throw AxeException("Unsupported dtype for matmul. Only Float32 and Float64 are supported.");
    }

    return result;
}

void Tensor::add_from_slice(const Tensor& source, size_t dim, size_t index) {
    if (device_ != source.device() || dtype_ != source.dtype()) {
        throw AxeException("Mismatched device or dtype for add_from_slice");
    }

    // This is a simplified implementation assuming float32 for brevity.
    // A full implementation would handle different dtypes.
    if (dtype_ != DType::Float32) {
        throw AxeException("add_from_slice currently only supports Float32");
    }

    float* dest_ptr = static_cast<float*>(data_);
    const float* src_ptr = static_cast<const float*>(source.data());

    size_t inner_nelements = 1;
    for (size_t i = dim + 1; i < shape_.size(); ++i) {
        inner_nelements *= shape_[i];
    }
    const size_t copy_size = inner_nelements;

    size_t outer_nelements = 1;
    for (size_t i = 0; i < dim; ++i) {
        outer_nelements *= shape_[i];
    }
    const size_t dim_stride = inner_nelements * shape_[dim];

    for (size_t i = 0; i < outer_nelements; ++i) {
        float* dest_offset = dest_ptr + (i * dim_stride + index * inner_nelements);
        const float* src_offset = src_ptr + (i * copy_size);
        for(size_t j = 0; j < copy_size; ++j) {
            dest_offset[j] += src_offset[j];
        }
    }
}

} // namespace axe
template<typename T>
T reduce(const T* data, size_t n, T initial_value, std::function<T(T, T)> op) {
    T accum = initial_value;
    for (size_t i = 0; i < n; ++i) {
        accum = op(accum, data[i]);
    }
    return accum;
}

axe::Tensor axe::Tensor::sum() const {
    Tensor result({}, dtype_, device_);
    switch (dtype_) {
        case DType::Float32:
            *static_cast<float*>(result.data()) = reduce<float>(static_cast<const float*>(data()), nelement(), 0.0f, std::plus<float>());
            break;
        case DType::Float64:
            *static_cast<double*>(result.data()) = reduce<double>(static_cast<const double*>(data()), nelement(), 0.0, std::plus<double>());
            break;
        case DType::Int32:
            *static_cast<int32_t*>(result.data()) = reduce<int32_t>(static_cast<const int32_t*>(data()), nelement(), 0, std::plus<int32_t>());
            break;
        case DType::Int64:
            *static_cast<int64_t*>(result.data()) = reduce<int64_t>(static_cast<const int64_t*>(data()), nelement(), 0, std::plus<int64_t>());
            break;
        default:
            throw std::runtime_error("Unsupported dtype for sum");
    }
    return result;
}

axe::Tensor axe::Tensor::slice(size_t dim, size_t index) const {
    if (dim >= shape_.size()) {
        throw AxeException("Slicing dimension out of bounds");
    }
    if (index >= shape_[dim]) {
        throw AxeException("Slice index out of bounds for dimension");
    }

    std::vector<size_t> new_shape;
    new_shape.reserve(shape_.size() > 1 ? shape_.size() - 1 : 1);
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i != dim) {
            new_shape.push_back(shape_[i]);
        }
    }
    axe::Tensor result(new_shape, dtype_, device_);
    char* dest_ptr = static_cast<char*>(result.data());
    const char* src_ptr = static_cast<const char*>(data_);
    const size_t item_size = dtype_size(dtype_);

    size_t inner_nelements = 1;
    for (size_t i = dim + 1; i < shape_.size(); ++i) {
        inner_nelements *= shape_[i];
    }
    const size_t copy_size = inner_nelements * item_size;

    size_t outer_nelements = 1;
    for (size_t i = 0; i < dim; ++i) {
        outer_nelements *= shape_[i];
    }
    const size_t dim_stride = inner_nelements * shape_[dim];

    for (size_t i = 0; i < outer_nelements; ++i) {
        const char* src_offset = src_ptr + (i * dim_stride + index * inner_nelements) * item_size;
        memcpy(dest_ptr, src_offset, copy_size);
        dest_ptr += copy_size;
    }

    return result;
}

axe::Tensor axe::Tensor::stack(const std::vector<axe::Tensor>& tensors, size_t dim) {
    if (tensors.empty()) {
        throw AxeException("Cannot stack an empty list of tensors.");
    }

    const auto& first_tensor = tensors[0];
    const auto& first_shape = first_tensor.shape();
    const DType dtype = first_tensor.dtype();
    const Device device = first_tensor.device();

    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].shape() != first_shape) {
            throw AxeException("All tensors in stack must have the same shape.");
        }
        if (tensors[i].dtype() != dtype) {
            throw AxeException("All tensors in stack must have the same dtype.");
        }
        if (tensors[i].device() != device) {
            throw AxeException("All tensors in stack must have the same device.");
        }
    }

    if (dim > first_shape.size()) {
        throw AxeException("Stack dimension is out of bounds.");
    }

    std::vector<size_t> new_shape = first_shape;
    new_shape.insert(new_shape.begin() + dim, tensors.size());

    axe::Tensor result(new_shape, dtype, device);
    char* dest_ptr = static_cast<char*>(result.data());
    const size_t item_size = dtype_size(dtype);
    const size_t num_tensors = tensors.size();

    size_t inner_nelements = 1;
    for (size_t i = dim; i < first_shape.size(); ++i) {
        inner_nelements *= first_shape[i];
    }
    const size_t copy_size_per_tensor = inner_nelements * item_size;

    size_t outer_nelements = 1;
    for (size_t i = 0; i < dim; ++i) {
        outer_nelements *= first_shape[i];
    }

    const size_t copy_block_size = (first_tensor.nelement() / outer_nelements) * item_size;

    for (size_t i = 0; i < outer_nelements; ++i) {
        for (size_t j = 0; j < num_tensors; ++j) {
            const char* src_data = static_cast<const char*>(tensors[j].data());
            const char* src_offset = src_data + i * copy_block_size;
            char* dest_offset = dest_ptr + (i * num_tensors * copy_block_size) + (j * copy_block_size);
            memcpy(dest_offset, src_offset, copy_block_size);
        }
    }

    return result;
}

axe::Tensor axe::Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_nelement = 1;
    for (auto d : new_shape) new_nelement *= d;

    if (new_nelement != nelement()) {
        throw AxeException("Cannot reshape tensor: total number of elements must remain the same.");
    }

    // Create a new tensor that shares data and ref count with the original
    Tensor result = *this; // Copy constructor increments ref count
    result.shape_ = new_shape; // Just change the shape
    return result;
}

axe::Tensor axe::Tensor::mean() const {
    size_t n = nelement();
    if (n == 0) {
        throw std::runtime_error("Cannot compute mean of an empty tensor");
    }
    Tensor sum_tensor = sum();
    Tensor result({}, dtype_, device_);
    switch (dtype_) {
        case DType::Float32:
            *static_cast<float*>(result.data()) = *static_cast<float*>(sum_tensor.data()) / static_cast<float>(n);
            break;
        case DType::Float64:
            *static_cast<double*>(result.data()) = *static_cast<double*>(sum_tensor.data()) / static_cast<double>(n);
            break;
        case DType::Int32:
             *static_cast<int32_t*>(result.data()) = *static_cast<int32_t*>(sum_tensor.data()) / n;
             break;
        case DType::Int64:
             *static_cast<int64_t*>(result.data()) = *static_cast<int64_t*>(sum_tensor.data()) / n;
             break;
        default:
            throw std::runtime_error("Unsupported dtype for mean");
    }
    return result;
}

axe::Tensor axe::Tensor::max() const {
    if (nelement() == 0) {
        throw std::runtime_error("Cannot compute max of an empty tensor");
    }
    Tensor result({}, dtype_, device_);
    switch (dtype_) {
        case DType::Float32: {
            const float* data_ptr = static_cast<const float*>(data());
            *static_cast<float*>(result.data()) = reduce<float>(data_ptr + 1, nelement() - 1, data_ptr[0], [](float a, float b){ return std::max(a, b); });
            break;
        }
        case DType::Float64: {
            const double* data_ptr = static_cast<const double*>(data());
            *static_cast<double*>(result.data()) = reduce<double>(data_ptr + 1, nelement() - 1, data_ptr[0], [](double a, double b){ return std::max(a, b); });
            break;
        }
        case DType::Int32: {
            const int32_t* data_ptr = static_cast<const int32_t*>(data());
            *static_cast<int32_t*>(result.data()) = reduce<int32_t>(data_ptr + 1, nelement() - 1, data_ptr[0], [](int32_t a, int32_t b){ return std::max(a, b); });
            break;
        }
        case DType::Int64: {
            const int64_t* data_ptr = static_cast<const int64_t*>(data());
            *static_cast<int64_t*>(result.data()) = reduce<int64_t>(data_ptr + 1, nelement() - 1, data_ptr[0], [](int64_t a, int64_t b){ return std::max(a, b); });
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for max");
    }
    return result;
}