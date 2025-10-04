#pragma once
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>

namespace axe {

enum class Device { CPU, GPU };

enum class DType { Float32, Float64, Int32, Int64 };

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, DType dtype, Device device = Device::CPU);
    ~Tensor();
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        size_t ref_count() const { return *ref_count_ptr_; }

    const std::vector<size_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    void* data() { return data_; }
    const void* data() const { return data_; }

    size_t nbytes() const;
    size_t nelement() const;

    static Tensor zeros(const std::vector<size_t>& shape, DType dtype, Device device = Device::CPU);
    static Tensor ones(const std::vector<size_t>& shape, DType dtype, Device device = Device::CPU);
    static Tensor arange(size_t start, size_t end, DType dtype, Device device = Device::CPU);

    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    Tensor sum() const;
    Tensor mean() const;
    Tensor max() const;
    Tensor transpose() const;
    Tensor reshape(const std::vector<size_t>& new_shape) const;

    // Batching utilities for vmap
    Tensor slice(size_t dim, size_t index) const;
    static Tensor stack(const std::vector<Tensor>& tensors, size_t dim);

private:
    std::vector<size_t> shape_;
    DType dtype_;
    Device device_;
    void* data_;
        size_t* ref_count_ptr_;
};

} // namespace axe
