#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>

// Forward-declare Device to break circular dependency with tensor.h
namespace axe {
enum class Device;
}

namespace axe {
namespace memory {

struct AllocatorStats {
    size_t allocated_bytes = 0;
    size_t peak_bytes = 0;
    size_t cached_bytes = 0;
};

enum class MemoryEventType { ALLOCATE, DEALLOCATE, FREE_CACHE };

struct MemoryEvent {
    MemoryEventType type;
    size_t size_bytes;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    size_t allocated_bytes_after;
    size_t cached_bytes_after;
};

class Allocator {
public:
    static Allocator& get_instance();

    void* allocate(size_t size, axe::Device device);
    void deallocate(void* ptr, size_t size, axe::Device device);

    AllocatorStats get_stats(axe::Device device) const;
    void reset_peak_bytes(axe::Device device);

    // NOTE: For testing purposes only!
    void debug_clear_everything();

    std::vector<MemoryEvent> get_memory_timeline(Device device) const;
    void clear_memory_timeline(Device device);

private:
    Allocator() = default;
    ~Allocator();

    // Make Allocator non-copyable and non-movable
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&&) = delete;
    Allocator& operator=(Allocator&&) = delete;

    void* allocate_cpu(size_t size);
    void deallocate_cpu(void* ptr, size_t size);
    void free_cache(Device device);

    mutable std::mutex cpu_mutex_;
    AllocatorStats cpu_stats_;
    std::unordered_map<size_t, std::vector<void*>> cpu_memory_pool_;
    std::vector<MemoryEvent> cpu_memory_timeline_;
};

} // namespace memory
} // namespace axe