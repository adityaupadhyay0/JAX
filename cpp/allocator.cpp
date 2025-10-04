#include "include/allocator.h"
#include "include/tensor.h"
#include "include/exception.h"
#include <cstdlib> // For malloc and free
#include <iostream>

namespace axe {
namespace memory {

// Singleton instance getter
Allocator& Allocator::get_instance() {
    static Allocator instance;
    return instance;
}

Allocator::~Allocator() {
    // Free all cached memory blocks on destruction
    for (auto const& [size, blocks] : cpu_memory_pool_) {
        for (void* block : blocks) {
            free(block);
        }
    }
    // TODO: Add GPU cleanup later
}

// Main allocation function, dispatches based on device
void* Allocator::allocate(size_t size, Device device) {
    switch (device) {
        case Device::CPU:
            return allocate_cpu(size);
        case Device::GPU:
            // GPU support to be added
            throw std::runtime_error("GPU allocator not yet implemented.");
        default:
            throw std::runtime_error("Unsupported device for allocation.");
    }
}

// Main deallocation function, dispatches based on device
void Allocator::deallocate(void* ptr, size_t size, Device device) {
    if (!ptr) return;
    switch (device) {
        case Device::CPU:
            deallocate_cpu(ptr, size);
            break;
        case Device::GPU:
            // GPU support to be added
            throw std::runtime_error("GPU deallocator not yet implemented.");
        default:
            throw std::runtime_error("Unsupported device for deallocation.");
    }
}

// Get memory statistics for a given device
AllocatorStats Allocator::get_stats(Device device) const {
    switch (device) {
        case Device::CPU: {
            std::lock_guard<std::mutex> lock(cpu_mutex_);
            return cpu_stats_;
        }
        case Device::GPU:
            throw std::runtime_error("GPU stats not yet implemented.");
        default:
            throw std::runtime_error("Unsupported device for stats.");
    }
}

// Reset peak memory usage for a given device
void Allocator::reset_peak_bytes(Device device) {
     switch (device) {
        case Device::CPU: {
            std::lock_guard<std::mutex> lock(cpu_mutex_);
            cpu_stats_.peak_bytes = cpu_stats_.allocated_bytes;
            break;
        }
        case Device::GPU:
            throw std::runtime_error("GPU reset_peak_bytes not yet implemented.");
        default:
            throw std::runtime_error("Unsupported device for reset_peak_bytes.");
    }
}


// --- CPU Specific Implementation ---

void Allocator::free_cache(Device device) {
    if (device == Device::CPU) {
        std::lock_guard<std::mutex> lock(cpu_mutex_);
        for (auto const& [size, blocks] : cpu_memory_pool_) {
            for (void* block : blocks) {
                free(block);
            }
        }
        cpu_memory_pool_.clear();
        cpu_stats_.cached_bytes = 0;
    }
    // TODO: Add GPU support later
}

void* Allocator::allocate_cpu(size_t size) {
    std::lock_guard<std::mutex> lock(cpu_mutex_);

    // Check if a suitable block is available in the pool
    auto it = cpu_memory_pool_.find(size);
    if (it != cpu_memory_pool_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        cpu_stats_.cached_bytes -= size;
        cpu_stats_.allocated_bytes += size;
        cpu_memory_timeline_.push_back({MemoryEventType::ALLOCATE, size, std::chrono::high_resolution_clock::now(), cpu_stats_.allocated_bytes, cpu_stats_.cached_bytes});
        return ptr;
    }

    // No suitable block found, try to allocate new memory
    void* new_ptr = malloc(size);

    if (!new_ptr) {
        // First attempt failed, try to free cache and retry
        free_cache(Device::CPU);
        new_ptr = malloc(size); // Retry allocation
        if (!new_ptr) {
            // Still failed, throw a more informative OOM error
            throw OOMError("Out of memory when allocating " + std::to_string(size) +
                           " bytes. No cached memory was available to free. "
                           "Try reducing tensor sizes or using `axe.checkpoint`.");
        }
    }

    // Update stats for the new allocation
    cpu_stats_.allocated_bytes += size;
    if (cpu_stats_.allocated_bytes > cpu_stats_.peak_bytes) {
        cpu_stats_.peak_bytes = cpu_stats_.allocated_bytes;
    }
    cpu_memory_timeline_.push_back({MemoryEventType::ALLOCATE, size, std::chrono::high_resolution_clock::now(), cpu_stats_.allocated_bytes, cpu_stats_.cached_bytes});

    return new_ptr;
}

void Allocator::deallocate_cpu(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(cpu_mutex_);

    // Add the block to the memory pool for caching
    cpu_memory_pool_[size].push_back(ptr);

    // Update stats
    cpu_stats_.allocated_bytes -= size;
    cpu_stats_.cached_bytes += size;
    cpu_memory_timeline_.push_back({MemoryEventType::DEALLOCATE, size, std::chrono::high_resolution_clock::now(), cpu_stats_.allocated_bytes, cpu_stats_.cached_bytes});
}

std::vector<MemoryEvent> Allocator::get_memory_timeline(Device device) const {
    if (device == Device::CPU) {
        std::lock_guard<std::mutex> lock(cpu_mutex_);
        return cpu_memory_timeline_;
    }
    throw std::runtime_error("GPU timeline not yet implemented.");
}

void Allocator::clear_memory_timeline(Device device) {
    if (device == Device::CPU) {
        std::lock_guard<std::mutex> lock(cpu_mutex_);
        cpu_memory_timeline_.clear();
    }
    // TODO: Add GPU support later
}

void Allocator::debug_clear_everything() {
    // This is a test-only method to reset the state of the singleton allocator.
    // It's not thread-safe and should not be used in production code.

    // Free all cached memory blocks
    for (auto const& [size, blocks] : cpu_memory_pool_) {
        for (void* block : blocks) {
            free(block);
        }
    }
    cpu_memory_pool_.clear();
    cpu_stats_ = {}; // Reset stats to zero

    // TODO: Add GPU cleanup later
}

} // namespace memory
} // namespace axe