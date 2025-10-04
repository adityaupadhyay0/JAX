#include <gtest/gtest.h>
#include "allocator.h"
#include "tensor.h"

using namespace axe::memory;
using namespace axe;

class AllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure the allocator is clean before each test
        Allocator::get_instance().debug_clear_everything();
    }

    void TearDown() override {
        // Clean up after each test to be safe
        Allocator::get_instance().debug_clear_everything();
    }
};

TEST_F(AllocatorTest, Singleton) {
    Allocator& a1 = Allocator::get_instance();
    Allocator& a2 = Allocator::get_instance();
    ASSERT_EQ(&a1, &a2);
}

TEST_F(AllocatorTest, SimpleAllocation) {
    Allocator& allocator = Allocator::get_instance();

    size_t size = 128;
    void* ptr = allocator.allocate(size, Device::CPU);

    ASSERT_NE(ptr, nullptr);

    AllocatorStats stats = allocator.get_stats(Device::CPU);
    EXPECT_EQ(stats.allocated_bytes, size);
    EXPECT_EQ(stats.peak_bytes, size);
    EXPECT_EQ(stats.cached_bytes, 0);

    allocator.deallocate(ptr, size, Device::CPU);
}

TEST_F(AllocatorTest, DeallocationAndCaching) {
    Allocator& allocator = Allocator::get_instance();

    size_t size = 256;
    void* ptr = allocator.allocate(size, Device::CPU);
    ASSERT_NE(ptr, nullptr);

    allocator.deallocate(ptr, size, Device::CPU);

    AllocatorStats stats = allocator.get_stats(Device::CPU);
    EXPECT_EQ(stats.allocated_bytes, 0);
    EXPECT_EQ(stats.peak_bytes, size);
    EXPECT_EQ(stats.cached_bytes, size);
}

TEST_F(AllocatorTest, ReuseFromCache) {
    Allocator& allocator = Allocator::get_instance();

    size_t size = 512;
    void* ptr1 = allocator.allocate(size, Device::CPU);
    allocator.deallocate(ptr1, size, Device::CPU);

    AllocatorStats stats_after_dealloc = allocator.get_stats(Device::CPU);
    EXPECT_EQ(stats_after_dealloc.cached_bytes, size);

    void* ptr2 = allocator.allocate(size, Device::CPU);
    ASSERT_EQ(ptr1, ptr2); // Should reuse the exact same block

    AllocatorStats stats_after_realloc = allocator.get_stats(Device::CPU);
    EXPECT_EQ(stats_after_realloc.allocated_bytes, size);
    EXPECT_EQ(stats_after_realloc.cached_bytes, 0);
    EXPECT_EQ(stats_after_realloc.peak_bytes, size);

    allocator.deallocate(ptr2, size, Device::CPU);
}

TEST_F(AllocatorTest, PeakMemoryTracking) {
    Allocator& allocator = Allocator::get_instance();

    void* p1 = allocator.allocate(100, Device::CPU);
    void* p2 = allocator.allocate(200, Device::CPU);

    AllocatorStats stats1 = allocator.get_stats(Device::CPU);
    EXPECT_EQ(stats1.allocated_bytes, 300);
    EXPECT_EQ(stats1.peak_bytes, 300);

    allocator.deallocate(p1, 100, Device::CPU);

    AllocatorStats stats2 = allocator.get_stats(Device::CPU);
    EXPECT_EQ(stats2.allocated_bytes, 200);
    EXPECT_EQ(stats2.cached_bytes, 100);
    EXPECT_EQ(stats2.peak_bytes, 300); // Peak should not change

    void* p3 = allocator.allocate(50, Device::CPU);

    AllocatorStats stats3 = allocator.get_stats(Device::CPU);
    EXPECT_EQ(stats3.allocated_bytes, 250);
    EXPECT_EQ(stats3.peak_bytes, 300);

    allocator.deallocate(p2, 200, Device::CPU);
    allocator.deallocate(p3, 50, Device::CPU);
}

TEST_F(AllocatorTest, ResetPeak) {
    Allocator& allocator = Allocator::get_instance();

    void* p1 = allocator.allocate(100, Device::CPU);
    allocator.allocate(200, Device::CPU);

    EXPECT_EQ(allocator.get_stats(Device::CPU).peak_bytes, 300);

    allocator.deallocate(p1, 100, Device::CPU);

    allocator.reset_peak_bytes(Device::CPU);

    EXPECT_EQ(allocator.get_stats(Device::CPU).peak_bytes, 200);
}