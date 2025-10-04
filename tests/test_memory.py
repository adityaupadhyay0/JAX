import pytest
import axe
import numpy as np

def setup_function():
    """Reset memory stats before each test."""
    # This is a debug-only function to ensure a clean slate for each test.
    axe.memory.debug_clear_everything()


def test_initial_memory_state():
    """Verify that memory usage starts at zero."""
    stats = axe.memory.get_stats()
    assert stats.allocated_bytes == 0
    assert stats.cached_bytes == 0

def test_tensor_allocation():
    """Test that creating a tensor increases allocated memory."""
    initial_allocated = axe.memory.allocated_bytes()

    # 100 floats, 4 bytes each = 400 bytes
    t = axe.zeros([100], dtype=axe.DType.Float32)

    current_allocated = axe.memory.allocated_bytes()
    assert current_allocated == initial_allocated + 400

    peak_bytes = axe.memory.peak_bytes()
    assert peak_bytes >= 400

def test_tensor_deallocation_and_caching():
    """Test that freeing a tensor moves its memory to the cache."""
    initial_stats = axe.memory.get_stats()

    # 200 floats, 4 bytes each = 800 bytes
    t = axe.zeros([200], dtype=axe.DType.Float32)

    stats_after_alloc = axe.memory.get_stats()
    assert stats_after_alloc.allocated_bytes == initial_stats.allocated_bytes + 800

    # Deleting the tensor should move memory to cache
    del t

    stats_after_del = axe.memory.get_stats()
    assert stats_after_del.allocated_bytes == initial_stats.allocated_bytes
    assert stats_after_del.cached_bytes >= 800

def test_cache_reuse():
    """Test that a new tensor reuses memory from the cache."""
    # Allocate and deallocate a tensor to populate the cache
    t1 = axe.zeros([256], dtype=axe.DType.Float32)
    size_bytes = t1.shape[0] * 4
    del t1

    cached_before = axe.memory.cached_bytes()
    assert cached_before >= size_bytes

    # Allocate a new tensor of the same size
    t2 = axe.zeros([256], dtype=axe.DType.Float32)

    # The new tensor should have reused the cached block
    cached_after = axe.memory.cached_bytes()
    assert cached_after == cached_before - size_bytes

def test_peak_memory_tracking():
    """Test that peak memory is tracked correctly across multiple allocations."""
    axe.memory.reset_peak_bytes()

    t1 = axe.zeros([10], dtype=axe.DType.Float32) # 40 bytes
    assert axe.memory.peak_bytes() == 40

    t2 = axe.zeros([20], dtype=axe.DType.Float32) # 80 bytes, total 120
    assert axe.memory.peak_bytes() == 120

    del t1 # free 40 bytes
    assert axe.memory.peak_bytes() == 120 # peak should not change

    t3 = axe.zeros([5], dtype=axe.DType.Float32) # 20 bytes, reuses t1's block
    assert axe.memory.peak_bytes() == 120

    del t2
    del t3

def test_reset_peak_bytes():
    """Test that the peak memory counter can be reset."""
    t = axe.zeros([50], dtype=axe.DType.Float32) # 200 bytes
    assert axe.memory.peak_bytes() == 200

    axe.memory.reset_peak_bytes()

    # After reset, peak should be the current allocated amount
    assert axe.memory.peak_bytes() == 200

    del t

    assert axe.memory.allocated_bytes() == 0
    axe.memory.reset_peak_bytes()
    assert axe.memory.peak_bytes() == 0

def test_memory_timeline():
    """Tests that the memory timeline logs events correctly."""
    axe.memory.clear_memory_timeline()

    # 1. Allocation
    t1 = axe.zeros([100], dtype=axe.DType.Float32) # 400 bytes

    timeline = axe.memory.get_memory_timeline()
    assert len(timeline) == 1
    event1 = timeline[0]
    assert event1.type == axe.memory.MemoryEventType.ALLOCATE
    assert event1.size_bytes == 400
    assert event1.allocated_bytes_after == 400

    # 2. Deallocation (moves to cache)
    del t1

    timeline = axe.memory.get_memory_timeline()
    assert len(timeline) == 2
    event2 = timeline[1]
    assert event2.type == axe.memory.MemoryEventType.DEALLOCATE
    assert event2.size_bytes == 400
    assert event2.allocated_bytes_after == 0
    assert event2.cached_bytes_after == 400

    # 3. Clear timeline
    axe.memory.clear_memory_timeline()
    assert len(axe.memory.get_memory_timeline()) == 0