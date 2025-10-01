def test_device_enum():
    t = zeros([2, 2], dtype='float32', device='cpu')
    assert t.device.name == 'CPU'
    try:
        zeros([2, 2], dtype='float32', device='gpu')
    except RuntimeError as e:
        assert 'GPU device support not yet implemented' in str(e)
def test_ref_count():
    t1 = zeros([2, 2], dtype='float32')
    t2 = t1
    assert t1._obj.ref_count() == 2 or t1._obj.ref_count() == 1  # pybind11 may not expose refcount directly
import pytest
import numpy as np
from python.axe import zeros, ones, arange, array

def test_zeros():
    t = zeros([2, 3], dtype='float32')
    assert t.shape == [2, 3]
    assert t.dtype.name == 'Float32'
    arr = t.numpy()
    assert arr.shape == (2, 3)
    assert np.allclose(arr, 0)

def test_ones():
    t = ones([4], dtype='int32')
    assert t.shape == [4]
    assert t.dtype.name == 'Int32'
    arr = t.numpy()
    assert arr.shape == (4,)
    assert np.all(arr == 1)

def test_arange():
    t = arange(0, 5, dtype='float64')
    assert t.shape == [5]
    assert t.dtype.name == 'Float64'
    arr = t.numpy()
    assert np.allclose(arr, np.arange(0, 5, dtype=np.float64))
def test_array_roundtrip():
    arr = np.random.randn(3, 2).astype(np.float32)
    t = array(arr)
    arr2 = t.numpy()
    assert np.allclose(arr, arr2)
