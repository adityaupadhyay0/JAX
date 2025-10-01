import pytest
import numpy as np
from python.axe import zeros, ones, arange, array

def test_device_enum():
    t = zeros([2, 2], dtype='float32', device='cpu')
    assert t.device.name == 'CPU'
    try:
        zeros([2, 2], dtype='float32', device='gpu')
    except RuntimeError as e:
        assert 'GPU device support not yet implemented' in str(e)

def test_zeros():
    t = zeros([2, 3], dtype='float32')
    assert t.shape == [2, 3]
    assert t.dtype.name == 'Float32'
    arr = np.array(t)
    assert arr.shape == (2, 3)
    assert np.allclose(arr, 0)

def test_ones():
    t = ones([4], dtype='int32')
    assert t.shape == [4]
    assert t.dtype.name == 'Int32'
    arr = np.array(t)
    assert arr.shape == (4,)
    assert np.all(arr == 1)

def test_arange():
    t = arange(0, 5, dtype='float64')
    assert t.shape == [5]
    assert t.dtype.name == 'Float64'
    arr = np.array(t)
    assert np.allclose(arr, np.arange(0, 5, dtype=np.float64))
def test_array_roundtrip():
    arr = np.random.randn(3, 2).astype(np.float32)
    t = array(arr)
    arr2 = np.array(t)
    assert np.allclose(arr, arr2)

def test_numpy_method():
    shape = (2, 3)
    np_arr = np.random.randn(*shape).astype(np.float32)
    t = array(np_arr)
    np_arr_back = t.numpy()
    assert isinstance(np_arr_back, np.ndarray)
    assert np_arr_back.shape == shape
    assert str(np_arr_back.dtype) == 'float32'
    assert np.allclose(np_arr, np_arr_back)
