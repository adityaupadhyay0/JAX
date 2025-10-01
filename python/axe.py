import numpy as np
import importlib.util
import os

# Load the C++ extension
import sys
sys.path.append(os.path.dirname(__file__))
from _axe import Tensor, Device, DType

def array(obj, dtype='float32', device='cpu'):
    np_arr = np.array(obj, dtype=dtype)
    dev = Device.CPU if device == 'cpu' else Device.GPU
    dt = getattr(DType, dtype.capitalize())
    shape = list(np_arr.shape)
    t = Tensor(shape, dt, dev)
    # Copy data from numpy to tensor
    arr_bytes = np_arr.tobytes()
    import ctypes
    ctypes.memmove(int(t.data), arr_bytes, len(arr_bytes))
    return t

def zeros(shape, dtype='float32', device='cpu'):
    dev = Device.CPU if device == 'cpu' else Device.GPU
    dt = getattr(DType, dtype.capitalize())
    return Tensor.zeros(shape, dt, dev)

def ones(shape, dtype='float32', device='cpu'):
    dev = Device.CPU if device == 'cpu' else Device.GPU
    dt = getattr(DType, dtype.capitalize())
    return Tensor.ones(shape, dt, dev)

def arange(start, end, dtype='float32', device='cpu'):
    dev = Device.CPU if device == 'cpu' else Device.GPU
    dt = getattr(DType, dtype.capitalize())
    return Tensor.arange(start, end, dt, dev)
