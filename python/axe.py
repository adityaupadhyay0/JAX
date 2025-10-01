import numpy as np
import importlib.util
import os

# Load the C++ extension
import sys
sys.path.append(os.path.dirname(__file__))
from _axe import Tensor, Device, DType

def array(obj, dtype=None, device='cpu'):
    # If obj is already a numpy array, use its dtype if not specified
    if isinstance(obj, np.ndarray) and dtype is None:
        dtype_map = {
            'float32': 'float32', 'float64': 'float64',
            'int32': 'int32', 'int64': 'int64'
        }
        dtype = dtype_map.get(str(obj.dtype), 'float32')

    # If dtype is still None, default to float32
    if dtype is None:
        dtype = 'float32'

    np_arr = np.array(obj, dtype=dtype)
    dev = Device.CPU if device == 'cpu' else Device.GPU

    # Map numpy dtype to our DType enum
    try:
        dt = getattr(DType, dtype.capitalize())
    except AttributeError:
        raise ValueError(f"Unsupported dtype: {dtype}")

    shape = list(np_arr.shape)

    # Create an empty tensor
    t = Tensor(shape, dt, dev)

    # Copy data from numpy array to our tensor using buffer protocol
    np.copyto(np.array(t, copy=False), np_arr)

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
