import numpy as np
import contextlib
from ._axe import Tensor, Device, DType, Variable
from ._axe import _value_and_grad_cpp_multi
from ._axe import set_grad_enabled, is_grad_enabled

# --- Gradient Tracking ---

@contextlib.contextmanager
def no_grad():
    """
    Context manager to disable gradient tracking.
    """
    prev = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(prev)

def value_and_grad(fn, argnums=0):
    """
    Returns a function that computes both the value and gradient of `fn`.
    """
    def vjp_fn(*args):
        # Wrap arguments in Variables if they are not already
        wrapped_args = []
        for i, arg in enumerate(args):
            if not isinstance(arg, (Variable, Tensor)):
                arg = array(arg)

            # Ensure the argument we want to differentiate is a Variable that requires grad
            is_diff_arg = (isinstance(argnums, int) and i == argnums) or \
                          (isinstance(argnums, (list, tuple)) and i in argnums)

            if is_diff_arg and not (isinstance(arg, Variable) and arg.requires_grad):
                 if isinstance(arg, Variable):
                     arg.requires_grad = True # Promote to grad-requiring
                 else:
                     arg = array(np.array(arg), requires_grad=True)

            wrapped_args.append(arg)

        value, all_grads = _value_and_grad_cpp_multi(fn, *wrapped_args)

        if isinstance(argnums, int):
            grads = all_grads[argnums]
        else:
            grads = tuple(all_grads[i] for i in argnums)

        return value, grads
    return vjp_fn

def grad(fn, argnums=0):
    """
    Returns a function that computes the gradient of `fn`.
    """
    vjp_fn = value_and_grad(fn, argnums=argnums)
    def grad_fn(*args):
        _, grads = vjp_fn(*args)
        return grads
    return grad_fn


# --- Tensor Creation ---

def array(obj, dtype=None, device='cpu', requires_grad=False):
    """
    Creates a tensor from a Python object.
    """
    if isinstance(obj, Variable):
        if requires_grad:
            obj.requires_grad = True
        return obj

    if isinstance(obj, np.ndarray) and dtype is None:
        dtype_map = {'float32': 'float32', 'float64': 'float64', 'int32': 'int32', 'int64': 'int64'}
        dtype = dtype_map.get(str(obj.dtype), 'float32')

    if dtype is None:
        dtype = 'float32'

    if requires_grad and 'float' not in dtype:
        raise ValueError("Only floating point tensors can require gradients")

    np_arr = np.array(obj, dtype=dtype)
    dev = Device.CPU if device == 'cpu' else Device.GPU

    try:
        dt = getattr(DType, dtype.capitalize())
    except AttributeError:
        raise ValueError(f"Unsupported dtype: {dtype}")

    shape = list(np_arr.shape)
    t = Tensor(shape, dt, dev)
    np.copyto(np.array(t, copy=False), np_arr)

    if requires_grad and is_grad_enabled():
        return Variable(t, requires_grad=True)
    else:
        return t

def zeros(shape, dtype='float32', device='cpu', requires_grad=False):
    t = Tensor.zeros(shape, getattr(DType, dtype.capitalize()), Device.CPU if device == 'cpu' else Device.GPU)
    if requires_grad:
        return Variable(t, requires_grad=True)
    return t

def ones(shape, dtype='float32', device='cpu', requires_grad=False):
    t = Tensor.ones(shape, getattr(DType, dtype.capitalize()), Device.CPU if device == 'cpu' else Device.GPU)
    if requires_grad:
        return Variable(t, requires_grad=True)
    return t

def arange(start, end, dtype='float32', device='cpu', requires_grad=False):
    t = Tensor.arange(start, end, getattr(DType, dtype.capitalize()), Device.CPU if device == 'cpu' else Device.GPU)
    if requires_grad:
        return Variable(t, requires_grad=True)
    return t

# --- Reduction Operations ---
def sum(x, axis=None, keepdims=False):
    if axis is not None or keepdims:
        raise NotImplementedError("sum with axis/keepdims is not yet supported for Variable objects.")
    if isinstance(x, Variable):
        return x.sum()
    return x.sum()

def mean(x, axis=None, keepdims=False):
    if axis is not None or keepdims:
        raise NotImplementedError("mean with axis/keepdims is not yet supported for Variable objects.")
    # This will be replaced with a C++ implementation
    if isinstance(x, Variable):
        return array(np.mean(x.data.numpy()), requires_grad=x.requires_grad)
    return x.mean()

def max(x, axis=None, keepdims=False):
    if axis is not None or keepdims:
        raise NotImplementedError("max with axis/keepdims is not yet supported for Variable objects.")
    if isinstance(x, Variable):
        # This will be replaced with a C++ implementation
        return array(np.max(x.data.numpy()), requires_grad=x.requires_grad)
    return x.max()