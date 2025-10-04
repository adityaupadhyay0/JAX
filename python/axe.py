import numpy as np
import contextlib
import functools
import inspect
from ._axe import Tensor, Device, DType, Variable, AxeError
from ._axe import _add, _sub, _mul, _matmul, _sum
from ._axe import _value_and_grad_cpp_multi
from ._axe import set_grad_enabled, is_grad_enabled
from ._axe import jit as _jit

# --- Save Original C++ Methods ---
# This is crucial to prevent recursion in our Python operator overrides
_original_tensor_add = Tensor.__add__
_original_tensor_sub = Tensor.__sub__
_original_tensor_mul = Tensor.__mul__
_original_tensor_matmul = Tensor.__matmul__
_original_tensor_truediv = Tensor.__truediv__
_original_tensor_sum = Tensor.sum

def _get_caller_location(depth=2):
    """Helper to get the filename and line number of the caller."""
    try:
        frame = inspect.stack()[depth]
        return frame.filename, frame.lineno
    except IndexError:
        return "unknown", 0

def _ensure_variable(obj):
    """Ensures an object is a Variable for an operation."""
    if isinstance(obj, Variable):
        return obj
    if isinstance(obj, Tensor):
        return Variable(obj, requires_grad=False)
    return Variable(array(obj), requires_grad=False)

def _should_build_graph(a, b):
    """Determines if an operation should be tracked in the computation graph."""
    if _jit.is_tracing():
        return True
    if is_grad_enabled():
        is_a_grad_var = isinstance(a, Variable) and a.requires_grad
        is_b_grad_var = isinstance(b, Variable) and b.requires_grad
        return is_a_grad_var or is_b_grad_var
    return False

def _clear_grads(*args):
    """Clears gradients for a list of Variables."""
    for arg in args:
        if isinstance(arg, Variable):
            arg.grad = None

# --- Python Operator Functions ---

def add(a, b):
    if not _should_build_graph(a, b):
        a_data = a.data if isinstance(a, Variable) else a
        b_data = b.data if isinstance(b, Variable) else b
        return _original_tensor_add(a_data, b_data)
    file, line = _get_caller_location()
    return _add(_ensure_variable(a), _ensure_variable(b), file, line)

def sub(a, b):
    if not _should_build_graph(a, b):
        a_data = a.data if isinstance(a, Variable) else a
        b_data = b.data if isinstance(b, Variable) else b
        return _original_tensor_sub(a_data, b_data)
    file, line = _get_caller_location()
    return _sub(_ensure_variable(a), _ensure_variable(b), file, line)

def mul(a, b):
    if not _should_build_graph(a, b):
        a_data = a.data if isinstance(a, Variable) else a
        b_data = b.data if isinstance(b, Variable) else b
        return _original_tensor_mul(a_data, b_data)
    file, line = _get_caller_location()
    return _mul(_ensure_variable(a), _ensure_variable(b), file, line)

def matmul(a, b):
    if not _should_build_graph(a, b):
        a_data = a.data if isinstance(a, Variable) else a
        b_data = b.data if isinstance(b, Variable) else b
        return _original_tensor_matmul(a_data, b_data)
    file, line = _get_caller_location()
    return _matmul(_ensure_variable(a), _ensure_variable(b), file, line)

def truediv(a, b):
    a_data = a.data if isinstance(a, Variable) else a
    b_data = b.data if isinstance(b, Variable) else b
    return _original_tensor_truediv(a_data, b_data)

def sum_op(x, axis=None, keepdims=False):
    if axis is not None or keepdims:
        raise NotImplementedError("sum with axis/keepdims is not yet supported.")
    if not isinstance(x, (Tensor, Variable)):
        raise TypeError(f"sum() expected a Tensor or Variable, but got {type(x)}")
    if _should_build_graph(x, x):
        file, line = _get_caller_location()
        return _sum(_ensure_variable(x), file, line)
    else:
        data = x.data if isinstance(x, Variable) else x
        return _original_tensor_sum(data)

# --- Monkey-patching operators ---
Variable.__add__ = add
Variable.__sub__ = sub
Variable.__mul__ = mul
Variable.__matmul__ = matmul
Variable.__truediv__ = truediv
Variable.__radd__ = lambda b, a: add(a, b)
Variable.__rsub__ = lambda b, a: sub(a, b)
Variable.__rmul__ = lambda b, a: mul(a, b)
Variable.__rmatmul__ = lambda b, a: matmul(a, b)
Variable.sum = sum_op

Tensor.__add__ = add
Tensor.__sub__ = sub
Tensor.__mul__ = mul
Tensor.__matmul__ = matmul
Tensor.__truediv__ = truediv
Tensor.__radd__ = lambda b, a: add(a, b)
Tensor.__rsub__ = lambda b, a: sub(a, b)
Tensor.__rmul__ = lambda b, a: mul(a, b)
Tensor.__rmatmul__ = lambda b, a: matmul(a, b)
Tensor.sum = sum_op

# --- JIT Compilation ---

def _get_arg_signature(arg):
    if isinstance(arg, Variable):
        return (tuple(arg.data.shape), str(arg.data.dtype))
    if isinstance(arg, Tensor):
        return (tuple(arg.shape), str(arg.dtype))
    return type(arg)

def jit(fn):
    cache = {}
    @functools.wraps(fn)
    def jit_wrapper(*args):
        signature = tuple(_get_arg_signature(arg) for arg in args)
        if signature not in cache:
            _jit.start_tracing()
            try:
                traced_args = [_ensure_variable(arg) for arg in args]
                for var in traced_args:
                    _jit.register_input(var)
                fn(*traced_args)
                graph = _jit.stop_tracing()
            finally:
                if _jit.is_tracing():
                    _jit.stop_tracing()
            if not graph:
                return fn(*args)
            cache[signature] = graph
        graph = cache[signature]
        input_tensors = [arg.data if isinstance(arg, Variable) else arg for arg in args if isinstance(arg, (Tensor, Variable))]
        return _jit.jit_execute_with_engine(graph, input_tensors)
    return jit_wrapper


try:
    from ._axe import _vmap_impl
except ImportError:
    _vmap_impl = None

# --- Vectorization ---

def vmap(fn=None, in_axes=0, out_axes=0):
    """
    Creates a function that maps `fn` over designated axes of the inputs.
    This is a high-performance alternative to a Python for-loop.

    Can be used as a decorator, e.g. `@vmap` or `@vmap(in_axes=...)`.

    Args:
        fn (Callable, optional): The function to be mapped. Defaults to None.
        in_axes (int or tuple): Specifies the axis of the inputs to be mapped over.
        out_axes (int): Specifies the axis of the outputs to be mapped over.

    Returns:
        Callable: A new function that maps `fn` over the specified axes.
    """
    def decorator(fn_):
        @functools.wraps(fn_)
        def vmapped_fn(*args):
            if _jit.is_tracing():
                raise NotImplementedError("jit(vmap(f)) is not supported. Try vmap(jit(f)) instead.")
            if _vmap_impl is None:
                raise NotImplementedError("vmap is not implemented in the C++ backend yet.")

            # The actual logic is in the C++ backend for performance.
            processed_args = []
            for arg in args:
                # Ensure that all inputs are Tensors or Variables for the C++ backend
                if not isinstance(arg, (Tensor, Variable)):
                    processed_args.append(array(arg))
                else:
                    processed_args.append(arg)

            return _vmap_impl(fn_, tuple(processed_args), in_axes, out_axes)
        return vmapped_fn

    if fn is None:
        # This is the case where it's called with arguments, like @vmap(in_axes=0)
        return decorator
    else:
        # This is the case where it's called without arguments, like @vmap
        return decorator(fn)


# --- Gradient Tracking ---

@contextlib.contextmanager
def no_grad():
    prev = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(prev)

def value_and_grad(fn, argnums=0):
    def vjp_fn(*args):
        _clear_grads(*args)
        wrapped_args = []
        for i, arg in enumerate(args):
            arg = _ensure_variable(arg)
            is_diff_arg = (isinstance(argnums, int) and i == argnums) or \
                          (isinstance(argnums, (list, tuple)) and i in argnums)
            if is_diff_arg:
                 arg.requires_grad = True
            wrapped_args.append(arg)

        value, all_grads = _value_and_grad_cpp_multi(fn, *wrapped_args)

        if isinstance(argnums, int):
            grads = all_grads[argnums]
        else:
            grads = tuple(all_grads[i] for i in argnums if i < len(all_grads))
        return value, grads
    return vjp_fn

def grad(fn, argnums=0):
    vjp_fn = value_and_grad(fn, argnums=argnums)
    def grad_fn(*args):
        _, grads = vjp_fn(*args)
        return grads
    return grad_fn

# --- Tensor Creation ---

def array(obj, dtype=None, device='cpu', requires_grad=False):
    if isinstance(obj, Variable):
        if requires_grad:
            obj.requires_grad = True
        return obj
    if isinstance(obj, Tensor):
        if requires_grad:
            return Variable(obj, requires_grad=True)
        return obj

    is_scalar = not isinstance(obj, (list, tuple, np.ndarray))
    if is_scalar:
        np_arr = np.array([obj])
    else:
        np_arr = np.array(obj)

    if dtype is None:
        dtype = 'float32'

    np_arr = np_arr.astype(dtype)

    if requires_grad and 'float' not in dtype:
        raise ValueError("Only floating point tensors can require gradients")

    dev = Device.CPU if device == 'cpu' else Device.GPU
    try:
        dt_str = dtype.capitalize() if 'int' not in dtype else dtype
        dt = getattr(DType, dt_str)
    except AttributeError:
        raise ValueError(f"Unsupported dtype: {dtype}")

    shape = [] if is_scalar else list(np_arr.shape)
    t = Tensor(shape, dt, dev)
    np.copyto(np.array(t, copy=False), np_arr.reshape(t.shape))

    if requires_grad and is_grad_enabled():
        return Variable(t, requires_grad=True)
    else:
        return t

def zeros(shape, dtype='float32', device='cpu', requires_grad=False):
    t = Tensor.zeros(shape, getattr(DType, dtype.capitalize()), Device.CPU if device == 'cpu' else Device.GPU)
    return Variable(t, requires_grad=True) if requires_grad else t

def ones(shape, dtype='float32', device='cpu', requires_grad=False):
    t = Tensor.ones(shape, getattr(DType, dtype.capitalize()), Device.CPU if device == 'cpu' else Device.GPU)
    return Variable(t, requires_grad=True) if requires_grad else t

def arange(start, end, dtype='float32', device='cpu', requires_grad=False):
    t = Tensor.arange(start, end, getattr(DType, dtype.capitalize()), Device.CPU if device == 'cpu' else Device.GPU)
    return Variable(t, requires_grad=True) if requires_grad else t

def stack(tensors, dim=0):
    """
    Concatenates a sequence of tensors along a new axis.
    All tensors need to be of the same size.

    Args:
        tensors (sequence of Tensors): Tensors to stack.
        dim (int): The axis in the result tensor along which the input
                   tensors are stacked. Defaults to 0.

    Returns:
        Tensor: The stacked tensor.
    """
    return Tensor.stack(tensors, dim)

def mean(x):
    data = x.data if isinstance(x, Variable) else x
    return data.mean()

def max(x):
    data = x.data if isinstance(x, Variable) else x
    return data.max()

# Overwrite the global sum with our new implementation
sum = sum_op