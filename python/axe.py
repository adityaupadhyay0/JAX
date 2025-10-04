import numpy as np
import contextlib
import functools
import inspect
from _axe import Tensor, Device, DType, Variable, AxeError, OOMError, memory
from _axe import _add, _sub, _mul, _matmul, _sum, _stack
from _axe import _value_and_grad_cpp_multi
from _axe import set_grad_enabled, is_grad_enabled
from _axe import jit as _jit
from _axe import _checkpoint
from _axe import _vmap_impl

def _to_dtype_enum(dtype):
    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, str):
        try:
            return getattr(DType, dtype.capitalize())
        except AttributeError:
            raise ValueError(f"Unsupported dtype string: {dtype}")
    raise TypeError(f"Unsupported dtype type: {type(dtype)}")

def _to_numpy_dtype_str(dtype):
    if isinstance(dtype, str):
        return dtype
    if isinstance(dtype, DType):
        return dtype.name.lower()
    raise TypeError(f"Unsupported dtype type for numpy conversion: {type(dtype)}")

# --- Save Original C++ Methods ---
_original_tensor_add = Tensor.__add__
_original_tensor_sub = Tensor.__sub__
_original_tensor_mul = Tensor.__mul__
_original_tensor_truediv = Tensor.__truediv__
_original_tensor_matmul = Tensor.__matmul__
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
        return Variable(obj)
    # Handle python scalars by creating a new Variable from scratch
    return array(obj)

def _should_build_graph(*args):
    """Determines if an operation should be tracked in the computation graph."""
    if _jit.is_tracing():
        return True
    if is_grad_enabled():
        return any(isinstance(arg, Variable) and arg.requires_grad for arg in args)
    return False

def _clear_grads(*args):
    """Clears gradients for a list of Variables."""
    for arg in args:
        if isinstance(arg, Variable):
            arg.grad = None

# --- Python Operator Functions ---

def add(a, b):
    va = _ensure_variable(a)
    vb = _ensure_variable(b)
    if not _should_build_graph(va, vb):
        return Variable(_original_tensor_add(va.data, vb.data))
    file, line = _get_caller_location()
    return _add(va, vb, file, line)

def sub(a, b):
    va = _ensure_variable(a)
    vb = _ensure_variable(b)
    if not _should_build_graph(va, vb):
        return Variable(_original_tensor_sub(va.data, vb.data))
    file, line = _get_caller_location()
    return _sub(va, vb, file, line)

def mul(a, b):
    va = _ensure_variable(a)
    vb = _ensure_variable(b)
    if not _should_build_graph(va, vb):
        return Variable(_original_tensor_mul(va.data, vb.data))
    file, line = _get_caller_location()
    return _mul(va, vb, file, line)

def matmul(a, b):
    va = _ensure_variable(a)
    vb = _ensure_variable(b)
    if not _should_build_graph(va, vb):
        return Variable(_original_tensor_matmul(va.data, vb.data))
    file, line = _get_caller_location()
    return _matmul(va, vb, file, line)

def truediv(a, b):
    # div does not support gradients, so we don't build a graph.
    va = _ensure_variable(a)
    vb = _ensure_variable(b)
    return Variable(_original_tensor_truediv(va.data, vb.data))

def sum_op(x, axis=None, keepdims=False):
    if axis is not None or keepdims:
        raise NotImplementedError("sum with axis/keepdims is not yet supported.")
    vx = _ensure_variable(x)
    if not _should_build_graph(vx):
        return Variable(_original_tensor_sum(vx.data))
    file, line = _get_caller_location()
    return _sum(vx, file, line)

# --- Monkey-patching operators ---
# We only patch Variable, since Tensor ops should not build graphs.
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
        return Variable(_jit.jit_execute_with_engine(graph, input_tensors))
    return jit_wrapper


# --- Vectorization ---

def vmap(fn=None, in_axes=0, out_axes=0):
    def decorator(fn_):
        @functools.wraps(fn_)
        def vmapped_fn(*args):
            if _jit.is_tracing():
                raise NotImplementedError("jit(vmap(f)) is not supported. Try vmap(jit(f)) instead.")

            processed_args = []
            for arg in args:
                processed_args.append(_ensure_variable(arg))

            return _vmap_impl(fn_, tuple(processed_args), in_axes, out_axes)
        return vmapped_fn

    if fn is None:
        return decorator
    else:
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

        value_variable = fn(*wrapped_args)

        value_variable.backward()

        all_grads = [arg.grad for arg in wrapped_args]

        if isinstance(argnums, int):
            grads = all_grads[argnums]
        else:
            grads = tuple(all_grads[i] for i in argnums if i < len(all_grads))

        # Wrap gradients in Variables to maintain API consistency
        if isinstance(grads, tuple):
            return value_variable.data, tuple(Variable(g) if g is not None else None for g in grads)
        else:
            return value_variable.data, Variable(grads) if grads is not None else None
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

    numpy_dtype_str = _to_numpy_dtype_str(dtype)
    np_arr = np_arr.astype(numpy_dtype_str)

    if requires_grad and 'float' not in numpy_dtype_str:
        raise ValueError("Only floating point tensors can require gradients")

    dev = Device.CPU if device == 'cpu' else Device.GPU
    dt_enum = _to_dtype_enum(dtype)

    shape = [] if is_scalar else list(np_arr.shape)
    t = Tensor(shape, dt_enum, dev)
    np.copyto(np.array(t, copy=False), np_arr.reshape(t.shape))

    if requires_grad and is_grad_enabled():
        return Variable(t, requires_grad=True)
    else:
        # Always return a Variable for API consistency
        return Variable(t, requires_grad=False)

def zeros(shape, dtype='float32', device='cpu', requires_grad=False):
    dt_enum = _to_dtype_enum(dtype)
    t = Tensor.zeros(shape, dt_enum, Device.CPU if device == 'cpu' else Device.GPU)
    return Variable(t, requires_grad=True) if requires_grad else Variable(t)

def ones(shape, dtype='float32', device='cpu', requires_grad=False):
    dt_enum = _to_dtype_enum(dtype)
    t = Tensor.ones(shape, dt_enum, Device.CPU if device == 'cpu' else Device.GPU)
    return Variable(t, requires_grad=True) if requires_grad else Variable(t)

def arange(start, end, dtype='float32', device='cpu', requires_grad=False):
    dt_enum = _to_dtype_enum(dtype)
    t = Tensor.arange(start, end, dt_enum, Device.CPU if device == 'cpu' else Device.GPU)
    return Variable(t, requires_grad=True) if requires_grad else Variable(t)

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
    variables = [_ensure_variable(t) for t in tensors]
    if not _should_build_graph(*variables):
        tensor_data = [v.data for v in variables]
        return Variable(Tensor.stack(tensor_data, dim))

    return _stack(variables, dim)

def mean(x):
    data = x.data if isinstance(x, Variable) else x
    return data.mean()

def max(x):
    data = x.data if isinstance(x, Variable) else x
    return data.max()

# Overwrite the global sum with our new implementation
sum = sum_op

# --- Gradient Checkpointing ---

def checkpoint(fn):
    @functools.wraps(fn)
    def checkpointed_fn(*args):
        return _checkpoint(fn, list(args))
    return checkpointed_fn