import pytest
import numpy as np
from python import axe
from unittest.mock import patch

def test_jit_correctness():
    """Tests that a JIT-compiled function gives the same result as a normal one."""
    def func(x, y):
        return axe.sum(x @ y)

    jitted_func = axe.jit(func)

    a = axe.array(np.random.randn(2, 3), requires_grad=True)
    b = axe.array(np.random.randn(3, 2), requires_grad=True)

    # Run original function
    expected = func(a, b)

    # Run jitted function twice to test both tracing and execution paths
    result1 = jitted_func(a, b) # First call now returns a Tensor
    result2 = jitted_func(a, b) # Second call executes from cache

    assert np.allclose(result1.numpy(), expected.data.numpy())
    assert np.allclose(result2.numpy(), expected.data.numpy())


def test_jit_caching():
    """Tests that the JIT function is only traced once for the same input shapes."""

    @axe.jit
    def func(x, y):
        return x + y

    with patch('python.axe._jit.start_tracing', wraps=axe._jit.start_tracing) as mock_start_tracing:
        a1 = axe.array([[1, 2], [3, 4]], requires_grad=True)
        b1 = axe.array([[5, 6], [7, 8]], requires_grad=True)

        # First call - should trace
        func(a1, b1)
        mock_start_tracing.assert_called_once()

        # Second call with same shapes - should use cache
        func(a1, b1)
        mock_start_tracing.assert_called_once()  # Should not be called again

        # Third call with different shapes - should trace again
        a2 = axe.array([1, 2, 3], requires_grad=True)
        b2 = axe.array([4, 5, 6], requires_grad=True)
        func(a2, b2)
        assert mock_start_tracing.call_count == 2


def test_jit_composition():
    """Tests a more complex function with multiple operations."""

    @axe.jit
    def complex_func(a, b, c):
        d = a @ b
        e = d + c
        f = axe.sum(e)
        return f

    a = axe.array(np.random.randn(4, 5), requires_grad=True)
    b = axe.array(np.random.randn(5, 3), requires_grad=True)
    c = axe.array(np.random.randn(4, 3), requires_grad=True)

    # Run jitted function twice to test caching with C++ execution
    result1 = complex_func(a, b, c)
    result2 = complex_func(a, b, c)

    # Manually compute expected result
    expected_d = a.data.numpy() @ b.data.numpy()
    expected_e = expected_d + c.data.numpy()
    expected_f = np.sum(expected_e)

    assert np.allclose(result1.numpy(), expected_f)
    assert np.allclose(result2.numpy(), expected_f)

def test_jit_with_constants():
    """Tests that JIT handles constants created inside the function."""
    @axe.jit
    def func_with_constant(x):
        # b is a constant that will be captured by the trace
        b = axe.array([1.0, 2.0, 3.0], requires_grad=False)
        return x + b

    a = axe.array([10.0, 20.0, 30.0], requires_grad=False)

    expected = a.numpy() + np.array([1.0, 2.0, 3.0])

    # Run once to trace and compile
    result1 = func_with_constant(a)
    # Run again to execute from cache
    result2 = func_with_constant(a)

    assert np.allclose(result1.numpy(), expected)
    assert np.allclose(result2.numpy(), expected)

def test_jit_dynamic_compilation_correctness():
    """
    Tests that the full dynamic compilation pipeline produces correct results.
    """
    def f_python(x, y, z):
        return (x @ y) * z + axe.array(2.0)

    @axe.jit
    def f_jit(x, y, z):
        return (x @ y) * z + axe.array(2.0)

    a = axe.array([[1., 2.], [3., 4.]])
    b = axe.array([[5., 6.], [7., 8.]])
    c = axe.array([[0.1, 0.2], [0.3, 0.4]])

    # Execute both the original Python version and the JIT version
    python_result = f_python(a, b, c)
    jit_result = f_jit(a, b, c)

    # The results should be numerically close
    assert np.allclose(python_result.numpy(), jit_result.numpy())

    # Run again to ensure cached version is also correct
    jit_result_cached = f_jit(a, b, c)
    assert np.allclose(python_result.numpy(), jit_result_cached.numpy())