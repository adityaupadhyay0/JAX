import pytest
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import python.axe as axe

def test_vmap_basic():
    def square(x):
        return x * x

    vmap_square = axe.vmap(square)

    vec = axe.array([1., 2., 3.])
    result = vmap_square(vec)

    expected = np.array([1., 4., 9.])
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

def test_vmap_multiple_args():
    def add_mul(x, y, z):
        return (x + y) * z

    vmap_add_mul = axe.vmap(add_mul, in_axes=(0, 0, None))

    xs = axe.array([[1, 2], [3, 4]])
    ys = axe.array([[5, 6], [7, 8]])
    z = axe.array(3)

    result = vmap_add_mul(xs, ys, z)

    expected_0 = (np.array([1, 2]) + np.array([5, 6])) * 3
    expected_1 = (np.array([3, 4]) + np.array([7, 8])) * 3
    expected = np.stack([expected_0, expected_1])

    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

def test_vmap_out_axes():
    # Test moving the batch axis to a different output dimension
    def func(x):
        return x.sum()

    # Input has shape (5, 3), batch axis is 0
    # Output of func is scalar
    # vmap output should have shape (5,)
    # with out_axes=0, this is the default.

    # with out_axes=0, we expect shape [5]
    vmap_func_0 = axe.vmap(func, in_axes=0, out_axes=0)
    x = axe.ones((5, 3))
    result_0 = vmap_func_0(x)
    assert result_0.shape == [5]
    np.testing.assert_allclose(result_0.numpy(), np.full(5, 3.0))

def test_nested_vmap():
    def matrix_vector_product(matrix, vector):
        return axe.matmul(matrix, vector)

    # vmap over the vector argument
    vmap_mv = axe.vmap(matrix_vector_product, in_axes=(None, 0))

    # vmap over the matrix argument of the already vmapped function
    vmap_mm = axe.vmap(vmap_mv, in_axes=(0, None))

    matrices = axe.ones((5, 3, 2)) # 5 matrices of shape (3, 2)
    vectors = axe.ones((10, 2))    # 10 vectors of shape (2,)

    result = vmap_mm(matrices, vectors)

    # Expected shape: [5, 10, 3]
    # 5 from the outer vmap (matrices)
    # 10 from the inner vmap (vectors)
    # 3 from the output of matmul
    assert result.shape == [5, 10, 3]

def test_vmap_grad():
    def f(x, y):
        return (x * y).sum()

    grad_f = axe.grad(f, argnums=(0,1))

    # vmap the gradient function
    vmap_grad_f = axe.vmap(grad_f, in_axes=(0, 0))

    xs = axe.arange(0, 10).reshape([5, 2])
    ys = axe.ones((5, 2))

    dxs, dys = vmap_grad_f(xs, ys)

    assert dxs.shape == [5, 2]
    assert dys.shape == [5, 2]

    # For f(x,y) = sum(x*y), d/dx is y, d/dy is x
    np.testing.assert_allclose(dxs.numpy(), ys.numpy())
    np.testing.assert_allclose(dys.numpy(), xs.numpy())

def test_vmap_jit_composition():
    @axe.jit
    def square(x):
        return x * x

    vmap_square = axe.vmap(square)

    vec = axe.array([1., 2., 3.])
    result = vmap_square(vec)

    expected = np.array([1., 4., 9.])
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

def test_jit_vmap_raises_error():
    def square(x):
        return x * x

    vmap_square = axe.vmap(square)
    jit_vmap_square = axe.jit(vmap_square)

    vec = axe.array([1., 2., 3.])
    with pytest.raises(NotImplementedError, match="jit\(vmap\(f\)\) is not supported"):
        jit_vmap_square(vec)

def test_vmap_performance():
    def f(x):
        return axe.matmul(x, x.transpose())

    vmap_f = axe.vmap(f)

    batch_size = 100
    data = axe.ones((batch_size, 16, 16))

    # Vmap execution
    start_vmap = time.time()
    vmap_f(data)
    end_vmap = time.time()
    vmap_time = end_vmap - start_vmap

    # Python loop execution
    start_loop = time.time()
    results = []
    for i in range(batch_size):
        results.append(f(data.slice(0, i)))
    axe.stack(results, 0)
    end_loop = time.time()
    loop_time = end_loop - start_loop

    print(f"Vmap time: {vmap_time:.6f}s, Loop time: {loop_time:.6f}s")
    if vmap_time >= loop_time:
        pytest.warns(UserWarning, match="vmap is not faster than a python loop yet")