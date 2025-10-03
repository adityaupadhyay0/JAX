import pytest
import numpy as np
from python import axe

def test_grad_simple():
    def f(x):
        return x * x

    grad_f = axe.grad(f)
    x = axe.array([3.0], requires_grad=True)

    # Compute gradient
    grads = grad_f(x)

    # Check the gradient
    assert np.allclose(grads.numpy(), [6.0])

def test_grad_two_args():
    def f(x, y):
        return x * y + x

    grad_f_x = axe.grad(f, argnums=0)
    grad_f_y = axe.grad(f, argnums=1)

    x = axe.array([2.0], requires_grad=True)
    y = axe.array([3.0], requires_grad=True)

    # df/dx = y + 1 = 4
    dx = grad_f_x(x, y)
    assert np.allclose(dx.numpy(), [4.0])

    # df/dy = x = 2
    dy = grad_f_y(x, y)
    assert np.allclose(dy.numpy(), [2.0])

def test_value_and_grad():
    def f(x):
        return x * x * x

    vjp_f = axe.value_and_grad(f)
    x = axe.array([2.0], requires_grad=True)

    value, grads = vjp_f(x)

    # f(2) = 8
    assert np.allclose(value.numpy(), [8.0])
    # df/dx = 3 * x^2 = 12
    assert np.allclose(grads.numpy(), [12.0])

def test_no_grad():
    x = axe.array([3.0], requires_grad=True)
    with axe.no_grad():
        y = x * x
    assert not isinstance(y, axe.Variable)
    assert isinstance(y, axe.Tensor)

def test_chain_rule():
    def f(x):
        return x * x

    def g(y):
        return y + y

    x = axe.array([5.0], requires_grad=True)
    y = f(x)
    z = g(y)

    z.backward()

    # dz/dx = dz/dy * dy/dx = 2 * (2*x) = 4x = 20
    assert np.allclose(x.grad.numpy(), [20.0])