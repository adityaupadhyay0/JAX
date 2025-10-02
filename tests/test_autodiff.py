import pytest
import numpy as np
from python import axe

def test_grad_simple():
    def square(x):
        return x * x

    x = axe.array([3.0], requires_grad=True)
    grad_fn = axe.grad(square)

    dy_dx = grad_fn(x).numpy()

    assert np.allclose(dy_dx, [6.0])

def test_value_and_grad():
    def square(x):
        return x * x

    x = axe.array([3.0], requires_grad=True)
    val_grad_fn = axe.value_and_grad(square)

    val, dy_dx = val_grad_fn(x)
    val = val.numpy()
    dy_dx = dy_dx.numpy()

    assert np.allclose(val, [9.0])
    assert np.allclose(dy_dx, [6.0])

def test_grad_chain_rule():
    def f(x):
        return x * x

    def g(y):
        return y * axe.array([2.0], requires_grad=True)

    def h(x):
        return g(f(x))

    x = axe.array([3.0], requires_grad=True)
    grad_fn = axe.grad(h)
    dy_dx = grad_fn(x).numpy()

    # h(x) = 2 * x^2
    # h'(x) = 4 * x
    assert np.allclose(dy_dx, [12.0])

def test_no_grad():
    x = axe.array([3.0], requires_grad=True)
    with axe.no_grad():
        y = x * x
    assert not isinstance(y, axe.Variable)
    assert isinstance(y, axe.Tensor)
    assert not hasattr(y, 'grad')


def test_linear_regression():
    # Model: y = w * x + b
    # Loss: L = sum((y_pred - y_true)^2)

    # Data
    X_np = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    Y_np = np.array([[2.0], [4.0], [6.0], [8.0]], dtype=np.float32)

    X = axe.array(X_np) # No grad needed for input data
    Y = axe.array(Y_np) # No grad needed for true labels

    # Initialize weights
    w = axe.array([[0.5]], requires_grad=True)
    b = axe.array([[0.1]], requires_grad=True)

    def model(w, b):
        return X @ w + b

    def loss_fn(w, b):
        y_pred = model(w,b)
        # The subtraction and power operations are broadcasted.
        # The result is a Variable that needs to be summed up to a scalar.
        diff = y_pred - Y
        return axe.sum(diff * diff)

    # Get gradients
    val, (grad_w, grad_b) = axe.value_and_grad(loss_fn, argnums=(0,1))(w,b)

    # Analytical gradients
    # L = sum((Xw + b - Y)^2)
    # dL/dw = sum(2 * (Xw + b - Y) * X)
    # dL/db = sum(2 * (Xw + b - Y))
    y_pred_np = X_np @ w.data.numpy() + b.data.numpy()
    diff = y_pred_np - Y_np

    expected_grad_w = np.sum(2 * diff * X_np)
    expected_grad_b = np.sum(2 * diff)

    assert grad_w is not None
    assert grad_b is not None

    assert np.allclose(grad_w.numpy(), [[expected_grad_w]])
    assert np.allclose(grad_b.numpy(), [[expected_grad_b]])