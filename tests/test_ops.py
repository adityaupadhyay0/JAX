import pytest
import numpy as np
from python import axe

# Set a seed for reproducibility
np.random.seed(0)

def test_add():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = axe.array(a_np, requires_grad=True)
    b_ax = axe.array(b_np, requires_grad=True)
    c_ax = a_ax + b_ax
    c_np = a_np + b_np
    assert np.allclose(c_ax.data.numpy(), c_np)

def test_sub():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = axe.array(a_np, requires_grad=True)
    b_ax = axe.array(b_np, requires_grad=True)
    c_ax = a_ax - b_ax
    c_np = a_np - b_np
    assert np.allclose(c_ax.data.numpy(), c_np)

def test_mul():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = axe.array(a_np, requires_grad=True)
    b_ax = axe.array(b_np, requires_grad=True)
    c_ax = a_ax * b_ax
    c_np = a_np * b_np
    assert np.allclose(c_ax.data.numpy(), c_np)

def test_div():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32) + 1 # Avoid division by zero
    a_ax = axe.array(a_np)
    b_ax = axe.array(b_np)
    c_ax = a_ax / b_ax
    c_np = a_np / b_np
    assert np.allclose(c_ax.numpy(), c_np)


def test_matmul():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(3, 4).astype(np.float32)
    a_ax = axe.array(a_np, requires_grad=True)
    b_ax = axe.array(b_np, requires_grad=True)
    c_ax = a_ax @ b_ax
    c_np = a_np @ b_np
    assert c_ax.data.shape == [2, 4]
    assert np.allclose(c_ax.data.numpy(), c_np)

def test_sum():
    a_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = axe.array(a_np, requires_grad=True)
    s_ax = axe.sum(a_ax)
    s_np = np.sum(a_np)
    assert np.allclose(s_ax.data.numpy(), s_np)

def test_mean():
    a_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = axe.array(a_np)
    m_ax = axe.mean(a_ax)
    m_np = np.mean(a_np)
    assert np.allclose(m_ax.numpy(), m_np)


def test_broadcasting():
    # Test broadcasting with a scalar
    a_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = axe.array(a_np, requires_grad=True)
    scalar_ax = axe.array(2.0, requires_grad=True)

    c_ax = a_ax * scalar_ax
    c_np = a_np * 2.0
    assert c_ax.data.shape == [2, 3]
    assert np.allclose(c_ax.data.numpy(), c_np)

    # Test broadcasting with a vector
    d_np = np.random.randn(1, 3).astype(np.float32)
    d_ax = axe.array(d_np, requires_grad=True)
    e_ax = a_ax + d_ax
    e_np = a_np + d_np
    assert e_ax.data.shape == [2, 3]
    assert np.allclose(e_ax.data.numpy(), e_np)


def test_mixed_type_ops():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax_var = axe.array(a_np, requires_grad=True)
    b_ax_tensor = axe.array(b_np, requires_grad=False)

    # Variable + Tensor
    c_ax = a_ax_var + b_ax_tensor
    c_np = a_np + b_np
    assert np.allclose(c_ax.data.numpy(), c_np)

    # Tensor + Variable
    d_ax = b_ax_tensor + a_ax_var
    d_np = b_np + a_np
    assert np.allclose(d_ax.data.numpy(), d_np)

def test_max():
    a_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = axe.array(a_np)
    m_ax = axe.max(a_ax)
    m_np = np.max(a_np)
    assert np.allclose(m_ax.numpy(), m_np)