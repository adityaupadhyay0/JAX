import pytest
import numpy as np
from python.axe import array

def test_add():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    b_ax = array(b_np)
    c_ax = a_ax + b_ax
    c_np = a_np + b_np
    assert np.allclose(np.array(c_ax), c_np)

def test_sub():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    b_ax = array(b_np)
    c_ax = a_ax - b_ax
    c_np = a_np - b_np
    assert np.allclose(np.array(c_ax), c_np)

def test_mul():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    b_ax = array(b_np)
    c_ax = a_ax * b_ax
    c_np = a_np * b_np
    assert np.allclose(np.array(c_ax), c_np)

def test_div():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32) + 1 # Avoid division by zero
    a_ax = array(a_np)
    b_ax = array(b_np)
    c_ax = a_ax / b_ax
    c_np = a_np / b_np
    assert np.allclose(np.array(c_ax), c_np)

def test_matmul():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(3, 4).astype(np.float32)
    a_ax = array(a_np)
    b_ax = array(b_np)
    c_ax = a_ax @ b_ax
    c_np = a_np @ b_np
    assert c_ax.shape == [2, 4]
    assert np.allclose(c_ax.numpy(), c_np)

def test_sum():
    a_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    s_ax = a_ax.sum()
    s_np = a_np.sum()
    assert s_ax.shape == [1]
    assert np.allclose(s_ax.numpy(), s_np)

def test_mean():
    a_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    m_ax = a_ax.mean()
    m_np = a_np.mean()
    assert m_ax.shape == [1]
    assert np.allclose(m_ax.numpy(), m_np)

def test_broadcasting():
    # Test broadcasting with a scalar
    a_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    scalar_ax = array(2.0)

    c_ax = a_ax * scalar_ax
    c_np = a_np * 2.0
    assert c_ax.shape == [2, 3]
    assert np.allclose(c_ax.numpy(), c_np)

    # Test broadcasting a vector to a matrix
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(3).astype(np.float32)
    a_ax = array(a_np)
    b_ax = array(b_np)

    c_ax = a_ax + b_ax
    c_np = a_np + b_np
    assert c_ax.shape == [2, 3]
    assert np.allclose(c_ax.numpy(), c_np)

    # Test broadcasting with different dimensions
    a_np = np.random.randn(2, 1, 3).astype(np.float32)
    b_np = np.random.randn(1, 4, 3).astype(np.float32)
    a_ax = array(a_np)
    b_ax = array(b_np)

    c_ax = a_ax - b_ax
    c_np = a_np - b_np
    assert c_ax.shape == [2, 4, 3]
    assert np.allclose(c_ax.numpy(), c_np)

def test_max():
    a_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    m_ax = a_ax.max()
    m_np = a_np.max()
    assert m_ax.shape == [1]
    assert np.allclose(m_ax.numpy(), m_np)