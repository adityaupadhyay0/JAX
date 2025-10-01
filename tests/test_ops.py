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
    assert np.allclose(c_ax.numpy(), c_np)

def test_sub():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    b_ax = array(b_np)
    c_ax = a_ax - b_ax
    c_np = a_np - b_np
    assert np.allclose(c_ax.numpy(), c_np)

def test_mul():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32)
    a_ax = array(a_np)
    b_ax = array(b_np)
    c_ax = a_ax * b_ax
    c_np = a_np * b_np
    assert np.allclose(c_ax.numpy(), c_np)

def test_div():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(2, 3).astype(np.float32) + 1 # Avoid division by zero
    a_ax = array(a_np)
    b_ax = array(b_np)
    c_ax = a_ax / b_ax
    c_np = a_np / b_np
    assert np.allclose(c_ax.numpy(), c_np)