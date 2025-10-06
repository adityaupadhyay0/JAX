import pytest
import numpy as np
import axe
from axe import nn
from axe import optim

def test_bmm():
    a = axe.randn(2, 3, 4)
    b = axe.randn(2, 4, 5)
    c = a.bmm(b)
    assert c.shape == [2, 3, 5]

def test_transpose():
    a = axe.randn(2, 3, 4)
    b = a.transpose(0, 1)
    assert b.shape == [3, 2, 4]

    c = b.transpose(0, 1)
    assert c.shape == [2, 3, 4]
    assert np.allclose(np.array(a.data), np.array(c.data))

def test_softmax():
    a = axe.randn(2, 5)
    b = axe.functional.softmax(a, axis=1)

    # The sum of each row should be 1
    sum_b = b.sum(axis=1)
    assert np.allclose(np.array(sum_b.data), 1.0)

def test_attention():
    N, L, E, S, V = 2, 3, 4, 5, 6
    query = axe.randn(N, L, E)
    key = axe.randn(N, S, E)
    value = axe.randn(N, S, V)

    attention = nn.Attention()
    output, attn_weights = attention(query, key, value)

    assert output.shape == [N, L, V]
    assert attn_weights.shape == [N, L, S]
    # The sum of the attention weights for each query should be 1
    sum_attn_weights = attn_weights.sum(axis=-1)
    assert np.allclose(np.array(sum_attn_weights.data), 1.0)