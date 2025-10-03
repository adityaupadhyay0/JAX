import pytest
from python import axe
import numpy as np

def test_matmul_shape_error():
    """
    Tests that a matmul operation with incompatible shapes raises an AxeError.
    """
    a = axe.array(np.random.randn(2, 3))
    b = axe.array(np.random.randn(4, 5)) # Incompatible shape

    with pytest.raises(axe.AxeError):
        a @ b