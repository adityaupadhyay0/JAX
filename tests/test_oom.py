import pytest
import axe
import numpy as np

def test_oom_error_is_raised():
    """
    Tests that allocating a ridiculously large tensor triggers the custom OOMError.
    """
    # This is a bit of a tricky test to write because it depends on the
    # system's available memory. We'll request an absurdly large amount
    # that is almost guaranteed to fail on any system.
    # 100 billion floats = 400 GB
    insane_size = 100 * 1000 * 1000 * 1000

    with pytest.raises(axe.OOMError) as excinfo:
        t = axe.zeros([insane_size], dtype=axe.DType.Float32)

    # Check that the error message is helpful
    assert "Out of memory" in str(excinfo.value)
    assert "Try reducing tensor sizes" in str(excinfo.value)
    assert "using `axe.checkpoint`" in str(excinfo.value)