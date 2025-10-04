import pytest
import axe
import numpy as np

def setup_function():
    """Reset memory stats before each test."""
    if hasattr(axe.memory, 'debug_clear_everything'):
        axe.memory.debug_clear_everything()

def test_checkpoint_saves_memory():
    """
    Tests that checkpointing reduces memory usage by not storing intermediate
    tensors for the backward pass.
    """
    axe.memory.debug_clear_everything()

    # Define a function with a large intermediate tensor
    def large_intermediate_fn(x):
        # Creates a large tensor that should NOT be saved by checkpointing
        large_tensor = axe.ones((1024, 1024)) # ~4MB
        return axe.sum(x * axe.sum(large_tensor))

    # --- Run without checkpointing ---
    x_no_checkpoint = axe.ones((1,), requires_grad=True)
    y_no_checkpoint = large_intermediate_fn(x_no_checkpoint)

    peak_mem_no_checkpoint = axe.memory.peak_bytes()

    # --- Run with checkpointing ---
    axe.memory.debug_clear_everything() # Reset for the next run

    checkpointed_fn = axe.checkpoint(large_intermediate_fn)
    x_checkpoint = axe.ones((1,), requires_grad=True)
    y_checkpoint = checkpointed_fn(x_checkpoint)

    peak_mem_checkpoint = axe.memory.peak_bytes()

    # The peak memory with checkpointing should be significantly less because
    # the large intermediate tensor is not stored.
    assert peak_mem_checkpoint < peak_mem_no_checkpoint

def test_checkpoint_grad_correctness():
    """
    Tests that the gradients computed with checkpointing are correct.
    """

    def f(x):
        y = x * x
        return axe.sum(y)

    # --- Standard gradient calculation ---
    x1 = axe.array([2.0, 3.0], requires_grad=True)
    grad_fn_standard = axe.grad(f)
    grad1 = grad_fn_standard(x1)

    # --- Checkpointed gradient calculation ---
    checkpointed_f = axe.checkpoint(f)
    x2 = axe.array([2.0, 3.0], requires_grad=True)
    grad_fn_checkpointed = axe.grad(checkpointed_f)
    grad2 = grad_fn_checkpointed(x2)

    # The gradients should be identical
    np.testing.assert_allclose(grad1.numpy(), grad2.numpy())
    np.testing.assert_allclose(grad1.numpy(), [4.0, 6.0]) # 2*x

def test_checkpoint_with_multiple_ops():
    """
    Test checkpointing a function with a chain of operations.
    """
    def f(x, y):
        a = x * y      # op1
        b = axe.sum(a) # op2
        c = b * 2.0    # op3
        return c

    checkpointed_f = axe.checkpoint(f)
    x = axe.array([1., 2., 3.], requires_grad=True)
    y = axe.array([4., 5., 6.], requires_grad=True)

    # --- Get checkpointed grads ---
    grad_fn = axe.grad(checkpointed_f, argnums=(0, 1))
    dx, dy = grad_fn(x, y)

    # --- Get standard grads for comparison ---
    x_std = axe.array([1., 2., 3.], requires_grad=True)
    y_std = axe.array([4., 5., 6.], requires_grad=True)
    grad_fn_std = axe.grad(f, argnums=(0, 1))
    dx_std, dy_std = grad_fn_std(x_std, y_std)

    np.testing.assert_allclose(dx.numpy(), dx_std.numpy())
    np.testing.assert_allclose(dy.numpy(), dy_std.numpy())