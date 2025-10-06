import pytest
import numpy as np
import axe
from axe import nn
from axe import optim

def test_cast():
    a = axe.ones((2, 2), dtype='float32')
    assert a.dtype == axe.DType.Float32

    b = a.data.cast(axe.DType.Float16)
    assert b.dtype == axe.DType.Float16

    c = b.cast(axe.DType.Float32)
    assert c.dtype == axe.DType.Float32

    # Check that the data is preserved through the cast
    assert np.allclose(np.array(a.data), np.array(c))

def test_autocast():
    a = axe.ones((2, 2), dtype='float32')
    b = axe.ones((2, 2), dtype='float32')

    # Outside the autocast context, the output should be float32
    c = axe.matmul(a, b)
    assert c.dtype == axe.DType.Float32

    # Inside the autocast context, the output should be float16
    with axe.amp.autocast():
        d = axe.matmul(a, b)
        assert d.dtype == axe.DType.Float16

    # Check that the context is properly exited
    e = axe.matmul(a, b)
    assert e.dtype == axe.DType.Float32

def test_amp_end_to_end():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 1)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = axe.amp.GradScaler()

    # Get initial parameter values to check for updates
    initial_params = [np.array(p.data) for p in model.parameters()]

    for _ in range(5):
        x = axe.randn(4, 10)
        y_true = axe.randn(4, 1)

        optimizer.zero_grad()

        with axe.amp.autocast():
            y_pred = model(x)

        # loss must be float32, calculated outside autocast
        loss = axe.mean((y_pred.cast('float32') - y_true) ** 2)
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)

    # Check that the parameters have been updated
    for i, p in enumerate(model.parameters()):
        p_np = np.array(p.data)
        initial_p_np = np.array(initial_params[i])
        assert not np.allclose(p_np, initial_p_np)


class DecoratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)

    @axe.amp
    def forward(self, x):
        # linear1 and linear2 will be autocasted due to the decorator
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def test_amp_decorator():
    """
    Tests the @axe.amp decorator for end-to-end mixed-precision training.
    """
    model = DecoratedModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Use a smaller initial scale to prevent overflow in this simple test case
    scaler = axe.amp.GradScaler(init_scale=1024.0)

    # Get initial parameter values to check for updates
    initial_params = [np.array(p.data) for p in model.parameters()]

    x = axe.randn(4, 10)
    y_true = axe.randn(4, 1)

    optimizer.zero_grad()

    # The forward pass is wrapped by the decorator, so it's autocasted
    y_pred = model(x)
    assert y_pred.dtype == axe.DType.Float16, "Output of decorated function should be float16"

    # Loss must be float32 for stability
    loss = axe.mean((y_pred.cast('float32') - y_true) ** 2)

    # Scale loss, backpropagate, and update weights
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(optimizer)

    # Check that the parameters have been updated
    updated_params = [np.array(p.data) for p in model.parameters()]
    for i, p_init in enumerate(initial_params):
        assert not np.allclose(p_init, updated_params[i]), f"Parameter {i} was not updated."