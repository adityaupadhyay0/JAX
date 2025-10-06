import pytest
import numpy as np
import axe
from axe import nn
from axe import optim

def test_freeze_and_unfreeze():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 1)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 1. Freeze the first linear layer
    model._modules['0'].freeze()

    # Check that the first layer's parameters are frozen
    for p in model._modules['0'].parameters():
        assert not p.requires_grad

    # Check that the second layer's parameters are not frozen
    for p in model._modules['1'].parameters():
        assert p.requires_grad

    # Store initial weights
    initial_params_0 = [np.array(p.data) for p in model._modules['0'].parameters()]
    initial_params_1 = [np.array(p.data) for p in model._modules['1'].parameters()]

    # 2. Run a training step
    x = axe.randn(4, 10)
    y_true = axe.randn(4, 1)

    optimizer.zero_grad()
    y_pred = model(x)
    loss = axe.mean((y_pred - y_true) ** 2)
    loss.backward()
    optimizer.step()

    # 3. Check that the frozen layer's weights have not changed
    for i, p in enumerate(model._modules['0'].parameters()):
        assert np.allclose(np.array(p.data), initial_params_0[i])

    # 4. Check that the unfrozen layer's weights have changed
    for i, p in enumerate(model._modules['1'].parameters()):
        assert not np.allclose(np.array(p.data), initial_params_1[i])

    # 5. Unfreeze the model and check that all parameters are trainable
    model.unfreeze()
    for p in model.parameters():
        assert p.requires_grad

def test_hot_swap_layer():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 1. Run a few training steps to populate optimizer state
    for _ in range(5):
        x = axe.randn(4, 10)
        y_true = axe.randn(4, 1)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = axe.mean((y_pred - y_true) ** 2)
        loss.backward()
        optimizer.step()

    # 2. Hot-swap the second layer
    old_layer_params = [p for p in model._modules['1'].parameters()]
    new_layer = nn.Linear(20, 1)
    model._modules['1'] = new_layer

    # 3. Manually update the optimizer's parameters to include the new layer
    optimizer.param_groups[0]['params'] = list(model.parameters())

    # 4. Run one more training step
    x = axe.randn(4, 10)
    y_true = axe.randn(4, 1)
    optimizer.zero_grad()
    y_pred = model(x)
    loss = axe.mean((y_pred - y_true) ** 2)
    loss.backward()
    optimizer.step()

    # 5. Check that the optimizer has created state for the new layer's parameters
    for p in new_layer.parameters():
        assert id(p) in optimizer.state

    # 6. Check that the optimizer's state for the old layer is still present
    # (since we haven't implemented pruning yet)
    for p in old_layer_params:
        assert id(p) in optimizer.state