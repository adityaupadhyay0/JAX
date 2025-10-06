import pytest
import axe
import axe.nn as nn
import axe.optim as optim
import math

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

def test_step_lr():
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.StepLR(optimizer, step_size=3, gamma=0.1)

    # Initial LR
    assert optimizer.param_groups[0]['lr'] == 0.1

    # Epoch 0
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1)

    # Epoch 1
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1)

    # Epoch 2
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1)

    # Epoch 3 - LR should decay
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.01)

    # Epoch 5
    scheduler.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.01)

    # Epoch 6 - LR should decay again
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.001)

def test_cosine_annealing_lr():
    model = SimpleModel()
    initial_lr = 0.1
    t_max = 10
    eta_min = 0.01
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = optim.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    # Initial LR
    assert optimizer.param_groups[0]['lr'] == initial_lr

    # Check learning rates over the cycle
    for epoch in range(t_max):
        scheduler.step() # last_epoch will be 0, 1, ..., 9
        expected_lr = eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * epoch / t_max)) / 2
        assert optimizer.param_groups[0]['lr'] == pytest.approx(expected_lr)

    # After t_max steps (i.e., at epoch 10), the LR should be eta_min
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(eta_min)

def test_lambda_lr():
    model = SimpleModel()
    initial_lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    # Simple lambda that halves the learning rate every epoch
    lr_lambda = lambda epoch: 0.5 ** epoch
    scheduler = optim.LambdaLR(optimizer, lr_lambda)

    # Initial LR
    assert optimizer.param_groups[0]['lr'] == initial_lr

    # Epoch 0
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * (0.5 ** 0))

    # Epoch 1
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * (0.5 ** 1))

    # Epoch 2
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * (0.5 ** 2))

    # Epoch 3
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * (0.5 ** 3))