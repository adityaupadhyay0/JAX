import pytest
import numpy as np
import axe
from axe import nn
from axe import optim

def test_linear_layer_and_sgd_optimizer():
    # 1. Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 1)

        def __call__(self, x):
            x = self.linear1(x)
            # A simple ReLU-like activation
            x = axe.mul(x, axe.array(np.array(x.data) > 0, dtype='float32'))
            x = self.linear2(x)
            return x

    model = SimpleModel()

    # 2. Create an SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 3. Generate synthetic data
    X = axe.randn(100, 10)
    # y = 2 * sum(X) + 1
    y_data = np.sum(np.array(X.data), axis=1, keepdims=True) * 2 + 1
    y = axe.array(y_data)


    # 4. Run a few training iterations
    initial_loss = None
    final_loss = None

    for i in range(10):
        optimizer.zero_grad()

        # a. Forward pass
        y_pred = model(X)

        # b. Calculate a loss (Mean Squared Error)
        diff = axe.sub(y_pred, y)
        loss = axe.mean(axe.mul(diff, diff))

        if i == 0:
            initial_loss = np.array(loss.data).item()

        # c. Backward pass
        loss.backward()

        # d. Update weights
        optimizer.step()

        if i == 9:
            final_loss = np.array(loss.data).item()

    # 5. Assert that the loss has decreased
    assert final_loss < initial_loss
    assert not np.isnan(final_loss)
    assert not np.isinf(final_loss)

    # Check if parameters have been updated
    for param in model.parameters():
        assert param.grad is None # Grads should be cleared by optimizer

def test_batchnorm2d():
    N, C, H, W = 4, 3, 5, 5
    bn = nn.BatchNorm2d(C)

    # Test training mode
    bn.train()
    x = axe.randn(N, C, H, W)
    y = bn(x)

    # Check output stats
    y_np = np.array(y.data)
    # Mean should be close to 0 and var close to 1 for each channel
    assert np.allclose(np.mean(y_np, axis=(0, 2, 3)), 0, atol=1e-6)
    assert np.allclose(np.var(y_np, axis=(0, 2, 3)), 1, atol=1e-4)

    # Check that running stats have been updated
    assert not np.all(np.array(bn.running_mean.data) == 0)
    assert not np.all(np.array(bn.running_var.data) == 1)

    # Test evaluation mode
    bn.eval()
    x_eval = axe.randn(N, C, H, W)
    y_eval = bn(x_eval)
    y_eval_np = np.array(y_eval.data)

    # Check that running stats are not updated further in eval mode
    running_mean_before = np.copy(np.array(bn.running_mean.data))
    running_var_before = np.copy(np.array(bn.running_var.data))
    bn(x_eval) # another forward pass
    assert np.all(np.array(bn.running_mean.data) == running_mean_before)
    assert np.all(np.array(bn.running_var.data) == running_var_before)

    # Check gradients
    bn.train()
    x.requires_grad = True
    y = bn(x)
    loss = y.sum()
    loss.backward()

    assert bn.weight.grad is not None
    assert bn.weight.grad.shape == bn.weight.shape
    assert bn.bias.grad is not None
    assert bn.bias.grad.shape == bn.bias.shape

def test_conv2d_backward():
    N, C_in, H_in, W_in = 2, 3, 5, 5
    C_out, K, S, P = 4, 3, 1, 1

    conv = nn.Conv2D(C_in, C_out, kernel_size=K, stride=S, padding=P)
    x = axe.randn(N, C_in, H_in, W_in)
    x.requires_grad = True

    y = conv(x)

    # Create a dummy loss and backpropagate
    loss = y.sum()
    loss.backward()

    # Check weight gradients
    assert conv.weight.grad is not None
    assert conv.weight.grad.shape == conv.weight.shape

    # Check bias gradients
    assert conv.bias.grad is not None
    assert conv.bias.grad.shape == conv.bias.shape

    # Check input gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_conv2d_forward_shape():
    N, C_in, H_in, W_in = 4, 3, 28, 28
    C_out, K, S, P = 16, 3, 1, 1

    conv = nn.Conv2D(C_in, C_out, kernel_size=K, stride=S, padding=P)
    x = axe.randn(N, C_in, H_in, W_in)

    y = conv(x)

    H_out = (H_in + 2 * P - K) // S + 1
    W_out = (W_in + 2 * P - K) // S + 1

    assert y.shape == [N, C_out, H_out, W_out]

def test_dropout_layer():
    # Test dropout during training
    p = 0.5
    dropout = nn.Dropout(p=p)
    dropout.train()
    x = axe.ones((100, 100))
    y = dropout(x)

    y_np = np.array(y.data)
    # Check that some values are zeroed out
    assert np.any(y_np == 0)
    # Check that some values are scaled
    assert np.any(y_np != 0)

    # Check that the mean is approximately the original mean (1.0)
    assert np.allclose(np.mean(y_np), 1.0, atol=0.1)


    # Test dropout during evaluation
    dropout.eval()
    x_eval = axe.ones((100, 100))
    y_eval = dropout(x_eval)
    # Check that output is identical to input
    assert np.all(np.array(y_eval.data) == np.array(x_eval.data))

    # Test with p=0, should be identity
    dropout_zero = nn.Dropout(p=0)
    dropout_zero.train()
    x_p0 = axe.ones((10,10))
    y_p0 = dropout_zero(x_p0)
    assert np.all(np.array(y_p0.data) == np.array(x_p0.data))

    # Test with p=1, should be all zeros
    dropout_one = nn.Dropout(p=1)
    dropout_one.train()
    x_p1 = axe.ones((10,10))
    y_p1 = dropout_one(x_p1)
    assert np.all(np.array(y_p1.data) == 0)

def test_linear_layer_and_adamw_optimizer():
    # 1. Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 1)

        def __call__(self, x):
            x = self.linear1(x)
            # A simple ReLU-like activation
            x = axe.mul(x, axe.array(np.array(x.data) > 0, dtype='float32'))
            x = self.linear2(x)
            return x

    model = SimpleModel()

    # 2. Create an AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    # 3. Generate synthetic data
    X = axe.randn(100, 10)
    # y = 2 * sum(X) + 1
    y_data = np.sum(np.array(X.data), axis=1, keepdims=True) * 2 + 1
    y = axe.array(y_data)


    # 4. Run a few training iterations
    initial_loss = None
    final_loss = None

    for i in range(10):
        optimizer.zero_grad()

        # a. Forward pass
        y_pred = model(X)

        # b. Calculate a loss (Mean Squared Error)
        diff = axe.sub(y_pred, y)
        loss = axe.mean(axe.mul(diff, diff))

        if i == 0:
            initial_loss = np.array(loss.data).item()

        # c. Backward pass
        loss.backward()

        # d. Update weights
        optimizer.step()

        if i == 9:
            final_loss = np.array(loss.data).item()

    # 5. Assert that the loss has decreased
    assert final_loss < initial_loss
    assert not np.isnan(final_loss)
    assert not np.isinf(final_loss)

    # Check if parameters have been updated
    for param in model.parameters():
        assert param.grad is None # Grads should be cleared by optimizer
        # A more robust check would be to store initial param values and compare
        # but for this test, loss decrease is a strong indicator.

def test_linear_layer_and_adam_optimizer():
    # 1. Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 1)

        def __call__(self, x):
            x = self.linear1(x)
            # A simple ReLU-like activation
            x = axe.mul(x, axe.array(np.array(x.data) > 0, dtype='float32'))
            x = self.linear2(x)
            return x

    model = SimpleModel()

    # 2. Create an Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 3. Generate synthetic data
    X = axe.randn(100, 10)
    # y = 2 * sum(X) + 1
    y_data = np.sum(np.array(X.data), axis=1, keepdims=True) * 2 + 1
    y = axe.array(y_data)


    # 4. Run a few training iterations
    initial_loss = None
    final_loss = None

    for i in range(10):
        optimizer.zero_grad()

        # a. Forward pass
        y_pred = model(X)

        # b. Calculate a loss (Mean Squared Error)
        diff = axe.sub(y_pred, y)
        loss = axe.mean(axe.mul(diff, diff))

        if i == 0:
            initial_loss = np.array(loss.data).item()

        # c. Backward pass
        loss.backward()

        # d. Update weights
        optimizer.step()

        if i == 9:
            final_loss = np.array(loss.data).item()

    # 5. Assert that the loss has decreased
    assert final_loss < initial_loss
    assert not np.isnan(final_loss)
    assert not np.isinf(final_loss)

    # Check if parameters have been updated
    for param in model.parameters():
        assert param.grad is None # Grads should be cleared by optimizer