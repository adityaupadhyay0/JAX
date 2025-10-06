from .module import Module
from .. import ones, zeros, var, mean, sqrt, array, no_grad, add, mul

class BatchNorm2d(Module):
    """
    Applies Batch Normalization over a 4D input.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters, shaped for broadcasting
        self.weight = ones((1, num_features, 1, 1)) # gamma
        self.bias = zeros((1, num_features, 1, 1))   # beta
        self.weight.requires_grad = True
        self.bias.requires_grad = True

        # Non-learnable buffers for running statistics
        self.running_mean = zeros(num_features)
        self.running_var = ones(num_features)

    def forward(self, x):
        if self.training:
            # Calculate batch statistics
            # Mean and Var are calculated over N, H, W, for each C
            batch_mean = mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = var(x, axis=(0, 2, 3), keepdims=True)

            # Update running statistics
            with no_grad():
                batch_mean_squeezed = batch_mean.reshape((self.num_features,))
                batch_var_squeezed = batch_var.reshape((self.num_features,))

                # Update buffers in-place
                self.running_mean.data = ((1 - self.momentum) * self.running_mean + self.momentum * batch_mean_squeezed).data
                self.running_var.data = ((1 - self.momentum) * self.running_var + self.momentum * batch_var_squeezed).data

            # Normalize using batch statistics
            x_hat = (x - batch_mean) / sqrt(batch_var + array(self.eps))
        else:
            # Reshape running stats for broadcasting
            running_mean_reshaped = self.running_mean.reshape((1, self.num_features, 1, 1))
            running_var_reshaped = self.running_var.reshape((1, self.num_features, 1, 1))
            # Use running statistics for normalization
            x_hat = (x - running_mean_reshaped) / sqrt(running_var_reshaped + array(self.eps))

        # Scale and shift using broadcast-ready parameters
        return mul(self.weight, x_hat) + self.bias

    def __repr__(self):
        return f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"