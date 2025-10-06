import math
from .module import Module
from .. import randn, zeros, matmul, add

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xW + b
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Using Kaiming He initialization for weights
        k_he = math.sqrt(1 / in_features)
        # The weight is stored in the shape (in_features, out_features) to be compatible with
        # the matmul operation for a batch of inputs: (N, C_in) @ (C_in, C_out) -> (N, C_out)
        self.weight = randn(in_features, out_features) * k_he
        self.weight.requires_grad = True

        if bias:
            self.bias = zeros(out_features)
            self.bias.requires_grad = True
        else:
            self.bias = None

    def forward(self, x):
        output = matmul(x, self.weight)
        if self.bias is not None:
            output = add(output, self.bias)
        return output

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"