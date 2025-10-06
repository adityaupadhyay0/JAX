import math
from .module import Module
from .. import randn, zeros, conv2d

def _pair(x):
    if isinstance(x, int):
        return (x, x)
    return x

class Conv2D(Module):
    """
    Applies a 2D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding

        # Initialize weights with Kaiming He uniform initialization
        k_he = math.sqrt(1 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = randn(out_channels, in_channels, *self.kernel_size) * k_he
        self.weight.requires_grad = True

        if bias:
            self.bias = zeros(out_channels)
            self.bias.requires_grad = True
        else:
            self.bias = None

    def __call__(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")