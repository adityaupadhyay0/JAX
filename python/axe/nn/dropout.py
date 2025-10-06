from .module import Module
from .. import array, rand

class Dropout(Module):
    """
    During training, randomly zeroes some of the elements of the input
    tensor with probability `p`.

    The elements to zero are randomized on every forward call.
    """

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x

        if self.p == 1:
            return x * array(0.0)

        # Create a mask from a uniform distribution
        mask_tensor = rand(*x.shape)

        # Inverted dropout: scale the output by 1/(1-p) during training
        scale_factor = 1.0 / (1.0 - self.p)

        # Create a boolean mask where True indicates elements to keep
        import numpy as np
        mask_np = np.array(mask_tensor.data) > self.p
        mask = array(mask_np, dtype='float32')

        # Apply the mask and scale
        masked_x = x * mask
        return masked_x * array(scale_factor)

    def __repr__(self):
        return f"Dropout(p={self.p})"