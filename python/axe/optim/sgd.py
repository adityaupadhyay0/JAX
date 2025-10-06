from .optimizer import Optimizer
from .. import no_grad, mul, array, sub
from ..nn import Module

class SGD(Optimizer):
    """
    Implements stochastic gradient descent (optionally with momentum).
    """

    def __init__(self, module: Module, lr: float):
        """
        Initializes the SGD optimizer.

        Args:
            module: The `Module` whose parameters should be optimized.
            lr (float): The learning rate.
        """
        super().__init__(module)
        self.lr = lr

    def step(self, scaler=None):
        """
        Performs a single optimization step.
        """
        with no_grad():
            for p in self.module.parameters():
                if p.grad is not None:
                    grad = p.grad
                    if scaler:
                        inv_scale = array(1.0) / scaler.get_scale()
                        grad = mul(grad, inv_scale).data

                    update = mul(array(self.lr), grad)
                    new_p = sub(p, update)
                    p.data = new_p.data
                    p.grad = None

    def __repr__(self):
        return f"SGD(lr={self.lr})"