from .optimizer import Optimizer
from .. import no_grad, mul, array, sub

class SGD(Optimizer):
    """
    Implements stochastic gradient descent.
    """

    def __init__(self, params, lr: float):
        """
        Initializes the SGD optimizer.

        Args:
            params: An iterable of `Variable`s to optimize.
            lr (float): The learning rate.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params)
        # Add lr to the param_groups
        for group in self.param_groups:
            group['lr'] = lr

    def step(self):
        """
        Performs a single optimization step.
        """
        with no_grad():
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        update = mul(array(lr), grad)
                        new_p = sub(p, update)
                        p.data = new_p.data
                        # We don't nullify grad here, zero_grad() is responsible for that

    def __repr__(self):
        return f"SGD(lr={self.param_groups[0]['lr']})"