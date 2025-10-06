from .. import Variable

class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, params):
        """
        Initializes the optimizer.

        Args:
            params: An iterable of `Variable`s to optimize. This can be a generator
                    or a list.
        """
        # Convert generator to list to be able to iterate multiple times
        param_list = list(params)

        if not param_list:
            raise ValueError("Optimizer received an empty parameter list.")
        if not all(isinstance(p, Variable) for p in param_list):
            raise TypeError("Optimizer was passed an iterable that does not contain only Variables.")

        # The main parameter group
        self.param_groups = [{'params': param_list}]

    def step(self):
        """
        Performs a single optimization step.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Clears the gradients of all optimized `Variable`s.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None