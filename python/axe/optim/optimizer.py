from ..nn import Module

class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, module: Module):
        """
        Initializes the optimizer.

        Args:
            module: The `Module` whose parameters should be optimized.
        """
        if not isinstance(module, Module):
            raise TypeError(f"Optimizer expects a Module instance, but got {type(module)}")
        self.module = module

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
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad = None