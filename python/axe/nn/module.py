from collections import OrderedDict
import inspect
from .. import Variable

class Module:
    """
    Base class for all neural network modules.

    Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes.
    """

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def __setattr__(self, name, value):
        if isinstance(value, Variable):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the module.
        Subclasses should override this method.
        """
        raise NotImplementedError(f"Module '{type(self).__name__}' has no forward method implemented.")

    def __call__(self, *args, **kwargs):
        """
        Calls the forward method of the module.
        """
        return self.forward(*args, **kwargs)

    def parameters(self):
        """
        Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        """
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            for param in module.parameters():
                yield param

    def train(self):
        """Sets the module in training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()
        return self

    def eval(self):
        """Sets the module in evaluation mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()
        return self

    def freeze(self):
        """Freezes the parameters of the module."""
        for param in self.parameters():
            param.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreezes the parameters of the module."""
        for param in self.parameters():
            param.requires_grad = True
        return self