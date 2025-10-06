from collections import OrderedDict
from .module import Module

class Sequential(Module):
    """A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can be passed in.
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for i, module in enumerate(args):
                self.add_module(str(i), module)

    def add_module(self, name, module):
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module)} is not a Module subclass")
        if not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {type(name)}")
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        if '.' in name:
            raise KeyError("module name can't contain \".\"")
        if name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def __call__(self, input):
        for module in self._modules.values():
            input = module(input)
        return input