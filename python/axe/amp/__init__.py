import functools
import sys
from .autocast_mode import autocast
from .grad_scaler import GradScaler

class _AmpModule(object):
    """
    This class is a proxy for the `amp` module, allowing it to be used as a decorator
    while also providing access to its members like `autocast` and `GradScaler`.
    """
    def __init__(self, module_name):
        self._module = sys.modules[module_name]
        sys.modules[module_name] = self
        self.autocast = autocast
        self.GradScaler = GradScaler

    def __call__(self, func):
        """
        A decorator that enables automatic mixed precision for the decorated function.
        It wraps the function call in an `axe.amp.autocast` context, which automatically
        casts operations to a lower-precision format like float16 for performance.

        Example:
            @axe.amp
            def model_forward(input):
                # Operations inside this function will be autocasted
                return model(input)
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.autocast():
                return func(*args, **kwargs)
        return wrapper

# This makes the module callable. So you can use `@axe.amp` as a decorator
# and still access `axe.amp.GradScaler` and `axe.amp.autocast`.
_AmpModule(__name__)