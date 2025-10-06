from .. import array, no_grad
import numpy as np

class GradScaler:
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self._scale = array(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss * self._scale

    def step(self, optimizer):
        # First, check for invalid gradients (inf or NaN)
        found_inf_or_nan = False
        with no_grad():
            for param in optimizer.params:
                if param.grad is not None:
                    grad_np = np.array(param.grad)
                    if np.any(np.isinf(grad_np)) or np.any(np.isnan(grad_np)):
                        found_inf_or_nan = True
                        break

        # If invalid grads are found, skip the optimizer step and decrease the scale
        if found_inf_or_nan:
            self._scale *= array(self.backoff_factor)
            self._growth_tracker = 0
            optimizer.zero_grad() # Clear invalid gradients
            return

        # If grads are valid, step the optimizer
        optimizer.step(scaler=self)

        # Update the scale for the next iteration
        self._growth_tracker += 1
        if self._growth_tracker >= self.growth_interval:
            self._scale *= array(self.growth_factor)
            self._growth_tracker = 0

    def state_dict(self):
        return {
            "scale": np.array(self._scale.data).item(),
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict):
        self._scale = array(state_dict["scale"])
        self._growth_tracker = state_dict["growth_tracker"]