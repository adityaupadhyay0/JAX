from .. import array, no_grad, truediv
import numpy as np

class GradScaler:
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._scale = array(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self.enabled = enabled

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        if not self.enabled:
            return loss
        return loss * self._scale

    def _unscale_grads(self, optimizer):
        inv_scale_tensor = truediv(array(1.0), self._scale).data
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = p.grad * inv_scale_tensor

    def step(self, optimizer):
        if not self.enabled:
            optimizer.step()
            return

        # Unscale the gradients before checking for inf/nan
        self._unscale_grads(optimizer)

        # Check for invalid gradients
        found_inf_or_nan = False
        with no_grad():
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        # Use NumPy to check for inf/nan
                        grad_np = np.array(p.grad)
                        if not np.isfinite(grad_np).all():
                            found_inf_or_nan = True
                            break
                if found_inf_or_nan:
                    break

        # If invalid grads are found, skip optimizer step and decrease scale
        if found_inf_or_nan:
            self._scale = self._scale * array(self.backoff_factor)
            self._growth_tracker = 0
            optimizer.zero_grad() # Clear invalid gradients
            return

        # If grads are valid, step the optimizer
        optimizer.step()

        # Update the scale for the next iteration
        self._growth_tracker += 1
        if self._growth_tracker >= self.growth_interval:
            self._scale = self._scale * array(self.growth_factor)
            self._growth_tracker = 0

    def state_dict(self):
        return {
            "scale": np.array(self._scale.data).item(),
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict):
        self._scale = array(state_dict["scale"])
        self._growth_tracker = state_dict["growth_tracker"]