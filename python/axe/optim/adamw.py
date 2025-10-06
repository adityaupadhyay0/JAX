from .optimizer import Optimizer
from .. import no_grad, zeros, add, sub, mul, array, sqrt, truediv
from ..nn import Module

class AdamW(Optimizer):
    """
    Implements the AdamW algorithm.
    """

    def __init__(self, module: Module, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        Initializes the AdamW optimizer.
        Args:
            module: The `Module` whose parameters should be optimized.
            lr (float): The learning rate.
            betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
            weight_decay (float): Weight decay coefficient.
        """
        super().__init__(module)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {} # Use dict for state
        self.v = {} # Use dict for state

    def step(self, scaler=None):
        """
        Performs a single optimization step.
        """
        self.t += 1

        # Prune state for parameters that are no longer in the model
        current_param_ids = {id(p) for p in self.module.parameters()}
        for param_id in list(self.m.keys()):
            if param_id not in current_param_ids:
                del self.m[param_id]
                del self.v[param_id]

        with no_grad():
            for p in self.module.parameters():
                if p.grad is None:
                    continue

                param_id = id(p)

                # Initialize state for new parameters
                if param_id not in self.m:
                    self.m[param_id] = zeros(p.shape, requires_grad=False)
                    self.v[param_id] = zeros(p.shape, requires_grad=False)

                grad = p.grad
                if scaler:
                    inv_scale = array(1.0) / scaler.get_scale()
                    grad = mul(grad, inv_scale).data

                # Decoupled weight decay
                p.data = sub(p, mul(array(self.lr * self.weight_decay), p)).data

                # Update biased first moment estimate
                self.m[param_id] = add(mul(array(self.beta1), self.m[param_id]), mul(array(1 - self.beta1), grad))

                # Update biased second raw moment estimate
                grad_sq = mul(grad, grad)
                self.v[param_id] = add(mul(array(self.beta2), self.v[param_id]), mul(array(1 - self.beta2), grad_sq))

                # Compute bias-corrected first moment estimate
                m_hat = truediv(self.m[param_id], array(1 - self.beta1 ** self.t))

                # Compute bias-corrected second raw moment estimate
                v_hat = truediv(self.v[param_id], array(1 - self.beta2 ** self.t))

                # Update parameters
                update_val = truediv(mul(array(self.lr), m_hat), add(sqrt(v_hat), array(self.eps)))

                new_p = sub(p, update_val)
                p.data = new_p.data
                p.grad = None

    def __repr__(self):
        return f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), eps={self.eps}, weight_decay={self.weight_decay})"