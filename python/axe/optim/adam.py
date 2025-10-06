from .optimizer import Optimizer
from .. import no_grad, zeros, add, sub, mul, array, sqrt, truediv

class Adam(Optimizer):
    """
    Implements the Adam algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """
        Initializes the Adam optimizer.

        Args:
            params: An iterable of `Variable`s to optimize.
            lr (float): The learning rate.
            betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params)
        for group in self.param_groups:
            group.update(defaults)

        self.state = {}

    def step(self):
        """
        Performs a single optimization step.
        """
        with no_grad():
            for group in self.param_groups:
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_id = id(p)
                    if param_id not in self.state:
                        self.state[param_id] = {
                            'step': 0,
                            'exp_avg': zeros(p.shape, requires_grad=False),
                            'exp_avg_sq': zeros(p.shape, requires_grad=False)
                        }

                    state = self.state[param_id]
                    state['step'] += 1

                    grad = p.grad
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    # Update biased first moment estimate
                    exp_avg.data = add(mul(array(beta1), exp_avg), mul(array(1 - beta1), grad)).data

                    # Update biased second raw moment estimate
                    grad_sq = mul(grad, grad)
                    exp_avg_sq.data = add(mul(array(beta2), exp_avg_sq), mul(array(1 - beta2), grad_sq)).data

                    # Compute bias-corrected first moment estimate
                    bias_correction1 = 1 - beta1 ** state['step']
                    m_hat = truediv(exp_avg, array(bias_correction1))

                    # Compute bias-corrected second raw moment estimate
                    bias_correction2 = 1 - beta2 ** state['step']
                    v_hat = truediv(exp_avg_sq, array(bias_correction2))

                    # Update parameters
                    update_val = truediv(mul(array(lr), m_hat), add(sqrt(v_hat), array(eps)))

                    p.data = sub(p, update_val).data

    def __repr__(self):
        group = self.param_groups[0]
        return f"Adam(lr={group['lr']}, betas=({group['betas'][0]}, {group['betas'][1]}), eps={group['eps']})"