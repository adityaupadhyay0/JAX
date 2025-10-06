import math
from .optimizer import Optimizer

class LRScheduler:
    """
    Base class for learning rate schedulers.
    A scheduler is associated with an optimizer and adjusts the learning rate
    based on some schedule.
    """
    def __init__(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.last_epoch = -1
        self._initial_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        """
        Compute what the learning rate should be for the current epoch.
        """
        raise NotImplementedError

    def step(self):
        """
        Update the learning rate for the optimizer's param groups.
        This should be called after every epoch.
        """
        self.last_epoch += 1
        new_lrs = self.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = new_lrs[i]

class StepLR(LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every step_size epochs.
    This is a stateless implementation.
    """
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        # Calculate the learning rate based on the initial LR, not the previous one.
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self._initial_lrs]

class CosineAnnealingLR(LRScheduler):
    """
    Set the learning rate of each parameter group using a cosine annealing schedule.
    This is a stateless implementation based on the closed-form formula.
    """
    def __init__(self, optimizer, T_max, eta_min=0):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        # Using the closed-form equation for cosine annealing
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self._initial_lrs]

class LambdaLR(LRScheduler):
    """
    Sets the learning rate of each parameter group to the initial lr times a given function.
    The function is called with the current epoch (0-indexed).
    """
    def __init__(self, optimizer, lr_lambda):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self._initial_lrs]