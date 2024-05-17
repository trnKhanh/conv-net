from torch.optim import Optimizer
import math


class CosineSchedule(object):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_lr: float,
        target_lr: float,
        max_steps: int,
        cur_step: int = 0,
    ):
        self.optimizer = optimizer
        self.cur_step = 0
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def __call__(self, cur_step: int):
        if cur_step <= self.warmup_steps:
            return cur_step / self.warmup_steps * self.base_lr

        cur_step -= self.warmup_steps
        if cur_step <= self.max_steps:
            return self.target_lr + 0.5 * (self.base_lr - self.target_lr) * (
                1 + math.cos(math.pi * cur_step / self.max_steps)
            )
        else:
            return self.target_lr
    
    def step(self):
        self.cur_step += 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.__call__(self.cur_step)

