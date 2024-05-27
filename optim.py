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

    # Calculate the learning based on the warmup_steps, max_steps
    # using formula of CosineAnealing function
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

    # This is for consistency with Pytorch API
    def state_dict(self):
        return {
            "cur_step": self.cur_step,
            "base_lr": self.base_lr,
            "target_lr": self.target_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }

    # This is for consistency with Pytorch API
    def load_state_dict(self, state_dict):
        self.cur_step = state_dict["cur_step"]
        self.base_lr = state_dict["base_lr"]
        self.target_lr = state_dict["target_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.max_steps = state_dict["max_steps"]
