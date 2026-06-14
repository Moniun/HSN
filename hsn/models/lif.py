from __future__ import annotations

import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        # triangular surrogate gradient around threshold
        grad = (1.0 - x.abs()).clamp(min=0.0)
        return grad_output * grad


class LIFSpike(nn.Module):
    def __init__(self, threshold: float = 1.0, decay: float = 0.5):
        super().__init__()
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.mem = None

    def reset_state(self) -> None:
        self.mem = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        self.mem = self.mem * self.decay + x
        spike = SpikeFn.apply(self.mem - self.threshold)
        self.mem = self.mem - spike.detach() * self.threshold
        return spike


def reset_lif_states(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, LIFSpike):
            m.reset_state()
