from __future__ import annotations

import torch
from torch import nn

from cl_bench.strategies.base import ContinualLearningStrategy


class BaselineStrategy(ContinualLearningStrategy):
    """Naive sequential fine-tuning baseline."""

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del task_id
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        return loss, logits, {"ce_loss": float(loss.detach().item())}


def build_baseline(
    model: nn.Module, device: torch.device, learning_rate: float
) -> BaselineStrategy:
    return BaselineStrategy(model=model, device=device, learning_rate=learning_rate)
