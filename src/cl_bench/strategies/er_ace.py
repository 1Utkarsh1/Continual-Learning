from __future__ import annotations

import torch
from torch import nn

from cl_bench.strategies.base import ContinualLearningStrategy
from cl_bench.strategies.replay import ReservoirReplayBuffer


class ERACEStrategy(ContinualLearningStrategy):
    """Experience replay with asymmetric cross-entropy for class-incremental streams."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        buffer_size: int,
        replay_batch_size: int,
        replay_loss_weight: float,
        task_classes: list[list[int]],
        seed: int,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.buffer = ReservoirReplayBuffer(capacity=buffer_size, seed=seed)
        self.replay_batch_size = replay_batch_size
        self.replay_loss_weight = replay_loss_weight
        self.task_classes = task_classes

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        logits = self.model(inputs)
        ce_loss = self.criterion(_mask_to_current_task(logits, self.task_classes[task_id]), targets)
        replay_loss = torch.zeros((), device=self.device)

        if len(self.buffer) > 0 and self.replay_batch_size > 0:
            replay_inputs, replay_targets = self.buffer.sample(self.replay_batch_size)
            replay_inputs = replay_inputs.to(self.device)
            replay_targets = replay_targets.to(self.device)
            replay_logits = self.model(replay_inputs)
            replay_loss = self.criterion(replay_logits, replay_targets)

        loss = ce_loss + self.replay_loss_weight * replay_loss
        return (
            loss,
            logits,
            {
                "er_ace_current_ce_loss": float(ce_loss.detach().item()),
                "replay_loss": float(replay_loss.detach().item()),
            },
        )

    def observe_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        task_id: int,
    ) -> None:
        del logits, task_id
        self.buffer.add_batch(inputs, targets)

    def extra_state_dict(self) -> dict[str, object]:
        return {
            "buffer_inputs": [sample.inputs for sample in self.buffer.samples],
            "buffer_targets": [sample.target for sample in self.buffer.samples],
            "seen_count": self.buffer.seen_count,
            "replay_loss_weight": self.replay_loss_weight,
        }


def _mask_to_current_task(logits: torch.Tensor, classes: list[int]) -> torch.Tensor:
    masked_logits = logits.clone()
    allowed = torch.zeros(logits.size(1), dtype=torch.bool, device=logits.device)
    allowed[torch.tensor(classes, dtype=torch.long, device=logits.device)] = True
    masked_logits[:, ~allowed] = -1e9
    return masked_logits


def build_er_ace(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    buffer_size: int,
    replay_batch_size: int,
    replay_loss_weight: float,
    task_classes: list[list[int]],
    seed: int,
) -> ERACEStrategy:
    return ERACEStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        replay_batch_size=replay_batch_size,
        replay_loss_weight=replay_loss_weight,
        task_classes=task_classes,
        seed=seed,
    )
