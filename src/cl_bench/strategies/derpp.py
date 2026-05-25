from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cl_bench.strategies.base import ContinualLearningStrategy
from cl_bench.strategies.replay import ReservoirReplayBuffer


class DERPPStrategy(ContinualLearningStrategy):
    """Dark Experience Replay++ with stored logits and replay labels."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        buffer_size: int,
        replay_batch_size: int,
        alpha: float,
        beta: float,
        seed: int,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.buffer = ReservoirReplayBuffer(capacity=buffer_size, seed=seed)
        self.replay_batch_size = replay_batch_size
        self.alpha = alpha
        self.beta = beta

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del task_id
        logits = self.model(inputs)
        ce_loss = self.criterion(logits, targets)
        distillation_loss = torch.zeros((), device=self.device)
        replay_ce_loss = torch.zeros((), device=self.device)

        if len(self.buffer) > 0 and self.replay_batch_size > 0:
            replay_inputs, replay_targets, replay_logits = self.buffer.sample_tensors(
                self.replay_batch_size,
                require_logits=True,
            )
            replay_inputs = replay_inputs.to(self.device)
            replay_targets = replay_targets.to(self.device)
            replay_logits = replay_logits.to(self.device)
            current_replay_logits = self.model(replay_inputs)
            distillation_loss = F.mse_loss(current_replay_logits, replay_logits)
            replay_ce_loss = self.criterion(current_replay_logits, replay_targets)

        loss = ce_loss + self.alpha * distillation_loss + self.beta * replay_ce_loss
        return (
            loss,
            logits,
            {
                "ce_loss": float(ce_loss.detach().item()),
                "derpp_distillation_loss": float(distillation_loss.detach().item()),
                "derpp_replay_ce_loss": float(replay_ce_loss.detach().item()),
            },
        )

    def observe_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        task_id: int,
    ) -> None:
        del task_id
        self.buffer.add_batch(inputs, targets, logits=logits)

    def extra_state_dict(self) -> dict[str, object]:
        return {
            "buffer_inputs": [sample.inputs for sample in self.buffer.samples],
            "buffer_targets": [sample.target for sample in self.buffer.samples],
            "buffer_logits": [sample.logits for sample in self.buffer.samples],
            "seen_count": self.buffer.seen_count,
            "alpha": self.alpha,
            "beta": self.beta,
        }


def build_derpp(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    buffer_size: int,
    replay_batch_size: int,
    alpha: float,
    beta: float,
    seed: int,
) -> DERPPStrategy:
    return DERPPStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        replay_batch_size=replay_batch_size,
        alpha=alpha,
        beta=beta,
        seed=seed,
    )
