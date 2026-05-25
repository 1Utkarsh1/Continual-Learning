from __future__ import annotations

import torch
from torch import nn

from cl_bench.strategies.base import ContinualLearningStrategy
from cl_bench.strategies.replay import ReservoirReplayBuffer


class AGEMStrategy(ContinualLearningStrategy):
    """A-GEM gradient projection against replay-memory reference gradients."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        buffer_size: int,
        memory_batch_size: int,
        seed: int,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.buffer = ReservoirReplayBuffer(capacity=buffer_size, seed=seed)
        self.memory_batch_size = memory_batch_size
        self.last_gradient_dot = 0.0
        self.last_projection_applied = 0.0

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del task_id
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        return (
            loss,
            logits,
            {
                "ce_loss": float(loss.detach().item()),
                "agem_gradient_dot": self.last_gradient_dot,
                "agem_projection_applied": self.last_projection_applied,
            },
        )

    def after_backward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int,
    ) -> None:
        del inputs, targets, task_id
        self.last_gradient_dot = 0.0
        self.last_projection_applied = 0.0
        if len(self.buffer) == 0 or self.memory_batch_size <= 0:
            return

        current_grad = _flatten_gradients(self.model)
        replay_inputs, replay_targets = self.buffer.sample(self.memory_batch_size)
        replay_inputs = replay_inputs.to(self.device)
        replay_targets = replay_targets.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        reference_loss = self.criterion(self.model(replay_inputs), replay_targets)
        reference_loss.backward()
        reference_grad = _flatten_gradients(self.model)

        dot_product = torch.dot(current_grad, reference_grad)
        self.last_gradient_dot = float(dot_product.detach().cpu().item())
        if dot_product < 0:
            reference_norm = torch.dot(reference_grad, reference_grad).clamp_min(1e-12)
            current_grad = current_grad - (dot_product / reference_norm) * reference_grad
            self.last_projection_applied = 1.0

        _assign_flattened_gradients(self.model, current_grad)

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
            "memory_batch_size": self.memory_batch_size,
        }


def _flatten_gradients(model: nn.Module) -> torch.Tensor:
    pieces = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(parameter.grad.detach().clone().reshape(-1))
    if not pieces:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.cat(pieces)


def _assign_flattened_gradients(model: nn.Module, flat_gradient: torch.Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        gradient = flat_gradient[offset : offset + count].view_as(parameter).clone()
        parameter.grad = gradient
        offset += count


def build_agem(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    buffer_size: int,
    memory_batch_size: int,
    seed: int,
) -> AGEMStrategy:
    return AGEMStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        memory_batch_size=memory_batch_size,
        seed=seed,
    )
