from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cl_bench.strategies.base import ContinualLearningStrategy, clone_state_dict, load_state_dict
from cl_bench.strategies.replay import BalancedReplayBuffer


class GDumbStrategy(ContinualLearningStrategy):
    """Greedy class-balanced memory with from-scratch training on stored examples."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        buffer_size: int,
        memory_epochs: int,
        batch_size: int,
        seed: int,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.buffer = BalancedReplayBuffer(capacity=buffer_size, seed=seed)
        self.initial_state = clone_state_dict(model)
        self.memory_epochs = memory_epochs
        self.batch_size = batch_size
        self.seed = seed

    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_id: int,
        epochs: int,
    ) -> list[dict[str, float | int]]:
        del epochs
        self.current_task = task_id
        self.seen_tasks = max(self.seen_tasks, task_id + 1)
        self.before_task(task_id)

        example_count = 0
        for inputs, targets in train_loader:
            self.buffer.add_batch(inputs, targets)
            example_count += int(targets.numel())

        self._fit_memory(task_id)
        val_metrics = self.evaluate(val_loader)
        return [
            {
                "task_id": task_id,
                "epoch": 1,
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "train_examples": float(example_count),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        ]

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del targets, task_id
        logits = self.model(inputs)
        return logits.sum() * 0.0, logits, {"gdumb_online_loss": 0.0}

    def observe_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        task_id: int,
    ) -> None:
        del logits, task_id
        self.buffer.add_batch(inputs, targets)

    def _fit_memory(self, task_id: int) -> None:
        if len(self.buffer) == 0 or self.memory_epochs <= 0:
            return
        load_state_dict(self.model, self.initial_state, self.device)
        self.optimizer = self._build_optimizer()
        inputs, targets = self.buffer.tensors()
        memory_dataset = TensorDataset(inputs, targets)
        memory_loader = DataLoader(
            memory_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed + task_id),
        )
        for _ in range(self.memory_epochs):
            self.model.train()
            for memory_inputs, memory_targets in memory_loader:
                memory_inputs = memory_inputs.to(self.device)
                memory_targets = memory_targets.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(self.model(memory_inputs), memory_targets)
                loss.backward()
                self.optimizer.step()

    def extra_state_dict(self) -> dict[str, object]:
        return {
            "buffer_inputs": [sample.inputs for sample in self.buffer.samples],
            "buffer_targets": [sample.target for sample in self.buffer.samples],
            "seen_count": self.buffer.seen_count,
            "memory_epochs": self.memory_epochs,
            "class_counts": dict(self.buffer.class_counts()),
        }


def build_gdumb(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    buffer_size: int,
    memory_epochs: int,
    batch_size: int,
    seed: int,
) -> GDumbStrategy:
    return GDumbStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        memory_epochs=memory_epochs,
        batch_size=batch_size,
        seed=seed,
    )
