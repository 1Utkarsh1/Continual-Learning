from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cl_bench.strategies.base import ContinualLearningStrategy, clone_state_dict, load_state_dict


class JointTrainingStrategy(ContinualLearningStrategy):
    """Oracle cumulative-data upper bound over all examples observed so far."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        batch_size: int,
        seed: int,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.batch_size = batch_size
        self.seed = seed
        self.initial_state = clone_state_dict(model)
        self.seen_inputs: list[torch.Tensor] = []
        self.seen_targets: list[torch.Tensor] = []

    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_id: int,
        epochs: int,
    ) -> list[dict[str, float | int]]:
        self.current_task = task_id
        self.seen_tasks = max(self.seen_tasks, task_id + 1)
        self.before_task(task_id)
        self._collect_task(train_loader)
        load_state_dict(self.model, self.initial_state, self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(epochs)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        history: list[dict[str, float | int]] = []
        joint_loader = self._joint_loader(task_id)
        for epoch in range(epochs):
            train_metrics = self._train_epoch(joint_loader, task_id)
            val_metrics = self.evaluate(val_loader)
            epoch_metrics = {
                "task_id": task_id,
                "epoch": epoch + 1,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(epoch_metrics)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                best_state = clone_state_dict(self.model)
            if self.scheduler is not None:
                self.scheduler.step()

        if best_state is not None:
            load_state_dict(self.model, best_state, self.device)
        self.after_task(train_loader, task_id)
        return history

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del task_id
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        return loss, logits, {"ce_loss": float(loss.detach().item())}

    def extra_state_dict(self) -> dict[str, object]:
        return {
            "seen_examples": sum(int(targets.numel()) for targets in self.seen_targets),
            "batch_size": self.batch_size,
        }

    def _collect_task(self, train_loader: DataLoader) -> None:
        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for batch_inputs, batch_targets in train_loader:
            inputs.append(batch_inputs.detach().cpu())
            targets.append(batch_targets.detach().cpu())
        if inputs:
            self.seen_inputs.append(torch.cat(inputs, dim=0))
            self.seen_targets.append(torch.cat(targets, dim=0))

    def _joint_loader(self, task_id: int) -> DataLoader:
        if not self.seen_inputs:
            raise ValueError("Joint training has no collected examples.")
        dataset = TensorDataset(
            torch.cat(self.seen_inputs, dim=0), torch.cat(self.seen_targets, dim=0)
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed + task_id),
        )


def build_joint(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> JointTrainingStrategy:
    return JointTrainingStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
