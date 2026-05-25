from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


class ContinualLearningStrategy(ABC):
    """Stable lifecycle interface shared by all continual-learning methods."""

    def __init__(self, model: nn.Module, device: torch.device, learning_rate: float):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer_name = "adam"
        self.momentum = 0.9
        self.weight_decay = 0.0
        self.scheduler_name = "none"
        self.warmup_epochs = 0
        self.label_smoothing = 0.0
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.current_task = -1
        self.seen_tasks = 0

    def configure_training(
        self,
        optimizer: str,
        momentum: float,
        weight_decay: float,
        scheduler: str,
        warmup_epochs: int,
        label_smoothing: float,
    ) -> None:
        self.optimizer_name = optimizer.lower().replace("-", "_")
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler.lower().replace("-", "_")
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = self._build_optimizer()

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
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(epochs)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        history: list[dict[str, float | int]] = []

        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader, task_id)
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

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                total_loss += float(loss.item()) * inputs.size(0)
                total_correct += int((logits.argmax(dim=1) == targets).sum().item())
                total_examples += int(targets.numel())

        if total_examples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "examples": 0.0}
        return {
            "loss": total_loss / total_examples,
            "accuracy": 100.0 * total_correct / total_examples,
            "examples": float(total_examples),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "current_task": self.current_task,
                "seen_tasks": self.seen_tasks,
                "strategy_state": self.extra_state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_task = int(checkpoint["current_task"])
        self.seen_tasks = int(checkpoint["seen_tasks"])
        self.load_extra_state_dict(checkpoint.get("strategy_state", {}))

    def before_task(self, task_id: int) -> None:
        del task_id

    def after_task(self, train_loader: DataLoader, task_id: int) -> None:
        del train_loader, task_id

    def extra_state_dict(self) -> dict[str, Any]:
        return {}

    def run_summary(self) -> dict[str, float | int | str | None]:
        return {}

    def load_extra_state_dict(self, state: dict[str, Any]) -> None:
        del state

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _build_scheduler(self, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.scheduler_name in {"none", "constant", ""}:
            return None
        if self.scheduler_name == "cosine":
            warmup_epochs = min(max(0, self.warmup_epochs), max(0, epochs - 1))

            def lr_lambda(epoch: int) -> float:
                if warmup_epochs and epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                cosine_epochs = max(1, epochs - warmup_epochs)
                progress = (epoch - warmup_epochs + 1) / float(cosine_epochs)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")

    def _train_epoch(self, train_loader: DataLoader, task_id: int) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {"loss": 0.0, "ce_loss": 0.0}
        total_correct = 0
        total_examples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            loss, logits, components = self.compute_loss(inputs, targets, task_id)
            loss.backward()
            self.after_backward(inputs, targets, task_id)
            self.optimizer.step()
            self.observe_batch(inputs, targets, logits, task_id)

            batch_size = int(targets.numel())
            total_examples += batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            totals["loss"] += float(loss.item()) * batch_size
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_size

        if total_examples == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        metrics = {name: value / total_examples for name, value in totals.items()}
        metrics["accuracy"] = 100.0 * total_correct / total_examples
        return metrics

    @abstractmethod
    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def after_backward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int,
    ) -> None:
        del inputs, targets, task_id

    def observe_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        task_id: int,
    ) -> None:
        del inputs, targets, logits, task_id


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Deep-copy a module state dict onto CPU to avoid mutable checkpoint aliases."""

    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def load_state_dict(
    module: nn.Module, state_dict: dict[str, torch.Tensor], device: torch.device
) -> None:
    module.load_state_dict({name: tensor.to(device) for name, tensor in state_dict.items()})
