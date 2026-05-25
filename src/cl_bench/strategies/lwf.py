from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cl_bench.strategies.base import ContinualLearningStrategy


class LwFStrategy(ContinualLearningStrategy):
    """Learning without Forgetting using a frozen teacher from the previous task."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        alpha: float,
        temperature: float,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_model: nn.Module | None = None

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del task_id
        logits = self.model(inputs)
        ce_loss = self.criterion(logits, targets)
        distillation_loss = torch.zeros((), device=self.device)

        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)
            temperature = self.temperature
            student_log_probs = F.log_softmax(logits / temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            distillation_loss = (
                F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                * temperature
                * temperature
            )

        loss = ce_loss + self.alpha * distillation_loss
        return (
            loss,
            logits,
            {
                "ce_loss": float(ce_loss.detach().item()),
                "distillation_loss": float(distillation_loss.detach().item()),
            },
        )

    def after_task(self, train_loader: DataLoader, task_id: int) -> None:
        del train_loader, task_id
        self.teacher_model = copy.deepcopy(self.model).to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

    def extra_state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "temperature": self.temperature,
            "teacher_state_dict": None
            if self.teacher_model is None
            else self.teacher_model.state_dict(),
        }

    def load_extra_state_dict(self, state: dict[str, Any]) -> None:
        self.alpha = float(state.get("alpha", self.alpha))
        self.temperature = float(state.get("temperature", self.temperature))
        teacher_state = state.get("teacher_state_dict")
        if teacher_state is None:
            self.teacher_model = None
            return
        self.teacher_model = copy.deepcopy(self.model).to(self.device)
        self.teacher_model.load_state_dict(teacher_state)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)


def build_lwf(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    alpha: float,
    temperature: float,
) -> LwFStrategy:
    return LwFStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        alpha=alpha,
        temperature=temperature,
    )
