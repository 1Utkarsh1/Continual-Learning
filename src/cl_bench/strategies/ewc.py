from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cl_bench.strategies.base import ContinualLearningStrategy, clone_state_dict


class EWCStrategy(ContinualLearningStrategy):
    """Elastic Weight Consolidation with deterministic empirical Fisher estimates."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        ewc_lambda: float,
        fisher_samples: int,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.fishers: list[dict[str, torch.Tensor]] = []
        self.optimal_parameters: list[dict[str, torch.Tensor]] = []

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del task_id
        logits = self.model(inputs)
        ce_loss = self.criterion(logits, targets)
        ewc_penalty = self._ewc_penalty()
        loss = ce_loss + self.ewc_lambda * ewc_penalty
        return (
            loss,
            logits,
            {
                "ce_loss": float(ce_loss.detach().item()),
                "ewc_penalty": float(ewc_penalty.detach().item()),
            },
        )

    def after_task(self, train_loader: DataLoader, task_id: int) -> None:
        del task_id
        self.fishers.append(self._estimate_fisher(train_loader))
        self.optimal_parameters.append(clone_state_dict(self.model))

    def extra_state_dict(self) -> dict[str, object]:
        return {
            "ewc_lambda": self.ewc_lambda,
            "fisher_samples": self.fisher_samples,
            "fishers": self.fishers,
            "optimal_parameters": self.optimal_parameters,
        }

    def load_extra_state_dict(self, state: dict[str, object]) -> None:
        self.ewc_lambda = float(state.get("ewc_lambda", self.ewc_lambda))
        self.fisher_samples = int(state.get("fisher_samples", self.fisher_samples))
        self.fishers = [
            {name: tensor.to(self.device) for name, tensor in fisher.items()}
            for fisher in state.get("fishers", [])
        ]
        self.optimal_parameters = [
            {name: tensor.to(self.device) for name, tensor in params.items()}
            for params in state.get("optimal_parameters", [])
        ]

    def _ewc_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), device=self.device)
        if not self.fishers:
            return penalty

        named_parameters = dict(self.model.named_parameters())
        for fisher, optimum in zip(self.fishers, self.optimal_parameters, strict=True):
            for name, parameter in named_parameters.items():
                if not parameter.requires_grad:
                    continue
                fisher_tensor = fisher[name].to(self.device)
                optimum_tensor = optimum[name].to(self.device)
                penalty = penalty + 0.5 * torch.sum(
                    fisher_tensor * torch.square(parameter - optimum_tensor)
                )
        return penalty

    def _estimate_fisher(self, data_loader: DataLoader) -> dict[str, torch.Tensor]:
        self.model.eval()
        fisher_loader = DataLoader(
            data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=0,
        )
        fisher = {
            name: torch.zeros_like(parameter, device=self.device)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        sample_count = 0
        max_samples = self.fisher_samples if self.fisher_samples > 0 else float("inf")

        for inputs, targets in fisher_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            for index in range(inputs.size(0)):
                if sample_count >= max_samples:
                    break
                self.model.zero_grad(set_to_none=True)
                logits = self.model(inputs[index : index + 1])
                log_probability = F.log_softmax(logits, dim=1)[0, targets[index]]
                log_probability.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None and parameter.requires_grad:
                        fisher[name] += torch.square(parameter.grad.detach())
                sample_count += 1
            if sample_count >= max_samples:
                break

        self.model.zero_grad(set_to_none=True)
        if sample_count == 0:
            return {name: value.detach().cpu() for name, value in fisher.items()}

        return {name: (value / sample_count).detach().cpu() for name, value in fisher.items()}


def build_ewc(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    ewc_lambda: float,
    fisher_samples: int,
) -> EWCStrategy:
    return EWCStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        ewc_lambda=ewc_lambda,
        fisher_samples=fisher_samples,
    )
