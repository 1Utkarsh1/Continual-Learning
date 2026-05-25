from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from cl_bench.strategies.base import ContinualLearningStrategy


@dataclass
class ReplaySample:
    inputs: torch.Tensor
    target: int
    logits: torch.Tensor | None = None
    features: torch.Tensor | None = None


class ReservoirReplayBuffer:
    """Bounded replay buffer using reservoir sampling over the observed stream."""

    def __init__(self, capacity: int, seed: int = 0):
        if capacity < 0:
            raise ValueError("Replay buffer capacity must be non-negative.")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.samples: list[ReplaySample] = []
        self.seen_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> None:
        logits_cpu = None if logits is None else logits.detach().cpu()
        features_cpu = None if features is None else features.detach().cpu()
        for index, (input_tensor, target) in enumerate(
            zip(inputs.detach().cpu(), targets.detach().cpu(), strict=True)
        ):
            sample_logits = None if logits_cpu is None else logits_cpu[index]
            sample_features = None if features_cpu is None else features_cpu[index]
            self.add(input_tensor, int(target.item()), sample_logits, sample_features)

    def add(
        self,
        inputs: torch.Tensor,
        target: int,
        logits: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> None:
        self.seen_count += 1
        if self.capacity == 0:
            return
        sample = ReplaySample(
            inputs=inputs.detach().cpu().clone(),
            target=int(target),
            logits=None if logits is None else logits.detach().cpu().clone(),
            features=None if features is None else features.detach().cpu().clone(),
        )
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            return

        replacement_index = self.rng.randrange(self.seen_count)
        if replacement_index < self.capacity:
            self.samples[replacement_index] = sample

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets, _ = self.sample_tensors(batch_size)
        return inputs, targets

    def sample_tensors(
        self,
        batch_size: int,
        require_logits: bool = False,
        require_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        chosen = self.sample_samples(
            batch_size,
            require_logits=require_logits,
            require_features=require_features,
        )
        inputs = torch.stack([sample.inputs for sample in chosen])
        targets = torch.tensor([sample.target for sample in chosen], dtype=torch.long)
        if require_logits:
            logits = torch.stack([sample.logits for sample in chosen if sample.logits is not None])
        elif require_features:
            logits = torch.stack(
                [sample.features for sample in chosen if sample.features is not None]
            )
        else:
            logits = None
        return inputs, targets, logits

    def sample_samples(
        self,
        batch_size: int,
        require_logits: bool = False,
        require_features: bool = False,
    ) -> list[ReplaySample]:
        candidates = [
            sample
            for sample in self.samples
            if (not require_logits or sample.logits is not None)
            and (not require_features or sample.features is not None)
        ]
        if not candidates:
            raise ValueError("Cannot sample from an empty replay buffer.")
        return self.rng.sample(candidates, k=min(batch_size, len(candidates)))


class BalancedReplayBuffer:
    """Class-balanced memory buffer for exemplar-only baselines such as GDumb."""

    def __init__(self, capacity: int, seed: int = 0):
        if capacity < 0:
            raise ValueError("Replay buffer capacity must be non-negative.")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.samples: list[ReplaySample] = []
        self.seen_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> None:
        logits_cpu = None if logits is None else logits.detach().cpu()
        features_cpu = None if features is None else features.detach().cpu()
        for index, (input_tensor, target) in enumerate(
            zip(inputs.detach().cpu(), targets.detach().cpu(), strict=True)
        ):
            sample_logits = None if logits_cpu is None else logits_cpu[index]
            sample_features = None if features_cpu is None else features_cpu[index]
            self.add(input_tensor, int(target.item()), sample_logits, sample_features)

    def add(
        self,
        inputs: torch.Tensor,
        target: int,
        logits: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> None:
        self.seen_count += 1
        if self.capacity == 0:
            return

        sample = ReplaySample(
            inputs=inputs.detach().cpu().clone(),
            target=int(target),
            logits=None if logits is None else logits.detach().cpu().clone(),
            features=None if features is None else features.detach().cpu().clone(),
        )
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            return

        counts = self.class_counts()
        target_count = counts.get(int(target), 0)
        max_count = max(counts.values(), default=0)
        if target_count >= max_count:
            return

        replacement_classes = [label for label, count in counts.items() if count == max_count]
        replacement_class = self.rng.choice(replacement_classes)
        replacement_indices = [
            index
            for index, existing_sample in enumerate(self.samples)
            if existing_sample.target == replacement_class
        ]
        self.samples[self.rng.choice(replacement_indices)] = sample

    def class_counts(self) -> Counter[int]:
        return Counter(sample.target for sample in self.samples)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.samples:
            raise ValueError("Cannot materialize an empty replay buffer.")
        inputs = torch.stack([sample.inputs for sample in self.samples])
        targets = torch.tensor([sample.target for sample in self.samples], dtype=torch.long)
        return inputs, targets

    def sample_samples(
        self,
        batch_size: int,
        require_logits: bool = False,
        require_features: bool = False,
    ) -> list[ReplaySample]:
        candidates = [
            sample
            for sample in self.samples
            if (not require_logits or sample.logits is not None)
            and (not require_features or sample.features is not None)
        ]
        if not candidates:
            raise ValueError("Cannot sample from an empty replay buffer.")
        return self.rng.sample(candidates, k=min(batch_size, len(candidates)))


class ReplayStrategy(ContinualLearningStrategy):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        buffer_size: int,
        replay_batch_size: int,
        replay_loss_weight: float,
        seed: int,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.buffer = ReservoirReplayBuffer(capacity=buffer_size, seed=seed)
        self.replay_batch_size = replay_batch_size
        self.replay_loss_weight = replay_loss_weight

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del task_id
        logits = self.model(inputs)
        ce_loss = self.criterion(logits, targets)
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
                "ce_loss": float(ce_loss.detach().item()),
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

    def after_task(self, train_loader: DataLoader, task_id: int) -> None:
        del train_loader, task_id

    def extra_state_dict(self) -> dict[str, object]:
        return {
            "buffer_inputs": [sample.inputs for sample in self.buffer.samples],
            "buffer_targets": [sample.target for sample in self.buffer.samples],
            "buffer_logits": [sample.logits for sample in self.buffer.samples],
            "seen_count": self.buffer.seen_count,
        }


def build_replay(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    buffer_size: int,
    replay_batch_size: int,
    replay_loss_weight: float,
    seed: int,
) -> ReplayStrategy:
    return ReplayStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        replay_batch_size=replay_batch_size,
        replay_loss_weight=replay_loss_weight,
        seed=seed,
    )
