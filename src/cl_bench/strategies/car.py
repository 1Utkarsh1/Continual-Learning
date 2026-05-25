from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cl_bench.strategies.base import ContinualLearningStrategy
from cl_bench.strategies.replay import BalancedReplayBuffer


@dataclass(frozen=True)
class CARWeights:
    logit_anchor: float
    replay_ce: float
    feature_anchor: float
    prototype_anchor: float


class BiasTemperatureCalibration(nn.Module):
    """Small post-task calibration head used for old/new class bias correction."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(min=0.05, max=20.0)
        return logits / temperature + self.bias

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.detach().exp().clamp(min=0.05, max=20.0).item())


class CARStrategy(ContinualLearningStrategy):
    """Calibrated Anchor Replay for class-incremental continual learning."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        buffer_size: int,
        replay_batch_size: int,
        task_classes: list[list[int]],
        num_classes: int,
        weights: CARWeights,
        calibration_epochs: int,
        calibration_lr: float,
        calibration_weight_decay: float,
        replay_augment: bool,
        use_current_task_mask: bool,
        seed: int,
    ):
        super().__init__(model=model, device=device, learning_rate=learning_rate)
        self.buffer = BalancedReplayBuffer(capacity=buffer_size, seed=seed)
        self.replay_batch_size = replay_batch_size
        self.task_classes = task_classes
        self.weights = weights
        self.calibration_epochs = calibration_epochs
        self.calibration_lr = calibration_lr
        self.calibration_weight_decay = calibration_weight_decay
        self.replay_augment = replay_augment
        self.use_current_task_mask = use_current_task_mask
        self.calibrator = BiasTemperatureCalibration(num_classes).to(device)
        self.class_prototypes: dict[int, torch.Tensor] = {}
        self._last_batch_features: torch.Tensor | None = None
        self._last_batch_logits: torch.Tensor | None = None
        self.last_calibration_loss = 0.0

    def compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        logits, features = _forward_with_features(self.model, inputs)
        self._last_batch_logits = logits.detach().cpu()
        self._last_batch_features = features.detach().cpu()

        current_logits = logits
        if self.use_current_task_mask and task_id < len(self.task_classes):
            current_logits = _mask_to_classes(logits, self.task_classes[task_id])
        ce_loss = self.criterion(current_logits, targets)

        replay_ce_loss = torch.zeros((), device=self.device)
        logit_anchor_loss = torch.zeros((), device=self.device)
        feature_anchor_loss = torch.zeros((), device=self.device)
        prototype_anchor_loss = torch.zeros((), device=self.device)

        if len(self.buffer) > 0 and self.replay_batch_size > 0:
            samples = self.buffer.sample_samples(
                self.replay_batch_size,
                require_logits=True,
                require_features=True,
            )
            replay_inputs = torch.stack([sample.inputs for sample in samples]).to(self.device)
            replay_inputs = (
                _augment_replay_batch(replay_inputs) if self.replay_augment else replay_inputs
            )
            replay_targets = torch.tensor(
                [sample.target for sample in samples],
                dtype=torch.long,
                device=self.device,
            )
            stored_logits = torch.stack(
                [sample.logits for sample in samples if sample.logits is not None]
            ).to(self.device)
            stored_features = torch.stack(
                [sample.features for sample in samples if sample.features is not None]
            ).to(self.device)

            replay_logits, replay_features = _forward_with_features(self.model, replay_inputs)
            replay_ce_loss = self.criterion(replay_logits, replay_targets)
            logit_anchor_loss = F.mse_loss(replay_logits, stored_logits)
            feature_anchor_loss = F.mse_loss(replay_features, stored_features)
            prototypes = self._prototype_targets(replay_targets)
            if prototypes is not None:
                prototype_anchor_loss = F.mse_loss(replay_features, prototypes)

        loss = (
            ce_loss
            + self.weights.replay_ce * replay_ce_loss
            + self.weights.logit_anchor * logit_anchor_loss
            + self.weights.feature_anchor * feature_anchor_loss
            + self.weights.prototype_anchor * prototype_anchor_loss
        )
        return (
            loss,
            logits,
            {
                "ce_loss": float(ce_loss.detach().item()),
                "car_replay_ce_loss": float(replay_ce_loss.detach().item()),
                "car_logit_anchor_loss": float(logit_anchor_loss.detach().item()),
                "car_feature_anchor_loss": float(feature_anchor_loss.detach().item()),
                "car_prototype_anchor_loss": float(prototype_anchor_loss.detach().item()),
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
        if self._last_batch_logits is None or self._last_batch_features is None:
            with torch.no_grad():
                batch_logits, batch_features = _forward_with_features(self.model, inputs)
            self._last_batch_logits = batch_logits.detach().cpu()
            self._last_batch_features = batch_features.detach().cpu()
        self.buffer.add_batch(
            inputs,
            targets,
            logits=self._last_batch_logits,
            features=self._last_batch_features,
        )
        self._last_batch_logits = None
        self._last_batch_features = None

    def after_task(self, train_loader: DataLoader, task_id: int) -> None:
        del train_loader, task_id
        self._refresh_buffer_anchors()
        self._fit_calibration()

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        self.calibrator.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.calibrator(self.model(inputs))
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

    def run_summary(self) -> dict[str, float]:
        return {
            "car_calibration_temperature": self.calibrator.temperature,
            "car_calibration_loss": self.last_calibration_loss,
            "car_num_prototypes": float(len(self.class_prototypes)),
        }

    def extra_state_dict(self) -> dict[str, Any]:
        return {
            "buffer_inputs": [sample.inputs for sample in self.buffer.samples],
            "buffer_targets": [sample.target for sample in self.buffer.samples],
            "buffer_logits": [sample.logits for sample in self.buffer.samples],
            "buffer_features": [sample.features for sample in self.buffer.samples],
            "seen_count": self.buffer.seen_count,
            "class_counts": dict(self.buffer.class_counts()),
            "calibration_temperature": self.calibrator.temperature,
            "calibration_bias": self.calibrator.bias.detach().cpu().tolist(),
            "weights": self.weights.__dict__,
        }

    def _prototype_targets(self, targets: torch.Tensor) -> torch.Tensor | None:
        prototypes: list[torch.Tensor] = []
        for target in targets.detach().cpu().tolist():
            prototype = self.class_prototypes.get(int(target))
            if prototype is None:
                return None
            prototypes.append(prototype)
        return torch.stack(prototypes).to(self.device)

    def _refresh_buffer_anchors(self) -> None:
        if len(self.buffer) == 0:
            return

        self.model.eval()
        feature_sums: dict[int, torch.Tensor] = defaultdict(torch.Tensor)
        feature_counts: dict[int, int] = defaultdict(int)
        with torch.no_grad():
            for sample in self.buffer.samples:
                inputs = sample.inputs.unsqueeze(0).to(self.device)
                logits, features = _forward_with_features(self.model, inputs)
                sample.logits = logits.squeeze(0).detach().cpu()
                sample.features = features.squeeze(0).detach().cpu()
                label = int(sample.target)
                if feature_counts[label] == 0:
                    feature_sums[label] = sample.features.clone()
                else:
                    feature_sums[label] = feature_sums[label] + sample.features
                feature_counts[label] += 1

        self.class_prototypes = {
            label: feature_sums[label] / float(count)
            for label, count in feature_counts.items()
            if count > 0
        }

    def _fit_calibration(self) -> None:
        if len(self.buffer) == 0 or self.calibration_epochs <= 0:
            return

        inputs, targets = self.buffer.tensors()
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=min(256, max(1, len(dataset))), shuffle=True)
        optimizer = torch.optim.AdamW(
            self.calibrator.parameters(),
            lr=self.calibration_lr,
            weight_decay=self.calibration_weight_decay,
        )
        self.model.eval()
        self.calibrator.train()
        last_loss = 0.0
        for _ in range(self.calibration_epochs):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                with torch.no_grad():
                    logits = self.model(batch_inputs)
                optimizer.zero_grad(set_to_none=True)
                calibrated_logits = self.calibrator(logits)
                loss = self.criterion(calibrated_logits, batch_targets)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().item())
        self.last_calibration_loss = last_loss


def _forward_with_features(
    model: nn.Module, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    features_module = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    if features_module is not None and classifier is not None:
        features = torch.flatten(features_module(inputs), 1)
        logits = classifier(features)
        return logits, features

    logits = model(inputs)
    return logits, logits.detach() if logits.requires_grad else logits


def _mask_to_classes(logits: torch.Tensor, classes: list[int]) -> torch.Tensor:
    masked_logits = logits.clone()
    allowed = torch.zeros(logits.size(1), dtype=torch.bool, device=logits.device)
    allowed[torch.tensor(classes, dtype=torch.long, device=logits.device)] = True
    masked_logits[:, ~allowed] = -1e9
    return masked_logits


def _augment_replay_batch(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 4 or inputs.size(-1) < 8 or inputs.size(-2) < 8:
        return inputs

    batch_size, _, height, width = inputs.shape
    padded = F.pad(inputs, (4, 4, 4, 4), mode="reflect")
    max_top = padded.size(-2) - height
    max_left = padded.size(-1) - width
    crops = []
    for index in range(batch_size):
        top = int(torch.randint(0, max_top + 1, (1,), device=inputs.device).item())
        left = int(torch.randint(0, max_left + 1, (1,), device=inputs.device).item())
        crop = padded[index : index + 1, :, top : top + height, left : left + width]
        if bool(torch.rand((), device=inputs.device) < 0.5):
            crop = torch.flip(crop, dims=(-1,))
        crops.append(crop)
    return torch.cat(crops, dim=0)


def build_car(
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    buffer_size: int,
    replay_batch_size: int,
    task_classes: list[list[int]],
    num_classes: int,
    logit_anchor_weight: float,
    replay_ce_weight: float,
    feature_anchor_weight: float,
    prototype_anchor_weight: float,
    calibration_epochs: int,
    calibration_lr: float,
    calibration_weight_decay: float,
    replay_augment: bool,
    use_current_task_mask: bool,
    seed: int,
) -> CARStrategy:
    return CARStrategy(
        model=model,
        device=device,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        replay_batch_size=replay_batch_size,
        task_classes=task_classes,
        num_classes=num_classes,
        weights=CARWeights(
            logit_anchor=logit_anchor_weight,
            replay_ce=replay_ce_weight,
            feature_anchor=feature_anchor_weight,
            prototype_anchor=prototype_anchor_weight,
        ),
        calibration_epochs=calibration_epochs,
        calibration_lr=calibration_lr,
        calibration_weight_decay=calibration_weight_decay,
        replay_augment=replay_augment,
        use_current_task_mask=use_current_task_mask,
        seed=seed,
    )
