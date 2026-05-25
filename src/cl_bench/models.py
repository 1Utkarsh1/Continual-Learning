from __future__ import annotations

from functools import reduce
from operator import mul

import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(_num_features(input_shape), num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class MLP(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = _num_features(input_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SmallCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"SmallCNN expects CHW input, got {input_shape}.")
        channels, height, width = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        pooled_height = max(1, height // 4)
        pooled_width = max(1, width // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_height * pooled_width, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(inputs) + self.shortcut(inputs))


class CifarConvNet(nn.Module):
    """Compact residual CNN tuned for 32x32 RGB continual-learning benchmarks."""

    def __init__(self, input_shape: tuple[int, ...], num_classes: int):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"CifarConvNet expects CHW input, got {input_shape}.")
        channels, _, _ = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False),
            _norm(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


class CifarResNet18(nn.Module):
    """ResNet-18 variant with a CIFAR stem and no initial max-pool."""

    def __init__(self, input_shape: tuple[int, ...], num_classes: int):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"CifarResNet18 expects CHW input, got {input_shape}.")
        channels, _, _ = input_shape
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.pool,
        )
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(self.features(inputs), 1)
        return self.classifier(features)


def get_model(model_name: str, input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    name = model_name.lower().replace("-", "_")
    if name == "linear":
        return LinearClassifier(input_shape, num_classes)
    if name == "mlp":
        return MLP(input_shape, num_classes)
    if name in {"small_cnn", "cnn"}:
        return SmallCNN(input_shape, num_classes)
    if name == "cifar_convnet":
        return CifarConvNet(input_shape, num_classes)
    if name in {"resnet18_cifar", "cifar_resnet18"}:
        return CifarResNet18(input_shape, num_classes)
    raise ValueError(f"Unknown model architecture: {model_name}")


def _num_features(input_shape: tuple[int, ...]) -> int:
    return int(reduce(mul, input_shape, 1))


def _norm(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels)
