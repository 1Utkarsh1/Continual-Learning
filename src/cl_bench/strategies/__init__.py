from __future__ import annotations

import torch
from torch import nn

from cl_bench.config import ExperimentConfig
from cl_bench.strategies.agem import AGEMStrategy, build_agem
from cl_bench.strategies.baseline import BaselineStrategy, build_baseline
from cl_bench.strategies.car import CARStrategy, build_car
from cl_bench.strategies.derpp import DERPPStrategy, build_derpp
from cl_bench.strategies.er_ace import ERACEStrategy, build_er_ace
from cl_bench.strategies.ewc import EWCStrategy, build_ewc
from cl_bench.strategies.gdumb import GDumbStrategy, build_gdumb
from cl_bench.strategies.joint import JointTrainingStrategy, build_joint
from cl_bench.strategies.lwf import LwFStrategy, build_lwf
from cl_bench.strategies.replay import ReplayStrategy, build_replay

Strategy = (
    BaselineStrategy
    | EWCStrategy
    | ReplayStrategy
    | LwFStrategy
    | DERPPStrategy
    | AGEMStrategy
    | ERACEStrategy
    | GDumbStrategy
    | CARStrategy
    | JointTrainingStrategy
)


def create_strategy(config: ExperimentConfig, model: nn.Module, device: torch.device) -> Strategy:
    method = config.method.lower().replace("-", "_")
    task_classes = _task_classes(config)
    if method == "baseline":
        strategy = build_baseline(model, device, config.learning_rate)
    elif method == "ewc":
        strategy = build_ewc(
            model,
            device,
            learning_rate=config.learning_rate,
            ewc_lambda=config.ewc_lambda,
            fisher_samples=config.fisher_samples,
        )
    elif method == "replay":
        strategy = build_replay(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            replay_batch_size=config.replay_batch_size,
            replay_loss_weight=config.replay_loss_weight,
            seed=config.seed,
        )
    elif method == "er_ace":
        strategy = build_er_ace(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            replay_batch_size=config.replay_batch_size,
            replay_loss_weight=config.replay_loss_weight,
            task_classes=task_classes,
            seed=config.seed,
        )
    elif method == "derpp":
        strategy = build_derpp(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            replay_batch_size=config.replay_batch_size,
            alpha=config.derpp_alpha,
            beta=config.derpp_beta,
            seed=config.seed,
        )
    elif method == "gdumb":
        strategy = build_gdumb(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            memory_epochs=config.gdumb_epochs,
            batch_size=config.batch_size,
            seed=config.seed,
        )
    elif method == "agem":
        strategy = build_agem(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            memory_batch_size=config.agem_memory_batch_size,
            seed=config.seed,
        )
    elif method == "lwf":
        strategy = build_lwf(
            model,
            device,
            learning_rate=config.learning_rate,
            alpha=config.lwf_alpha,
            temperature=config.lwf_temperature,
        )
    elif method == "joint":
        strategy = build_joint(
            model,
            device,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            seed=config.seed,
        )
    elif method in {"car", "bic", "icarl", "x_der_lite"}:
        use_calibration = method != "icarl"
        strategy = build_car(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            replay_batch_size=config.replay_batch_size,
            task_classes=task_classes,
            num_classes=_num_classes(config),
            logit_anchor_weight=0.0 if method == "bic" else config.car_logit_anchor_weight,
            replay_ce_weight=config.car_replay_ce_weight,
            feature_anchor_weight=0.0 if method == "bic" else config.car_feature_anchor_weight,
            prototype_anchor_weight=0.0 if method == "bic" else config.car_prototype_anchor_weight,
            calibration_epochs=config.car_calibration_epochs if use_calibration else 0,
            calibration_lr=config.car_calibration_lr,
            calibration_weight_decay=config.car_calibration_weight_decay,
            replay_augment=config.car_replay_augment,
            use_current_task_mask=config.car_use_current_task_mask,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown continual-learning method: {config.method}")

    strategy.configure_training(
        optimizer=config.optimizer,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        scheduler=config.scheduler,
        warmup_epochs=config.warmup_epochs,
        label_smoothing=config.label_smoothing,
    )
    return strategy


def _task_classes(config: ExperimentConfig) -> list[list[int]]:
    return [
        [int(label) for label in task.classes] for task in config.tasks if task.classes != "all"
    ]


def _num_classes(config: ExperimentConfig) -> int:
    explicit_classes = [
        int(label) for task in config.tasks if task.classes != "all" for label in task.classes
    ]
    if explicit_classes:
        return max(explicit_classes) + 1
    return 1000


__all__ = [
    "AGEMStrategy",
    "BaselineStrategy",
    "CARStrategy",
    "DERPPStrategy",
    "ERACEStrategy",
    "EWCStrategy",
    "GDumbStrategy",
    "JointTrainingStrategy",
    "LwFStrategy",
    "ReplayStrategy",
    "Strategy",
    "create_strategy",
]
