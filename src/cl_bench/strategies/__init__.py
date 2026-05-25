from __future__ import annotations

import torch
from torch import nn

from cl_bench.config import ExperimentConfig
from cl_bench.strategies.agem import AGEMStrategy, build_agem
from cl_bench.strategies.baseline import BaselineStrategy, build_baseline
from cl_bench.strategies.derpp import DERPPStrategy, build_derpp
from cl_bench.strategies.er_ace import ERACEStrategy, build_er_ace
from cl_bench.strategies.ewc import EWCStrategy, build_ewc
from cl_bench.strategies.gdumb import GDumbStrategy, build_gdumb
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
)


def create_strategy(config: ExperimentConfig, model: nn.Module, device: torch.device) -> Strategy:
    method = config.method.lower().replace("-", "_")
    if method == "baseline":
        return build_baseline(model, device, config.learning_rate)
    if method == "ewc":
        return build_ewc(
            model,
            device,
            learning_rate=config.learning_rate,
            ewc_lambda=config.ewc_lambda,
            fisher_samples=config.fisher_samples,
        )
    if method == "replay":
        return build_replay(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            replay_batch_size=config.replay_batch_size,
            replay_loss_weight=config.replay_loss_weight,
            seed=config.seed,
        )
    if method == "er_ace":
        return build_er_ace(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            replay_batch_size=config.replay_batch_size,
            replay_loss_weight=config.replay_loss_weight,
            task_classes=[
                [int(label) for label in task.classes]
                for task in config.tasks
                if task.classes != "all"
            ],
            seed=config.seed,
        )
    if method == "derpp":
        return build_derpp(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            replay_batch_size=config.replay_batch_size,
            alpha=config.derpp_alpha,
            beta=config.derpp_beta,
            seed=config.seed,
        )
    if method == "gdumb":
        return build_gdumb(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            memory_epochs=config.gdumb_epochs,
            batch_size=config.batch_size,
            seed=config.seed,
        )
    if method == "agem":
        return build_agem(
            model,
            device,
            learning_rate=config.learning_rate,
            buffer_size=config.replay_buffer_size,
            memory_batch_size=config.agem_memory_batch_size,
            seed=config.seed,
        )
    if method == "lwf":
        return build_lwf(
            model,
            device,
            learning_rate=config.learning_rate,
            alpha=config.lwf_alpha,
            temperature=config.lwf_temperature,
        )
    raise ValueError(f"Unknown continual-learning method: {config.method}")


__all__ = [
    "AGEMStrategy",
    "BaselineStrategy",
    "DERPPStrategy",
    "ERACEStrategy",
    "EWCStrategy",
    "GDumbStrategy",
    "LwFStrategy",
    "ReplayStrategy",
    "Strategy",
    "create_strategy",
]
