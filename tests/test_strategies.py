from __future__ import annotations

import torch

from cl_bench.config import ExperimentConfig, TaskSpec
from cl_bench.datasets import build_task_loaders
from cl_bench.models import get_model
from cl_bench.strategies.agem import AGEMStrategy
from cl_bench.strategies.base import clone_state_dict
from cl_bench.strategies.baseline import BaselineStrategy
from cl_bench.strategies.derpp import DERPPStrategy
from cl_bench.strategies.er_ace import ERACEStrategy
from cl_bench.strategies.ewc import EWCStrategy
from cl_bench.strategies.gdumb import GDumbStrategy
from cl_bench.strategies.lwf import LwFStrategy
from cl_bench.strategies.replay import BalancedReplayBuffer


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="tiny",
        method="baseline",
        seed=5,
        model="linear",
        epochs=1,
        batch_size=8,
        eval_batch_size=16,
        learning_rate=0.05,
        val_fraction=0.2,
        tasks=[
            TaskSpec(
                name="a",
                dataset="synthetic",
                classes=[0, 1],
                samples_per_class=8,
                test_samples_per_class=4,
            )
        ],
    )


def test_strategy_lifecycle_trains_and_evaluates() -> None:
    config = _tiny_config()
    tasks, input_shape, num_classes = build_task_loaders(config)
    model = get_model("linear", input_shape, num_classes)
    strategy = BaselineStrategy(model=model, device=torch.device("cpu"), learning_rate=0.05)

    history = strategy.train_task(tasks[0].train_loader, tasks[0].val_loader, task_id=0, epochs=1)
    metrics = strategy.evaluate(tasks[0].test_loader)

    assert len(history) == 1
    assert strategy.current_task == 0
    assert strategy.seen_tasks == 1
    assert 0.0 <= metrics["accuracy"] <= 100.0


def test_clone_state_dict_does_not_alias_model_parameters() -> None:
    model = torch.nn.Linear(2, 2)
    cloned = clone_state_dict(model)

    with torch.no_grad():
        model.weight.add_(10.0)

    assert not torch.equal(cloned["weight"], model.state_dict()["weight"])


def test_ewc_fisher_is_deterministic_and_normalized() -> None:
    config = _tiny_config()
    tasks, input_shape, num_classes = build_task_loaders(config)
    model_a = get_model("linear", input_shape, num_classes)
    model_b = get_model("linear", input_shape, num_classes)
    model_b.load_state_dict(model_a.state_dict())

    strategy_a = EWCStrategy(model_a, torch.device("cpu"), 0.01, ewc_lambda=1.0, fisher_samples=4)
    strategy_b = EWCStrategy(model_b, torch.device("cpu"), 0.01, ewc_lambda=1.0, fisher_samples=4)

    fisher_a = strategy_a._estimate_fisher(tasks[0].train_loader)
    fisher_b = strategy_b._estimate_fisher(tasks[0].train_loader)

    assert fisher_a.keys() == fisher_b.keys()
    for name in fisher_a:
        assert torch.allclose(fisher_a[name], fisher_b[name])
        assert torch.all(fisher_a[name] >= 0)


def test_lwf_creates_frozen_teacher_after_task() -> None:
    config = _tiny_config()
    tasks, input_shape, num_classes = build_task_loaders(config)
    model = get_model("linear", input_shape, num_classes)
    strategy = LwFStrategy(model, torch.device("cpu"), 0.05, alpha=0.5, temperature=2.0)

    strategy.train_task(tasks[0].train_loader, tasks[0].val_loader, task_id=0, epochs=1)

    assert strategy.teacher_model is not None
    assert all(not parameter.requires_grad for parameter in strategy.teacher_model.parameters())


def test_derpp_stores_logits_and_uses_replay_components() -> None:
    config = _tiny_config()
    tasks, input_shape, num_classes = build_task_loaders(config)
    model = get_model("linear", input_shape, num_classes)
    strategy = DERPPStrategy(
        model,
        torch.device("cpu"),
        learning_rate=0.05,
        buffer_size=16,
        replay_batch_size=4,
        alpha=0.5,
        beta=1.0,
        seed=1,
    )
    inputs, targets = next(iter(tasks[0].train_loader))
    logits = strategy.model(inputs)

    strategy.observe_batch(inputs, targets, logits, task_id=0)
    loss, _, components = strategy.compute_loss(inputs, targets, task_id=1)

    assert len(strategy.buffer) == inputs.size(0)
    assert all(sample.logits is not None for sample in strategy.buffer.samples)
    assert loss.item() > 0.0
    assert "derpp_distillation_loss" in components
    assert "derpp_replay_ce_loss" in components


def test_agem_projects_conflicting_gradient() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    strategy = AGEMStrategy(
        model,
        torch.device("cpu"),
        learning_rate=0.1,
        buffer_size=4,
        memory_batch_size=2,
        seed=1,
    )
    strategy.buffer.add_batch(
        torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
        torch.tensor([0, 0]),
    )
    current_inputs = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    current_targets = torch.tensor([1, 1])

    loss, _, _ = strategy.compute_loss(current_inputs, current_targets, task_id=1)
    strategy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    strategy.after_backward(current_inputs, current_targets, task_id=1)

    assert strategy.last_gradient_dot < 0.0
    assert strategy.last_projection_applied == 1.0


def test_er_ace_masks_current_task_logits_and_replays_memory() -> None:
    config = _tiny_config()
    tasks, input_shape, num_classes = build_task_loaders(config)
    model = get_model("linear", input_shape, num_classes)
    strategy = ERACEStrategy(
        model,
        torch.device("cpu"),
        learning_rate=0.05,
        buffer_size=16,
        replay_batch_size=4,
        replay_loss_weight=1.0,
        task_classes=[[0, 1]],
        seed=1,
    )
    inputs, targets = next(iter(tasks[0].train_loader))
    strategy.observe_batch(inputs, targets, strategy.model(inputs), task_id=0)

    loss, _, components = strategy.compute_loss(inputs, targets, task_id=0)

    assert loss.item() > 0.0
    assert len(strategy.buffer) == inputs.size(0)
    assert "er_ace_current_ce_loss" in components
    assert "replay_loss" in components


def test_balanced_replay_buffer_rebalances_classes() -> None:
    buffer = BalancedReplayBuffer(capacity=4, seed=1)
    buffer.add_batch(torch.randn(4, 1, 2, 2), torch.tensor([0, 0, 0, 0]))
    buffer.add_batch(torch.randn(4, 1, 2, 2), torch.tensor([1, 1, 1, 1]))

    counts = buffer.class_counts()

    assert counts[0] == 2
    assert counts[1] == 2


def test_gdumb_collects_balanced_memory_and_trains_after_task() -> None:
    config = _tiny_config()
    tasks, input_shape, num_classes = build_task_loaders(config)
    model = get_model("linear", input_shape, num_classes)
    strategy = GDumbStrategy(
        model,
        torch.device("cpu"),
        learning_rate=0.05,
        buffer_size=16,
        memory_epochs=2,
        batch_size=8,
        seed=1,
    )

    strategy.train_task(tasks[0].train_loader, tasks[0].val_loader, task_id=0, epochs=1)
    metrics = strategy.evaluate(tasks[0].test_loader)

    assert len(strategy.buffer) > 0
    assert 0.0 <= metrics["accuracy"] <= 100.0
