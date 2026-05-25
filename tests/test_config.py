from __future__ import annotations

from cl_bench.config import ExperimentConfig, load_config, load_config_with_overrides


def test_load_named_smoke_config() -> None:
    config = load_config("smoke")

    assert isinstance(config, ExperimentConfig)
    assert config.name == "smoke"
    assert config.method == "baseline"
    assert config.tasks[0].classes == [0, 1]
    assert config.tasks[1].dataset == "synthetic"


def test_load_real_data_quick_config() -> None:
    config = load_config("split_mnist_quick")

    assert config.name == "split_mnist_quick"
    assert config.model == "small_cnn"
    assert config.epochs == 3
    assert len(config.tasks) == 5
    assert all(task.dataset == "mnist" for task in config.tasks)
    assert all(task.train_limit == 600 for task in config.tasks)


def test_load_cifar_headline_config_and_overrides() -> None:
    config = load_config("split_cifar10_headline")

    assert config.name == "split_cifar10_headline"
    assert config.method == "derpp"
    assert config.model == "cifar_convnet"
    assert config.tracking == "both"
    assert config.epochs == 5
    assert config.replay_buffer_size == 5000
    assert config.replay_batch_size == 256
    assert config.replay_loss_weight == 3.0
    assert config.derpp_alpha == 0.1
    assert config.derpp_beta == 2.0
    assert len(config.tasks) == 5
    assert all(task.dataset == "cifar10" for task in config.tasks)

    overridden = load_config_with_overrides(
        "split_cifar10_headline",
        ["method=agem", "training.epochs=1", "tracking.mode=json"],
    )
    assert overridden.method == "agem"
    assert overridden.epochs == 1
    assert overridden.tracking == "json"


def test_nested_training_and_strategy_values_are_parsed() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "unit",
            "method": "ewc",
            "training": {"epochs": 3, "batch_size": 12, "learning_rate": 0.02},
            "strategy": {"ewc_lambda": 123.0, "fisher_samples": 7},
            "tasks": [
                {
                    "name": "a",
                    "dataset": "synthetic",
                    "classes": [0, 1],
                }
            ],
        }
    )

    assert config.epochs == 3
    assert config.batch_size == 12
    assert config.learning_rate == 0.02
    assert config.ewc_lambda == 123.0
    assert config.fisher_samples == 7
