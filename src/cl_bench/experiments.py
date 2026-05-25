from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch

from cl_bench.config import BenchmarkResult, ExperimentConfig, dump_config
from cl_bench.datasets import build_task_loaders
from cl_bench.metrics import compute_forgetting, matrix_to_jsonable, summarize_accuracy
from cl_bench.models import get_model
from cl_bench.strategies import create_strategy
from cl_bench.tracking import ExperimentTracker, MLflowRunLogger, create_run_dir, git_commit


def run_experiment(config: ExperimentConfig, repo_dir: str | Path | None = None) -> BenchmarkResult:
    set_seed(config.seed)
    device = resolve_device(config.device)
    tasks, input_shape, num_classes = build_task_loaders(config)
    model = get_model(config.model, input_shape=input_shape, num_classes=num_classes).to(device)
    strategy = create_strategy(config, model, device)

    run_dir = create_run_dir(config.output_dir, config.name, config.method)
    tracker = ExperimentTracker(run_dir)
    config_path = run_dir / "config.yaml"
    dump_config(config, config_path)

    commit = git_commit(repo_dir or Path.cwd())
    metadata = {
        "benchmark": config.name,
        "method": config.method,
        "seed": config.seed,
        "device": str(device),
        "git_commit": commit,
    }
    tracker.write_json("run_metadata.json", metadata)
    mlflow_enabled = config.tracking.lower() in {"mlflow", "both"}

    with MLflowRunLogger(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.mlflow_experiment,
        run_name=f"{config.name}_{config.method}_seed_{config.seed}",
        enabled=mlflow_enabled,
    ) as mlflow_logger:
        mlflow_logger.log_params(config.to_dict())
        mlflow_logger.set_tags(metadata)
        mlflow_logger.log_environment()

        start_time = time.perf_counter()
        accuracy_matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=float)

        for task_id, task in enumerate(tasks):
            tracker.log_event(
                {
                    "event": "task_started",
                    "task_id": task_id,
                    "task_name": task.name,
                    "classes": task.classes,
                }
            )
            history = strategy.train_task(
                task.train_loader,
                task.val_loader,
                task_id=task_id,
                epochs=config.epochs,
            )
            for epoch_metrics in history:
                tracker.log_event({"event": "epoch_finished", **epoch_metrics})
                mlflow_logger.log_metrics(
                    {
                        f"task_{task_id}_{key}": value
                        for key, value in epoch_metrics.items()
                        if key not in {"task_id", "epoch"}
                    },
                    step=task_id * config.epochs + int(epoch_metrics["epoch"]),
                )

            for eval_task_id in range(task_id + 1):
                eval_task = tasks[eval_task_id]
                metrics = strategy.evaluate(eval_task.test_loader)
                accuracy_matrix[task_id, eval_task_id] = metrics["accuracy"]
                tracker.log_event(
                    {
                        "event": "evaluation",
                        "after_task_id": task_id,
                        "eval_task_id": eval_task_id,
                        "eval_task_name": eval_task.name,
                        **metrics,
                    }
                )
                mlflow_logger.log_metrics(
                    {
                        f"eval_after_{task_id}_task_{eval_task_id}_accuracy": metrics["accuracy"],
                        f"eval_after_{task_id}_task_{eval_task_id}_loss": metrics["loss"],
                    },
                    step=task_id,
                )

        runtime_seconds = time.perf_counter() - start_time
        forgetting_matrix = compute_forgetting(accuracy_matrix)
        summary = summarize_accuracy(accuracy_matrix)
        summary.update(
            {
                "runtime_seconds": runtime_seconds,
                "seed": config.seed,
                "num_tasks": len(tasks),
                "model": config.model,
                "replay_buffer_size": config.replay_buffer_size,
                "replay_batch_size": config.replay_batch_size,
            }
        )

        tracker.write_json("accuracy_matrix.json", matrix_to_jsonable(accuracy_matrix))
        tracker.write_json("forgetting_matrix.json", matrix_to_jsonable(forgetting_matrix))
        tracker.write_matrix_csv("accuracy_matrix.csv", accuracy_matrix)
        tracker.write_matrix_csv("forgetting_matrix.csv", forgetting_matrix)

        if config.save_checkpoint:
            strategy.save_checkpoint(run_dir / "checkpoints" / "final_model.pt")

        metrics_path = tracker.write_json(
            "metrics.json",
            {
                "benchmark": config.name,
                "method": config.method,
                "task_names": [task.name for task in tasks],
                "summary": summary,
                "accuracy_matrix": matrix_to_jsonable(accuracy_matrix),
                "forgetting_matrix": matrix_to_jsonable(forgetting_matrix),
                "runtime_seconds": runtime_seconds,
                "seed": config.seed,
                "git_commit": commit,
            },
        )
        mlflow_logger.log_metrics(summary)
        mlflow_logger.log_artifacts(run_dir)

    return BenchmarkResult(
        run_dir=run_dir,
        method=config.method,
        task_names=[task.name for task in tasks],
        accuracy_matrix=matrix_to_jsonable(accuracy_matrix),
        forgetting_matrix=matrix_to_jsonable(forgetting_matrix),
        summary=summary,
        metrics_path=metrics_path,
        config_path=config_path,
        runtime_seconds=runtime_seconds,
        git_commit=commit,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    return device
